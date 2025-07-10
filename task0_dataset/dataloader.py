#!/usr/bin/env python3
"""
captchaloader.py - flexible PyTorch dataloaders for the new CAPTCHA dataset
──────────────────────────────────────────────────────────────────────────────
Features
========
● works with *train* **and** *test* splits that contain                   
      train/<difficulty>/<class>/captcha/img.png                              
● difficulty filter      - pick any subset of {"easy","medium","hard"}   
● length     filter      - pick any subset of character lengths           
● few-shot sampling      - fixed #images per class *or* total #images     
                            (spread equally over difficulties ▸ lengths ▸ classes)
● reproducible RNG via `seed`                                             
● helper loaders:                                                         
      get_data_loader                      
      special_train_loader / special_test_loader (100 class task)         
========
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ────────────────────────────────────────────────────────────────────────────
# Core dataset
# ────────────────────────────────────────────────────────────────────────────
class CaptchaDataset(Dataset):
    """
    Generic Dataset for *either* train or test split of the new dataset.

    Parameters
    ----------
    root          Path to *train* or *test* directory **containing the
                  difficulty sub-folders** (easy / medium / hard)
                  e.g.  task0_dataset/train   or   task0_dataset/test
    difficulties  iterable of difficulties to include  {"easy","medium","hard"}
    lengths       iterable of character lengths to include  (3…10)
    shots         fixed #images per class (few-shot); ignored if None
    total_images  global budget - sample exactly this many images distributed
                  equally across [difficulty ↘ length ↘ class]. Overrides
                  `shots` if both supplied.
    classes_subset optional container of class-names - keep only those
    transform     torchvision transform (default = ToTensor)
    seed          RNG seed for deterministic sampling
    """

    def __init__(
        self,
        root: str | Path,
        *,
        difficulties: Iterable[str] = ("easy", "medium", "hard"),
        lengths: Iterable[int] = range(3, 11),
        shots: int | None = None,
        total_images: int | None = None,
        classes_subset: set[str] | None = None,
        transform=None,
        seed: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.difficulties = [d.lower() for d in difficulties]
        self.lengths = set(lengths)
        self.transform = transform or transforms.ToTensor()
        self.rng = random.Random(seed)

        if shots is not None and total_images is not None:
            raise ValueError("Pass *either* shots *or* total_images, not both.")

        # ── 1) collect all images that satisfy the filters ─────────────────
        # Type‑annotates bucket as a dict whose keys are 3‑tuples of (str, int, str), and values are list[Path].
        # Initializes bucket to be a defaultdict(list), i.e. time you access a key that doesn’t yet exist, you get back a fresh empty list.
        # dict[(diff, length, cls)] = [img1.png, img2.png, ...]
        # E.g. bucket looks like:
        # {
        #   ("easy", 5, "apple"): [
        #       Path("root/easy/apple/img1.png"),
        #       Path("root/easy/apple/img2.png"),
        #       Path("root/easy/apple/img3.png"),
        #   ]
        # }
        bucket: dict[tuple[str, int, str], list[Path]] = defaultdict(list)

        for diff in self.difficulties:
            diff_dir = self.root / diff
            if not diff_dir.is_dir():
                raise FileNotFoundError(f"Missing difficulty folder: {diff_dir}")

            for cls_dir in diff_dir.iterdir():
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name
                if len(cls) not in self.lengths:
                    continue
                if classes_subset is not None and cls not in classes_subset:
                    continue
                
                captcha_dir = cls_dir / "captcha"
                imgs = sorted(captcha_dir.glob("*.png"))
                if not imgs:
                    continue
                bucket[(diff, len(cls), cls)].extend(imgs)

        if not bucket:
            raise RuntimeError(
                "No images found - check 'root', 'difficulties', 'lengths', "
                "and 'classes_subset' filters."
            )

        # ── 2) build label↔index mapping (classes are case‑sensitive strings) ──
        self.classes = sorted({key[2] for key in bucket})
        # classes_to_idx = { c1: 0, c2: 1, c3: 2, ... }
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # idx_to_classes = { 0: c1, 1: c2, 2: c3, ... }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # ── 3) sample images  ──────────────────────────────────────────────
        self.samples: list[tuple[Path, int]] = []

        if total_images is None:
            # simple mode: optional few‑shot *shots* per class
            # since each length variant has same amount of classes, we are sampling equally from each len
            # similarly for each difficulty
            for (diff, length, cls), imgs in bucket.items():
                imgs = imgs.copy()
                if shots is not None:
                    self.rng.shuffle(imgs)
                    imgs = imgs[: shots]
                self.samples.extend((p, (diff, length, self.class_to_idx[cls]), cls) for p in imgs)
        else:
            # advanced mode: equal distribution over diff ▸ length ▸ class
            combos = sorted(bucket.keys())  # keys of bucket i.e. (diff,length,cls)
            # 1st tier  - difficulty
            # diff_groups = {
            #     'easy': [('easy', 5, 'apple')],
            #     'medium': [('medium', 5, 'banana')],
            #     'hard': [('hard', 6, 'cherry')]
            # }
            diff_groups: dict[str, list[tuple[str, int, str]]] = defaultdict(list) # NOTE No need to reset as not in loop
            for combo in combos:
                diff_groups[combo[0]].append(combo)

            imgs_per_diff = total_images // len(self.difficulties)
            leftovers = total_images - imgs_per_diff * len(self.difficulties)

            for diff in sorted(self.difficulties):
                diff_budget = imgs_per_diff + (1 if leftovers > 0 else 0)
                leftovers -= 1 if leftovers > 0 else 0

                # 2nd tier - length within this difficulty
                # length_groups = {
                #     '5': [('easy', 5, 'apple'), ('easy', 5, 'banana')],    -> only one difficulty at a time 
                #     '6': [('easy', 6, 'apples'), ('easy', 5, 'bananas')],  -> only one difficulty at a time 
                # }
                length_groups: dict[int, list[tuple[str, int, str]]] = defaultdict(list) # NOTE new dict reated. Auto reset for each difficulty
                for combo in diff_groups[diff]: 
                    length_groups[combo[1]].append(combo)  # for a given difficulty

                imgs_per_len = diff_budget // len(self.lengths)
                len_leftover = diff_budget - imgs_per_len * len(self.lengths)

                for length in sorted(self.lengths):
                    len_budget = imgs_per_len + (1 if len_leftover > 0 else 0)
                    len_leftover -= 1 if len_leftover > 0 else 0

                    # combos_here = selecting all (diff, length, cls) tuples of a particular "length"
                    # for a given difficulty (because we are iterating over difficulties) and length
                    combos_here = length_groups[length]   # NOTE automatically resets for each length 
                    if not combos_here:
                        continue
                    # diff, len fixed => len(combos_here) is the number of classes
                    imgs_per_class = max(1, len_budget // len(combos_here))  
                    tmp_leftover = len_budget - imgs_per_class * len(combos_here)

                    # combo = diff, length, cls all fixed
                    for combo in combos_here:
                        cls = combo[2]
                        imgs = bucket[combo].copy()
                        self.rng.shuffle(imgs)
                        n_take = min(imgs_per_class, len(imgs))
                        # distribute any leftover 1‑by‑1
                        if tmp_leftover > 0:
                            take_extra = min(1, len(imgs) - n_take)
                            n_take += take_extra
                            tmp_leftover -= take_extra
                        # samples = list of (path, diff, length, class_index, class_name) tuples
                        self.samples.extend(
                            (p, (diff, length, self.class_to_idx[cls]), cls) for p in imgs[: n_take]
                        )

        if not self.samples:
            raise RuntimeError("Sampling produced an empty dataset!")

    # ── Dataset API ────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, (diff, len, cls_idx), cls_name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # rotate easy images (rotate first then tensor) as all are identicle initially
        rotate = transforms.RandomRotation(
            degrees=(-15, 15),        # rotation range
            fill=(255, 255, 255)      # white fill for RGB
        )
        if diff == "easy": img = rotate(img)
        img = self.transform(img)
        # img, label
        return img, cls_idx
    
    # ── Validate Equal ──────────────────────────────────────────────────────
    def validate_equal(self) -> bool:
        """
        Validate that the dataset equal distribution acrosss difficulty, length, class.
        """
        # Empty dataset check
        if not self.samples:
            return False
        
        # Count occurrences
        cd, cl, cidx = {}, {}, {}
        for _, (diff, length, cls_idx), _ in self.samples:
            cd[diff] = cd.get(diff, 0) + 1 
            cl[length] = cl.get(length, 0) + 1
            cidx[cls_idx] = cidx.get(cls_idx, 0) + 1
        
        ## NOTE DEBUG (uncomment to always see mappings)
        # print(f"\n\nDifficulty counts: {cd}")
        # print(f"\nLength counts: {cl}")
        # print(f"\nClassses mapping: {cidx}")
        # print(f"\nClassses: {sorted(list(cidx.keys()))}\n\n")
        
        # Check if all counts are equal
        flag = True
        first_count = next(iter(cd.values()))
        if not all(count == first_count for count in cd.values()):
            print(f"\n\nDifficulty counts: {cd}")
            flag = False
        first_count = next(iter(cl.values()))
        if not all(count == first_count for count in cl.values()):
            print(f"\nLength counts: {cl}")
            flag = False
        first_count = next(iter(cidx.values()))
        if not all(count == first_count for count in cidx.values()):
            print(f"\nClassses mapping: {cidx}")
            print(f"\nClassses: {sorted(list(cidx.keys()))}\n\n")
            flag = False
        return flag
    

# ────────────────────────────────────────────────────────────────────────────
# Robust Dataloaders
# ────────────────────────────────────────────────────────────────────────────
def get_data_loader(
    *,
    root: str | Path = "task0_dataset/train",
    difficulties: Iterable[str] = ("easy", "medium", "hard"),
    lengths: Iterable[int] = (3, 4, 5, 6, 7, 8, 9, 10),
    shots: int | None = None,
    total_images: int | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    transform = None,
    num_workers: int = 4,
    validate: bool = False
):
    """
    Generic TRAIN or TEST loader depending on path.

    • Pass *either* shots (few-shot per class)  *or* total_images.
    • Returns (DataLoader, class_to_idx)  so the mapping can be reused for test.
    """
    ds = CaptchaDataset(
        root,
        difficulties=difficulties,
        lengths=lengths,
        shots=shots,
        total_images=total_images,
        transform=transform,
        seed=seed,
    )
    
    if validate: 
        print(f"Validation on {root}: {ds.validate_equal()}")
    
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dl, ds.class_to_idx


# ────────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ────────────────────────────────────────────────────────────────────────────

def _choose_balanced_classes(
    root: Path,
    difficulties: Sequence[str],
    lengths: Sequence[int],
    total_classes: int,
    seed: int = 0,
) -> set[str]:
    """
    Pick *total_classes* distinct class names, equally distributed over (difficulty x length).  
    Returns the selected class-name set.
    """
    rng = random.Random(seed)
    per_bucket = math.ceil(total_classes / (len(difficulties) * len(lengths)))

    # bucket: (difficulty, length) and collect class names
    buckets: dict[tuple[str, int], list[str]] = defaultdict(list)
    for diff in difficulties:
        for cls_dir in (root / diff).iterdir():
            if cls_dir.is_dir() and len(cls_dir.name) in lengths:
                buckets[(diff, len(cls_dir.name))].append(cls_dir.name)

    chosen: set[str] = set()
    for key, cls_list in buckets.items():
        rng.shuffle(cls_list)
        chosen.update(cls_list[: per_bucket])

    # trim to exact count (possible slight overshoot)
    chosen = set(list(chosen)[: total_classes])
    return chosen

def plot_images(
    data_loader,
    imgs_to_plot = 5,
    save = True
):
    """
    Plots (matplotlib) first imgs_to_plot from data_loader's first batch
    Optionally saves figure to current directory 
    """
    # Plot images
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    
    # Get images 
    imgs, labels = next(iter(data_loader))
    
    # Create a figure (W, H)
    plt.figure(figsize=(12, 3))  
    
    # Plot first 5 images
    for i in range(imgs_to_plot):
        img = imgs[i]  # shape: [C, H, W]
        img = F.to_pil_image(img)  # convert tensor to PIL image

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")

    plt.tight_layout()
    if save: 
        plt.savefig("preview_batch.png")  # save to current directory
        plt.close()  # optional: close the figure to free memory



if __name__ == "__main__":
  
    import torchvision.transforms as T
    # from captchaloader import (
    #     get_train_loader,
    #     get_test_loader,
    #     special_train_loader,
    #     special_test_loader,
    # )

    # Create image transformation (required)
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # [0 - 1] -> [-1, 1]
    ])
    # transform for input to a resnet model
    mean_255 = [int(m*255) for m in (0.485, 0.456, 0.406)] # post normalization 0s
    resnet_tfm = transforms.Compose([
        # Pad height 
        transforms.Pad((0,48,0,48), fill=tuple(mean_255), padding_mode='constant'),
        # Now crop a 224×224 patch from center of the 224×256 image
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ── Control total images vs shorts per class | Equally sampled across 
    train_loader, cls_to_idx = get_data_loader(
        # data
        root = "task1_classification/data/train",
        # select from
        difficulties = ["medium"],
        lengths      = [3, 4, 5, 6],
        # how many
        total_images = 10_000,   # overrides shots
        shots        = None,     # if total_images = None 
        # misc
        transform    = resnet_tfm,
        seed         = 123,
        batch_size   = 32,
        validate     = True
    )
    
    # Check shape and images
    imgs, labels = next(iter(train_loader))
    print("\n", imgs.shape, f"img.max: {imgs[0].max()}", f"img.min: {imgs[0].min()}" , "\n\n", labels[:10], "\n")
    
    # plot images
    plot_images(train_loader, save=True)




#### OLD 

# def get_test_loader(
#     *,
#     root: str | Path = "./test",
#     difficulties: Iterable[str] = ("easy", "medium", "hard"),
#     lengths: Iterable[int] = (3, 4, 5, 6, 7, 8, 9, 10),
#     batch_size: int = 32,
#     shuffle: bool = False,
#     transform=None,
#     num_workers: int = 4,
#     class_to_idx: dict[str, int] | None = None,
# ):
#     """
#     Generic TEST loader.

#     • Pass *either* shots (few-shot per class)  *or* total_images.
#     • If *class_to_idx* (from train) is provided, label indices stay consistent.
#     """
#     ds = CaptchaDataset(
#         root,
#         difficulties=difficulties,
#         lengths=lengths,
#         shots=None,
#         transform=transform,
#     )
#     if class_to_idx is not None:
#         ds.class_to_idx = class_to_idx
#         ds.idx_to_class = {v: k for k, v in class_to_idx.items()}
#         ds.samples = [
#             (p, class_to_idx[p.parent.name]) for (p, _) in ds.samples
#         ]

#     dl = DataLoader(
#         ds,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#     return dl

# def special_data_loader(
#     *,
#     train_root: str | Path = "task1_classification/data/train",
#     test_root: str | Path = "task1_classification/data/test",
#     total_classes: int = 100,
#     total_train_images: int | None = None,
#     total_test_images: int | None = 1000,
#     shots: int | None = None,
#     batch_size: int = 32,
#     seed: int = 0,
#     transform=None,
#     num_workers: int = 4,
# ):
#     """
#     • Select **100 classes** equally across  ⟨easy,medium⟩ x ⟨3,4,5,6⟩.
#     • Image sampling: pass *either* shots (per class) or total_images (global).
#     • Returns  (DataLoader, class_to_idx, chosen_class_set)
#     """
#     difficulties = ("easy", "medium")
#     lengths = (3, 4, 5, 6)
#     train_root = Path(train_root)
#     test_root = Path(test_root)

#     chosen_classes = _choose_balanced_classes(
#         train_root, difficulties, lengths, total_classes, seed
#     )

#     # Train
#     ds_tr = CaptchaDataset(
#         train_root,
#         difficulties=difficulties,
#         lengths=lengths,
#         shots=shots,
#         total_images=total_train_images,
#         classes_subset=chosen_classes,
#         transform=transform,
#         seed=seed,
#     )
#     dl_tr = DataLoader(
#         ds_tr,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
    
#     # Test 
#     ds_ts = CaptchaDataset(
#         test_root,
#         difficulties=difficulties,
#         lengths=lengths,
#         total_images=total_test_images,
#         classes_subset=chosen_classes,
#         transform=transform,
#         seed=seed,
#     )
#     dl_ts = DataLoader(
#         ds_ts,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
    
#     return dl_tr, dl_ts