#!/usr/bin/env python3
# =============================================================================
# dataset_fixed.py – CAPTCHA data-set generator (fixed for no clipping & minor vertical variation)
# =============================================================================
# ▸ Dual output: noisy “captcha” image + plain “clean” reference
# ▸ Medium/Hard: per-character random font, colour, size (minor variation),
#                horizontal sine-warp, noisy background, optional occluding lines
# ▸ 128×256 canvas, 8 shots per class, 3-10 char strings,
#   200 k train / 20 k test (no class overlap)
# =============================================================================

from __future__ import annotations

# Generic
import os
import sys
import math
import random
import string
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Custom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from task0_dataset.captcha import ImageCaptcha


# ────────────────────────────────────────────────────────────────────────────
# USER-ADJUSTABLE CONSTANTS
# ────────────────────────────────────────────────────────────────────────────

# Resolution
HEIGHT = 224
WIDTH = 224

# Difficulty level: "easy", "medium", or "hard"
DIFFICULTY: str = "medium"

# Paths
ROOT = Path("task1_classification/data")
SUBDIR_CAPTCHA = "captcha"
SUBDIR_CLEAN   = "clean"

# Lengths of CAPTCHA strings to generate
LENGTHS: Sequence[int] = range(3, 7)  # 3 … 6 => 4 variants 
# Images 
TRAIN_IMAGES = 5000   # 500
IMAGES_PER_CLASS = 50 # 5
TEST_IMAGES_PER_CLASS  = IMAGES_PER_CLASS // 5 # 1 
# Images per length varient = total / len variants (5k/4 = 1.25k)
# Number of classes (5k/50 = 100)

# Characters and Fonts
ALPHABET = string.ascii_letters + string.digits
FONT_PATHS = [
    "/home/user/ocr/task0_dataset/fonts/OpenSans_Condensed-Regular.ttf",
    "/home/user/ocr/task0_dataset/fonts/BebasNeue-Regular.ttf",
    "/home/user/ocr/task0_dataset/fonts/Royalacid.ttf",
]

# Captcha 
easy = ImageCaptcha(width = WIDTH, height = HEIGHT,
                    fonts=FONT_PATHS, font_sizes = (62, 70, 76)
)
medium = ImageCaptcha(
    # resolution
    width  = WIDTH,
    height = HEIGHT,
    # distortion
    character_offset_dx = (0, 4),
    character_offset_dy = (0, 6),
    character_rotate  = (-20, 20),
    character_warp_dx = (0.1, 0.3),
    character_warp_dy = (0.2, 0.3),
    word_space_probability =  0.35,
    word_offset_dx = 0.35,
    # fonts
    fonts      = FONT_PATHS, 
    font_sizes = (40, 48, 55),
    # difficulty
    difficulty = "medium", 
    max_gaussian_noise_rate=0.2
)
hard = ImageCaptcha(
    # resolution
    width  = WIDTH,
    height = HEIGHT,
    # distortion
    character_offset_dx = (1, 6),
    character_offset_dy = (1, 7),
    character_rotate  = (-30, 30),
    character_warp_dx = (0.2, 0.4),
    character_warp_dy = (0.3, 0.4),
    word_space_probability =  0.55,
    word_offset_dx = 0.45,
    # fonts
    fonts      = FONT_PATHS, 
    font_sizes = (40, 48, 55),
    # difficulty
    difficulty = "hard", 
    max_gaussian_noise_rate=0.4
)


random.seed(42)



# ────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

# ─────────────  Unqiue CAPTCHA strings  ─────────────
def _unique_strings(n: int, length: int, forbidden: set[str]) -> List[str]:
    """Return *n* unique random strings of given length not in *forbidden*."""
    out: set[str] = set()
    while len(out) < n:
        s = "".join(random.choices(ALPHABET, k=length))
        if s not in forbidden:
            out.add(s)
    return sorted(out)


# ─────────────  Split Helpers  ─────────────
def _classes_per_length(images_per_length: int) -> dict[int, int]:
    base = images_per_length // IMAGES_PER_CLASS
    rem  = images_per_length % IMAGES_PER_CLASS
    mapping = {l: base for l in LENGTHS}
    if rem > 0 and len(LENGTHS) > 0:
        for l in itertools.islice(itertools.cycle(LENGTHS), rem):
            mapping[l] += 1
    return mapping

TRAIN_CLASSES_PER_LENGTH = _classes_per_length(TRAIN_IMAGES // len(LENGTHS) if LENGTHS else 0)
print("\nClasses per length:", TRAIN_CLASSES_PER_LENGTH); print("")


# ─────────────  Main writer  ─────────────
def build_split(
    *,
    split_name: str,
    difficulty: str,
    classes_plan: dict[int, int],
    images_per_class: int = IMAGES_PER_CLASS,
    existing_vocab = None
) -> set[str]:
    
    # Paths 
    root = ROOT / split_name / difficulty
    root.mkdir(parents=True, exist_ok=True)
    labels_fp = (root / "labels.txt").open("w", encoding="utf8")

    # Prepare classes
    out_vocab = {}
    used_classes: set[str] = set()
    for length in sorted(LENGTHS):
        if existing_vocab is None:
            n_cls = classes_plan.get(length, 0)
            if n_cls == 0: continue
            new_cls = _unique_strings(n_cls, length,  used_classes)
            used_classes.update(new_cls)
            out_vocab[length] = list(new_cls)
        else: 
            new_cls = existing_vocab[length]

        pbar_desc = f"{split_name}-{difficulty}-len{length}"
        for cls in tqdm(new_cls, desc=pbar_desc, unit="cls"):
            cls_dir     = root / cls
            captcha_dir = cls_dir / SUBDIR_CAPTCHA
            clean_dir   = cls_dir / SUBDIR_CLEAN
            captcha_dir.mkdir(parents=True, exist_ok=True)
            clean_dir.mkdir(parents=True,   exist_ok=True)

            # Images per class
            for idx in range(images_per_class):
                fname = f"{cls}_{idx:03d}.png"
                cpath = captcha_dir / fname
                epath = clean_dir   / fname

                # Difficulty
                if difficulty == "easy":
                    easy.write_clean(cls, str(cpath))    
                elif difficulty == "medium":
                    medium.write(cls, str(cpath))        
                elif difficulty == "hard":
                    hard.write(cls, str(cpath))          
                
                # Always write a clean version
                easy.write_clean(cls, str(epath))   

                # Write labels
                labels_fp.write(f"{cls}/{SUBDIR_CAPTCHA}/{fname} {cls}\n")
                labels_fp.write(f"{cls}/{SUBDIR_CLEAN}/{fname} {cls}\n")

    labels_fp.close()
    return out_vocab


# ─────────────  Entry point  ─────────────
def main() -> None:
    
    train_vocab = None
    cpy_train_vocab = None
    
    for diff in ("easy", "medium", "hard"):
        print(f"\n=== CAPTCHA builder | difficulty= {diff} ===")
            
        # Train
        train_vocab = build_split(split_name="train",
                                difficulty=diff,
                                classes_plan=TRAIN_CLASSES_PER_LENGTH,
                                images_per_class=IMAGES_PER_CLASS,
                                existing_vocab=cpy_train_vocab)

        if diff == "easy": 
            cpy_train_vocab = train_vocab
        
        # Test
        build_split(split_name="test",
                    difficulty=diff,
                    classes_plan=None,
                    images_per_class=TEST_IMAGES_PER_CLASS,
                    existing_vocab=cpy_train_vocab)
        print("\n✔ Dataset ready at:", (ROOT.resolve()))
        # Copy after easy finishes


# Example
if __name__ == "__main__":
    main()