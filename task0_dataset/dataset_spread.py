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

# PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Custom
from captcha_spread import ImageCaptcha


# ────────────────────────────────────────────────────────────────────────────
# USER-ADJUSTABLE CONSTANTS
# ────────────────────────────────────────────────────────────────────────────

# RESOLUTION 
WIDTH = 256
HEIGHT = 256 

# Paths
ROOT = Path("task0_dataset/data_hard_new")
SUBDIR_CAPTCHA = "captcha"
SUBDIR_CLEAN   = "clean"

# Lengths of CAPTCHA strings to generate
LENGTHS: Sequence[int] = range(3, 6) 
# Images 
TRAIN_IMAGES = 90_000
TEST_IMAGES  = TRAIN_IMAGES // 10
IMAGES_PER_CLASS = 50   
# Per len = 100k/4 = 25k
# Num classes per len = 25k/25 = 1000
# Total Num classes = 100k / 25 = 4k = 1000 * 4 

# NOTE 
# Train Classes per length: {3: 600, 4: 600, 5: 600}
# Test Classes per length: {3: 60, 4: 60, 5: 60}

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
TEST_CLASSES_PER_LENGTH  = _classes_per_length(TEST_IMAGES  // len(LENGTHS) if LENGTHS else 0)
print(f"\nTrain Classes per length: {TRAIN_CLASSES_PER_LENGTH}")
print(f"Test Classes per length: {TEST_CLASSES_PER_LENGTH}\n") 


# ─────────────  Main writer  ─────────────
def build_split(
    *,
    split_name: str,
    difficulty: str,
    classes_plan: dict[int, int],
    existing_vocab: set[str] | None = None,
    output_mode: str = "normal",
    write_labels: str = False
) -> set[str]:
    
    # For test split 
    if existing_vocab is None: 
        existing_vocab = set()
    
    # Paths 
    if output_mode == "normal":
        root = ROOT / split_name / difficulty
        root.mkdir(parents=True, exist_ok=True)
    if write_labels: labels_fp = (root / "labels.txt").open("w", encoding="utf8")

    # Prepare classes
    out_vocab: set[str] = set()
    for length in sorted(LENGTHS):
        n_cls = classes_plan.get(length, 0)
        if n_cls == 0:
            continue
        new_cls = _unique_strings(n_cls, length, existing_vocab | out_vocab)
        out_vocab.update(new_cls)

        pbar_desc = f"{split_name}-{difficulty}-len{length}"
        for cls in tqdm(new_cls, desc=pbar_desc, unit="cls"):
            # normal
            if output_mode == "normal":
                cls_dir     = root / cls
                captcha_dir = cls_dir / SUBDIR_CAPTCHA
                clean_dir   = cls_dir / SUBDIR_CLEAN
                captcha_dir.mkdir(parents=True, exist_ok=True)
                clean_dir.mkdir(parents=True,   exist_ok=True)
            # pix2pix
            elif output_mode == "pix2pix":
                A_dir = ROOT / "A" / split_name
                B_dir = ROOT / "B" / split_name
                A_dir.mkdir(parents=True, exist_ok=True)
                B_dir.mkdir(parents=True, exist_ok=True)

            # Images per class
            for idx in range(IMAGES_PER_CLASS):
                fname = f"{cls}_{idx:03d}.png"
                if output_mode == "normal":
                    cpath = captcha_dir / fname
                    epath = clean_dir   / fname
                elif output_mode == "pix2pix":
                    cpath = A_dir / fname
                    epath = B_dir / fname 
                     
                # CAPTCHA (medium)
                if difficulty == "medium": 
                    medium.write(cls, output=str(cpath), output_clean=str(epath))    
                        
                # CAPTCHA (hard)
                elif difficulty == "hard":
                    hard.write(cls, output=str(cpath), output_clean=str(epath))

                # Write labels
                if write_labels: 
                    labels_fp.write(f"{cls}/{SUBDIR_CAPTCHA}/{fname} {cls}\n")
                    labels_fp.write(f"{cls}/{SUBDIR_CLEAN}/{fname} {cls}\n")

    if write_labels: labels_fp.close()
    return out_vocab



if __name__ == "__main__":
    # difficulty of captcha 
    DIFFICULTY = "hard"
    OUTPUT_MODE = "pix2pix"
    
    print(f"\n=== CAPTCHA builder | difficulty={DIFFICULTY} ===")
    train_vocab = build_split(split_name="train",
                              difficulty=DIFFICULTY,
                              classes_plan=TRAIN_CLASSES_PER_LENGTH,
                              output_mode=OUTPUT_MODE)
    build_split(split_name="test",
                difficulty=DIFFICULTY,
                classes_plan=TEST_CLASSES_PER_LENGTH,
                existing_vocab=train_vocab,
                output_mode=OUTPUT_MODE)
    print("✔ Dataset ready at:", (ROOT.resolve()))