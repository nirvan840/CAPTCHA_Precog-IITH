# ===== captcha_colour_xbound_pipeline.py =====
#
# Use a binary mask (white = character, black = background) to find each
# character’s **left and right** X-limits only.  The crop keeps the full
# vertical extent of the CAPTCHA; the pad-to-square step then centre-pads
# and resizes the coloured glyph to 128 × 128.
# ───────────────────────────────────────────────────────────────────
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch



# ───────────────────────────────────────────────────────────────────
# 1. Build binary “ink” mask
# ------------------------------------------------------------------
def foreground_mask(img):
    """Return a cleaned, hole-filled 1-channel mask (uint8 0/255)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mask = np.logical_or(s > 25, v < 235).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    mask = cv2.dilate(mask, k, 1)

    # fill interior holes so each glyph is one blob
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


# ───────────────────────────────────────────────────────────────────
# 2. Helpers
# ------------------------------------------------------------------
def _merge_adjacent_boxes(boxes, max_gap=2):
    """
    boxes must be left-to-right sorted.
    Any two consecutive boxes whose horizontal gap ≤ max_gap pixels
    are merged into one (x,0,w,fullH) box.
    """
    if not boxes:
        return []

    merged = [list(boxes[0])]          # start with first box as [x,0,w,H]
    for x, y, w, h in boxes[1:]:
        prev_x, _, prev_w, _ = merged[-1]
        gap = x - (prev_x + prev_w)    # horizontal space between boxes

        if gap <= max_gap:             # blobs touch / nearly touch → same char
            # extend the previous box to include this one
            new_right = max(prev_x + prev_w, x + w)
            merged[-1][2] = new_right - prev_x
        else:
            merged.append([x, y, w, h])

    return [tuple(b) for b in merged]

def _vertical_trim(boxes, mask, pad=5):
    """
    For each (x,0,w,H) box, shrink its vertical span so it starts
    `pad` px above the highest white pixel and ends `pad` px below the
    lowest white pixel inside that X-range.  Caps at image edges.
    """
    H = mask.shape[0]
    trimmed = []
    for x, _, w, _ in boxes:
        col_slice = mask[:, x:x + w]               # all rows in that X-range
        rows = np.where(col_slice.any(axis=1))[0]  # rows containing ink
        if rows.size == 0:                         # safety
            continue
        top    = max(0, rows[0]  - pad)
        bottom = min(H - 1, rows[-1] + pad)
        trimmed.append((x, top, w, bottom - top + 1))
    return trimmed

def pad_to_square(img):
    h, w = img.shape[:2]
    sz = max(h, w)
    t = (sz - h) // 2
    b = sz - h - t
    l = (sz - w) // 2
    r = sz - w - l
    return cv2.copyMakeBorder(img, t, b, l, r,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])

def crop_colour(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


# ───────────────────────────────────────────────────────────────────
# 3. Character Bounding boxes
# ------------------------------------------------------------------
def x_bounds_from_mask(img, min_area=100, max_gap=1, vpad=5):
    """
    Returns boxes whose Y-limits hug the glyph vertically with `vpad`
    pixels of extra margin.
    """
    mask = foreground_mask(img)
    H, W = mask.shape

    # raw blob → full-height box
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    raw = [(stats[i][0], 0, stats[i][2], H)          # (x,0,w,H)
           for i in range(1, n) if stats[i][4] >= min_area]

    raw.sort(key=lambda b: b[0])                     # left→right
    merged = _merge_adjacent_boxes(raw, max_gap)     # collapse dot+stem
    final_boxes  = _vertical_trim(merged, mask, pad=vpad)  # trim top/bottom

    return final_boxes, mask


# ───────────────────────────────────────────────────────────────────
# 4. Save individual characters 
# ------------------------------------------------------------------
def save_char_images(path, colour_img, boxes, out_root, stop=False, debug=True):
    label = path.stem.split('_', 1)[0]
    chars = list(label)
    
    if stop: 
        assert len(chars) == len(boxes), \
            f"{path}: {len(boxes)} boxes vs {len(chars)} chars."
    elif len(chars) != len(boxes): 
        return 1

    if not debug:
        for ch, box in zip(chars, boxes):
            crop = crop_colour(colour_img, box)
            square = pad_to_square(crop)
            resized = cv2.resize(square, (128, 128),
                                interpolation=cv2.INTER_LINEAR)

            train_dir = out_root / 'train' / ch
            test_dir  = out_root / 'test'  / ch
            train_dir.mkdir(parents=True, exist_ok=True)
            test_dir.mkdir(parents=True, exist_ok=True)

            if len(list(train_dir.iterdir())) < 50:
                save_path = train_dir / path.name
            elif len(list(test_dir.iterdir())) < 10:
                save_path = test_dir / path.name
            else:
                continue
            cv2.imwrite(str(save_path), resized)
    
    return 0


# ───────────────────────────────────────────────────────────────────
# 5. Tensor export  (3×128×128 RGB)
# ------------------------------------------------------------------
def extract_chars_as_tensor_dict(img_path, min_area=100, stop=False):
    img = cv2.imread(str(img_path))
    boxes, _ = x_bounds_from_mask(img, min_area)
    chars = list(Path(img_path).stem.split('_', 1)[0])
    
    if stop: 
        assert len(chars) == len(boxes), \
            f"\n{img_path}: {len(boxes)} boxes vs {len(chars)} chars\n"
        return 0
    elif len(chars) != len(boxes):
        return 0

    tensors = {}
    for ch, box in zip(chars, boxes):
        crop = crop_colour(img, box)
        square = pad_to_square(crop)
        resized = cv2.resize(square, (128, 128),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).div(255.0)
        tensors[ch] = t
    return tensors


# ───────────────────────────────────────────────────────────────────
# 6. Driver routines
# ------------------------------------------------------------------
def process_folder(in_dir, out_root, debug=True):
    errors = 0
    
    in_dir, out_root = Path(in_dir), Path(out_root)
    for img_path in tqdm(sorted(in_dir.iterdir()), desc='Batch', unit='img'):
        if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        img = cv2.imread(str(img_path))
        boxes, _ = x_bounds_from_mask(img)
        errors += save_char_images(img_path, img, boxes, out_root, debug=debug)
    
    print(f"Errors: {errors}")

def run_tests(img_path, out_dir):
    out_dir = Path(out_dir)
    base = Path(img_path).stem
    label = base.split('_', 1)[0]
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    boxes, mask = x_bounds_from_mask(img)
    if (len(boxes)!= len(label)):
        print(f'Boxes: {len(boxes)} | Chars: {len(label)}')
    
    # 1) mask
    cv2.imwrite(str(label_dir / f'{base}-mask.png'), mask)

    # 2) draw X-boundary boxes
    vis = img.copy()
    for x, _, w, _ in boxes:
        cv2.rectangle(vis, (x, 0), (x + w, img.shape[0] - 1),
                      (0, 255, 0), 2)
    cv2.imwrite(str(label_dir / f'{base}-overlay.png'), vis)

    # 3) first glyph sample
    if boxes:
        crop = crop_colour(img, boxes[0])
        square = pad_to_square(crop)
        resized = cv2.resize(square, (128, 128),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(label_dir / f'{base}-first.png'), resized)


# ───────────────────────────────────────────────────────────────────
# 7. MAIN
# ------------------------------------------------------------------
if __name__ == '__main__':
    mode = 'batch'                     # 'batch', 'test', 'tensor'

    # batch settings
    input_dir   = 'task0_dataset/data_med_new/A/train'
    output_root = 'task2_generation/data_chars'

    # test / tensor settings
    img_path = 'task2_generation/samples/clean/IUHI7.png'
    out_dir  = 'task2_generation/samples/test_output'

    if mode == 'batch':
        process_folder(input_dir, output_root, debug=True)

    elif mode == 'test':
        run_tests(img_path, out_dir)

    elif mode == 'tensor':
        tdict = extract_chars_as_tensor_dict(img_path)
        k = next(iter(tdict))
        print(k, tdict[k].shape)       # e.g. '0 torch.Size([3,128,128])'
