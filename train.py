"""Train the Day 1 bean classifier (reproduces the submitted model "E1").

Runs on the training PC (Ultralytics + PyTorch + OpenCV), NOT on the Raspberry Pi.
Training-only dependencies (not in requirements.txt, which is for infer.py):
    pip install ultralytics==8.4.208   # brings torch, opencv-python, numpy, pillow
The bean finder is used as-is (weights/source_pt/bean_finder.pt, trained earlier on an external
coffee-bean dataset, 1 class "bean"); only the 7-class bean classifier is trained here.

Pipeline (the same preprocessing as infer.py):
  1. every organizer image: EXIF transpose -> normalize (resize x0.75 LANCZOS + JPEG q40, in memory)
  2. bean finder (.pt) on the normalized image -> boxes (conf 0.25, area/aspect filter) -> crops (+10 % pad)
     split by IMAGE: file names containing --val-key (default "grid") go to val, the rest to train
  3. simulated piles from TRAIN images only: each bean is cut out with a colour mask, the original paper
     is cleaned, beans are re-placed in a random cluster with gaps (random rotation/flip), 9-20 beans per pile
  4. crops from those piles via infer.py (NCNN finder, exactly like at test time) are added to train only
  5. YOLO26n-cls training (hyper-parameters below) -> best.pt -> NCNN export -> new weights folder

Usage (paths are required on purpose):
  python train.py --data day1_data --out runs_train/e1 --finder weights/source_pt/bean_finder.pt \
      --base-model yolo26n-cls.pt --weights weights
Result: <out>/weights_new/ = drop-in replacement for weights/ (copy it yourself after checking it).
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="organizer data: <data>/<class>/*.jpg")
    p.add_argument("--out", required=True, type=Path, help="new work folder (must not exist)")
    p.add_argument("--finder", required=True, type=Path, help="bean finder .pt (weights/source_pt/bean_finder.pt)")
    p.add_argument("--base-model", required=True, help="classifier start weights, e.g. yolo26n-cls.pt")
    p.add_argument("--weights", required=True, type=Path, help="current weights/ (NCNN finder + classes.txt)")
    p.add_argument("--val-key", default="grid", help="images whose name contains this go to val")
    p.add_argument("--piles-per-image", type=int, default=4, help="simulated piles per train image (0 = none)")
    p.add_argument("--pile-seed", type=int, default=1)
    # classifier hyper-parameters (values used for the submitted model)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps", help="mps (Mac) / 0 (NVIDIA GPU) / cpu")
    p.add_argument("--freeze", type=int, default=None, help="freeze first N layers (None = train all)")
    p.add_argument("--optimizer", default="auto", help="auto ignores --lr0; e.g. AdamW / SGD to use --lr0")
    return p.parse_args()


def load_infer():
    sys.path.insert(0, str(HERE))
    import infer  # same normalize / crop / NCNN code as at test time
    return infer


# ---------------------------------------------------------------- step 2: crops with the .pt finder
def finder_boxes(np, finder, image_rgb, device):
    r = finder.predict(image_rgb, imgsz=1024, conf=0.25, device=device, max_det=1000, verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    if len(b) == 0:
        return b
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    area = w * h
    med = np.median(area)
    asp = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)
    return b[(area >= 0.2 * med) & (area <= 3.0 * med) & (asp <= 4.0)]


# ---------------------------------------------------------------- step 3: simulated piles
def bean_mask(cv2, np, rgb, box, pad=0.25):
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x0, y0 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    xe, ye = int(min(W, x2 + pw)), int(min(H, y2 + ph))
    crop = rgb[y0:ye, x0:xe]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    paper = np.median(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]), axis=0)
    dist = np.sqrt(((lab - paper) ** 2).sum(-1))
    dist = np.clip(dist * (255.0 / max(dist.max(), 1)), 0, 255).astype(np.uint8)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    _, m = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(cc == big, 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cnts, -1, 255, -1)
    if m.sum() / 255 < 0.25 * (x2 - x1) * (y2 - y1):
        return None
    return x0, y0, crop, m


def clean_background(cv2, np, rgb, masks, dil):
    H, W = rgb.shape[:2]
    hole = np.zeros((H, W), np.uint8)
    for x0, y0, _, m in masks:
        hole[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))
    small = cv2.resize(rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    hs = cv2.dilate(cv2.resize(hole, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    up = cv2.resize(cv2.inpaint(small, hs, 5, cv2.INPAINT_TELEA), (W, H), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.GaussianBlur(hole, (31, 31), 0).astype(np.float32)[..., None] / 255.0
    return (rgb * (1 - alpha) + up * alpha).astype(np.uint8)


def rotate_bean(cv2, np, crop, m, angle, flip):
    if flip:
        crop, m = crop[:, ::-1], m[:, ::-1]
    ys, xs = np.nonzero(m)
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c) + 2, int(h * c + w * s) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rc = cv2.warpAffine(np.ascontiguousarray(crop), M, (nw, nh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    rm = cv2.warpAffine(np.ascontiguousarray(m), M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    ys, xs = np.nonzero(rm > 127)
    return rc[ys.min():ys.max() + 1, xs.min():xs.max() + 1], rm[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make_pile(cv2, np, bg, beans, n, rng, size_ref, gap_min=0.05, gap_max=0.5):
    H, W = bg.shape[:2]
    occ = np.zeros((H, W), np.uint8)
    order = list(range(len(beans)))
    rng.shuffle(order)
    order += [rng.randrange(len(beans)) for _ in range(max(0, n - len(beans)))]
    placed, layers = [], []
    for bi in order:
        rc, rm = rotate_bean(cv2, np, *beans[bi], rng.uniform(0, 360), rng.random() < 0.5)
        h, w = rm.shape
        gap = int(rng.uniform(gap_min, gap_max) * size_ref)
        mbin = (rm > 127).astype(np.uint8)
        grown = cv2.copyMakeBorder(mbin, gap, gap, gap, gap, cv2.BORDER_CONSTANT, value=0)
        if gap > 0:
            grown = cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1)))
        for _ in range(400):
            if not placed:
                cx, cy = rng.uniform(0.35, 0.65) * W, rng.uniform(0.35, 0.65) * H
            else:
                ax, ay = placed[rng.randrange(len(placed))]
                th = rng.uniform(0, 2 * np.pi)
                d = rng.uniform(0.6, 1.6) * size_ref + gap
                cx, cy = ax + d * np.cos(th), ay + d * np.sin(th)
            x0, y0 = int(cx - w / 2), int(cy - h / 2)
            if x0 - gap < 0 or y0 - gap < 0 or x0 + w + gap > W or y0 + h + gap > H:
                continue
            if (occ[y0 - gap:y0 + h + gap, x0 - gap:x0 + w + gap] & grown).any():
                continue
            occ[y0:y0 + h, x0:x0 + w] |= mbin
            layers.append((x0, y0, rc, rm))
            placed.append((x0 + w / 2, y0 + h / 2))
            break
    canvas = bg.astype(np.float32).copy()
    for x0, y0, rc, rm in layers:
        h, w = rm.shape
        al = cv2.GaussianBlur(rm, (3, 3), 0).astype(np.float32)[..., None] / 255.0
        canvas[y0:y0 + h, x0:x0 + w] = canvas[y0:y0 + h, x0:x0 + w] * (1 - al) + rc.astype(np.float32) * al
    return np.clip(canvas, 0, 255).astype(np.uint8), len(layers)


# ---------------------------------------------------------------- main
def main():
    a = parse_args()
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    data, out = a.data.expanduser().resolve(), a.out.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists - choose a new --out (nothing is overwritten)")
    classes = sorted(d.name for d in data.iterdir() if d.is_dir() and not d.name.startswith("."))
    infer = load_infer()
    if set(classes) != set(infer.CLASSES):
        raise SystemExit(f"class folders {classes} != {infer.CLASSES}")
    images = sorted((f for c in classes for f in (data / c).iterdir() if f.suffix.lower() in IMG_EXT),
                    key=lambda f: f.name)
    m = infer.load_models(a.weights.expanduser().resolve(), 4)  # NCNN finder + normalize settings
    finder = YOLO(str(a.finder.expanduser().resolve()))
    ds = out / "dataset"
    for split in ("train", "val"):
        for c in classes:
            (ds / split / c).mkdir(parents=True)

    # step 1-2: crops from organizer images
    print("== step 1-2: normalize + bean crops (split by image)")
    for f in images:
        c = f.parent.name
        split = "val" if a.val_key in f.stem else "train"
        img = infer.load_image(m, f)
        boxes = finder_boxes(np, finder, img, a.device)
        for i, b in enumerate(boxes):
            infer.crop_bean(img, b, m["cfg"]["pad"]).save(ds / split / c / f"{f.stem}_{i:02d}.jpg", quality=95)
        print(f"  {c}/{f.name:24s} {split:5s} {len(boxes)} beans")

    # step 3-4: simulated piles from TRAIN images only -> crops via the test-time pipeline
    if a.piles_per_image > 0:
        print("== step 3-4: simulated piles from train images only")
        piles = out / "piles"
        piles.mkdir()
        rng = random.Random(a.pile_seed)
        meta = {}
        for f in (f for f in images if a.val_key not in f.stem):
            c = f.parent.name
            rgb = np.asarray(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            boxes = finder_boxes(np, finder, rgb[:, :, ::-1].copy(), a.device)
            masks = [bm for b in boxes if (bm := bean_mask(cv2, np, rgb, b)) is not None]
            if not masks:
                raise SystemExit(f"no bean mask in {f.name}")
            size_ref = float(np.median([max(mm.shape) for _, _, _, mm in masks])) / 1.5
            bg = clean_background(cv2, np, rgb, masks, int(0.8 * size_ref) | 1)
            beans = [(cc, mm) for _, _, cc, mm in masks]
            for k in range(a.piles_per_image):
                n = rng.randint(9, 20)
                pile, placed = make_pile(cv2, np, bg, beans, n, rng, size_ref)
                name = f"pile_{c}_{f.stem}_{k:02d}.jpg"
                Image.fromarray(pile).save(piles / name, quality=95)
                meta[name] = c
        n_added = 0
        for name, c in sorted(meta.items()):
            img = infer.load_image(m, piles / name)
            for i, b in enumerate(infer.find_beans(m, img)):
                infer.crop_bean(img, b, m["cfg"]["pad"]).save(ds / "train" / c / f"{Path(name).stem}_{i:02d}.jpg",
                                                             quality=95)
                n_added += 1
        print(f"  {len(meta)} piles -> {n_added} extra train crops")
    for split in ("train", "val"):
        print(f"  {split}: " + ", ".join(f"{c} {len(list((ds / split / c).glob('*.jpg')))}" for c in classes))

    # step 5: train + export
    print("== step 5: train classifier")
    model = YOLO(a.base_model)
    model.train(data=str(ds), imgsz=a.imgsz, epochs=a.epochs, patience=a.patience, batch=a.batch,
                lr0=a.lr0, weight_decay=a.weight_decay, dropout=a.dropout, seed=a.seed, device=a.device,
                freeze=a.freeze, optimizer=a.optimizer,
                hsv_h=0.0, hsv_s=0.3, degrees=180, flipud=0.5,  # colour cues matter: no hue shift
                project=str(out), name="train", exist_ok=False)
    best = out / "train" / "weights" / "best.pt"
    YOLO(str(best)).export(format="ncnn", imgsz=a.imgsz, device="cpu")
    new = out / "weights_new"
    shutil.copytree(a.weights / infer.FINDER_DIR, new / infer.FINDER_DIR)
    shutil.copytree(best.parent / "best_ncnn_model", new / infer.CLASSIFIER_DIR,
                    ignore=shutil.ignore_patterns("model_ncnn.py"))
    names = YOLO(str(best)).names
    (new / "classes.txt").write_text("\n".join(names[i] for i in range(len(names))) + "\n")
    cfg = a.weights / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, new / "config.json")
    if a.imgsz != infer.DEFAULT_CONFIG["cls_imgsz"]:
        conf = json.loads((new / "config.json").read_text()) if (new / "config.json").is_file() else {}
        conf["cls_imgsz"] = a.imgsz
        (new / "config.json").write_text(json.dumps(conf, indent=2) + "\n")
    (new / "source_pt").mkdir()
    shutil.copy2(best, new / "source_pt" / "best.pt")
    shutil.copy2(a.finder, new / "source_pt" / "bean_finder.pt")
    print(f"\nDone. New weights folder: {new}\nCheck it with infer.py, then copy it over weights/ yourself.")


if __name__ == "__main__":
    main()
