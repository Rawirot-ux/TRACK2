"""Day 1 coffee-bean image classifier (TESA Top Gun Rally 2026, Track 2).

Runs on Raspberry Pi 4 CPU (Pi OS 32-bit, Python 3.13) using only ncnn + numpy + pillow
(no torch / ultralytics). One class per image, decided by majority vote over beans:

  1. open image -> EXIF transpose -> normalize (resize x scale LANCZOS + JPEG q once, in memory)
  2. bean finder (YOLO26n detect, NCNN) on a letterboxed square -> boxes (score > conf, NMS)
  3. drop boxes with abnormal area / aspect ratio
  4. crop each bean (+pad) -> bean classifier (YOLO26n-cls, NCNN) -> class per bean
  5. majority vote -> image class (if no bean is found, the whole image is classified)

Usage:
  python infer.py --input test/ --weights weights/ --out outputs/

weights/ must contain:
  bean_finder_ncnn_model/model.ncnn.param + model.ncnn.bin
  cls_ncnn_model/model.ncnn.param + model.ncnn.bin
  classes.txt   (class names in model output order, one per line)
  config.json   (optional; preprocessing values, see DEFAULT_CONFIG)
"""

import argparse
import csv
import io
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

CLASSES = ("black", "broken", "good", "husk", "immature", "parchment", "shell")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # INFER_SPEC: .jpg .jpeg .png, case-insensitive
FINDER_DIR = "bean_finder_ncnn_model"
CLASSIFIER_DIR = "cls_ncnn_model"

# Values used when the models were trained / validated (see README "Preprocessing").
DEFAULT_CONFIG = {
    "norm_scale": 0.75,      # resize factor before JPEG (same as training data)
    "norm_quality": 40,      # JPEG quality of the normalize step
    "finder_imgsz": 1024,    # finder input (square letterbox)
    "conf": 0.25,            # keep boxes with score > conf
    "iou": 0.5,              # NMS IoU (ultralytics predict default)
    "max_det": 1000,
    "min_area_ratio": 0.2,   # keep boxes with area in [min, max] x median area
    "max_area_ratio": 3.0,
    "max_aspect": 4.0,       # long side / short side
    "pad": 0.10,             # crop padding per side (fraction of box size)
    "cls_imgsz": 224,        # classifier input
    "vote": "majority",      # "majority" = most bean votes · "probsum" = highest summed bean probability
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Test image or folder (searched recursively)")
    parser.add_argument("--weights", required=True, type=Path, help="weights/ folder with the NCNN models")
    parser.add_argument("--out", required=True, type=Path, help="Empty or new output folder")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


# ---------------------------------------------------------------- models
def load_net(ncnn, model_dir, threads):
    param, binary = model_dir / "model.ncnn.param", model_dir / "model.ncnn.bin"
    if not param.is_file() or not binary.is_file():
        raise ValueError(f"Missing model.ncnn.param / model.ncnn.bin in {model_dir}")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = threads
    # Force fp32: ncnn's ARM default (fp16) changes probabilities by up to ~0.035 vs the trained model.
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_arithmetic = False
    if hasattr(net.opt, "use_bf16_storage"):
        net.opt.use_bf16_storage = False
    if net.load_param(str(param)) != 0 or net.load_model(str(binary)) != 0:
        raise ValueError(f"ncnn could not load the model in {model_dir}")
    return net


def run_net(ncnn, np, net, chw):
    """float32 (3,H,W) -> numpy output 'out0'.
    ncnn.Mat(numpy) does not copy: keep `arr` alive and clone, otherwise results are random."""
    arr = np.ascontiguousarray(chw, dtype=np.float32)
    mat = ncnn.Mat(arr).clone()
    extractor = net.create_extractor()
    extractor.input("in0", mat)
    ret, out = extractor.extract("out0")
    if ret != 0:
        raise RuntimeError("ncnn extract failed")
    result = np.array(out).copy()
    del extractor, arr
    return result


def load_models(weights, threads):
    try:
        import ncnn
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    if not weights.is_dir():
        raise ValueError(f"--weights must be the weights/ folder: {weights}")
    classes_file = weights / "classes.txt"
    if not classes_file.is_file():
        raise ValueError(f"Missing {classes_file}")
    names = [line.strip().lower() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != len(CLASSES) or set(names) != set(CLASSES):
        raise ValueError(f"classes.txt must list {CLASSES}; found {names}")

    config = dict(DEFAULT_CONFIG)
    config_file = weights / "config.json"
    if config_file.is_file():
        extra = json.loads(config_file.read_text(encoding="utf-8"))
        unknown = set(extra) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown keys in {config_file}: {sorted(unknown)}")
        config.update(extra)

    finder = load_net(ncnn, weights / FINDER_DIR, threads)
    classifier = load_net(ncnn, weights / CLASSIFIER_DIR, threads)
    return {"ncnn": ncnn, "np": np, "Image": Image, "ImageOps": ImageOps,
            "finder": finder, "classifier": classifier, "names": names, "cfg": config}


# ---------------------------------------------------------------- preprocessing
def normalize(m, image):
    """Same as training: resize x norm_scale (LANCZOS) -> JPEG quality once (in memory) -> decode."""
    Image, cfg = m["Image"], m["cfg"]
    image = image.convert("RGB")
    scale = cfg["norm_scale"]
    if scale != 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(cfg["norm_quality"]))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize_linear(np, a, new_w, new_h):
    """uint8 (H,W,3) resize identical to cv2.resize INTER_LINEAR (no antialias), numpy only."""
    h, w = a.shape[:2]

    def index(n_out, n_in):
        f = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        t[i0 < 0] = 0.0
        i0 = np.clip(i0, 0, n_in - 1)
        t[i0 >= n_in - 1] = 0.0
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, t.astype(np.float32)

    y0, y1, ty = index(new_h, h)
    x0, x1, tx = index(new_w, w)
    a = a.astype(np.float32)
    rows = a[y0] * (1 - ty)[:, None, None] + a[y1] * ty[:, None, None]
    out = rows[:, x0] * (1 - tx)[None, :, None] + rows[:, x1] * tx[None, :, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def letterbox(np, image, size):
    """PIL RGB -> (float32 CHW in 0..1, ratio, left pad, top pad); same padding as ultralytics LetterBox."""
    w, h = image.size
    r = min(size / h, size / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    a = np.asarray(image)
    if (nw, nh) != (w, h):
        a = resize_linear(np, a, nw, nh)
    top, left = round((size - nh) / 2 - 0.1), round((size - nw) / 2 - 0.1)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = a
    return canvas.transpose(2, 0, 1).astype(np.float32) / 255.0, r, left, top


def nms(np, boxes, scores, iou_thr):
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou_thr]
    return np.array(keep, dtype=np.int64)


def find_beans(m, image):
    """-> list of (x1, y1, x2, y2) in normalized-image pixels, after the area/aspect filter."""
    np, cfg = m["np"], m["cfg"]
    chw, r, left, top = letterbox(np, image, int(cfg["finder_imgsz"]))
    out = run_net(m["ncnn"], np, m["finder"], chw)
    if out.ndim != 2 or out.shape[0] != 5:
        raise ValueError(f"Unexpected finder output shape {out.shape}; expected (5, N)")
    cx, cy, bw, bh, score = out
    keep = score > cfg["conf"]
    cx, cy, bw, bh, score = cx[keep], cy[keep], bw[keep], bh[keep], score[keep]
    if score.size > 30000:
        best = np.argsort(-score, kind="stable")[:30000]
        cx, cy, bw, bh, score = cx[best], cy[best], bw[best], bh[best], score[best]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    boxes = boxes[nms(np, boxes, score, cfg["iou"])[: int(cfg["max_det"])]]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / r, 0, image.width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / r, 0, image.height)
    if len(boxes) == 0:
        return []
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    area = w * h
    median = np.median(area)
    aspect = np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)
    ok = ((area >= cfg["min_area_ratio"] * median) & (area <= cfg["max_area_ratio"] * median)
          & (aspect <= cfg["max_aspect"]))
    return [tuple(float(v) for v in b) for b in boxes[ok]]


def classify_crop(m, crop):
    return m["names"][int(m["np"].argmax(classify_probs(m, crop)))]


def classify_probs(m, crop):
    """ultralytics classify_transforms: resize short side (BILINEAR) -> center crop -> /255 (no mean/std)."""
    np, Image, size = m["np"], m["Image"], int(m["cfg"]["cls_imgsz"])
    w, h = crop.size
    nw, nh = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    if (nw, nh) != (w, h):
        crop = crop.resize((nw, nh), Image.BILINEAR)
    top, left = int(round((nh - size) / 2.0)), int(round((nw - size) / 2.0))
    crop = crop.crop((left, top, left + size, top + size))
    chw = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
    probs = run_net(m["ncnn"], np, m["classifier"], chw).reshape(-1)
    if probs.size != len(m["names"]) or not bool(np.isfinite(probs).all()):
        raise ValueError("Invalid class probabilities from the classifier")
    return probs


def crop_bean(image, box, pad):
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return image.crop((max(0, x1 - pw), max(0, y1 - ph), min(image.width, x2 + pw), min(image.height, y2 + ph)))


def load_image(m, path):
    with m["Image"].open(path) as original:
        return normalize(m, m["ImageOps"].exif_transpose(original))


def predict_image(m, path):
    """-> (class name, number of beans used, vote counter)."""
    np, names = m["np"], m["names"]
    image = load_image(m, path)
    boxes = find_beans(m, image)
    probs = [classify_probs(m, crop_bean(image, box, m["cfg"]["pad"])) for box in boxes]
    if not probs:  # no bean found: classify the whole image so every image still gets a valid class
        probs = [classify_probs(m, image)]
    votes = Counter(names[int(np.argmax(p))] for p in probs)
    if m["cfg"]["vote"] == "probsum":
        prediction = names[int(np.argmax(np.sum(probs, axis=0)))]
    elif m["cfg"]["vote"] == "majority":
        prediction = votes.most_common(1)[0][0]  # tie -> class seen first (box order by score)
    else:
        raise ValueError(f"config vote must be 'majority' or 'probsum', got {m['cfg']['vote']!r}")
    if prediction not in CLASSES:
        raise ValueError(f"Invalid predicted class {prediction!r}")
    return prediction, len(boxes), votes


# ---------------------------------------------------------------- main
def run(args):
    source = args.input.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not weights.exists():
        raise ValueError(f"Weights folder does not exist: {weights}")
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if source.is_dir() and (output == source or source in output.parents or output in source.parents):
        raise ValueError("Input and output folders must be separate and must not contain one another")
    if output.exists() and not output.is_dir():
        raise ValueError(f"--out exists and is not a folder: {output}")
    images = sorted(
        (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if source.is_dir() else ([source] if source.suffix.lower() in IMAGE_EXTENSIONS else [])
    )
    if not images:
        raise ValueError(f"No supported test images (.jpg/.jpeg/.png) found in {source}")

    filenames = [path.name for path in images]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Input images must have unique filenames, including across subfolders")

    model = load_models(weights, args.threads)

    output.mkdir(parents=True, exist_ok=True)
    # A previous run (or the sample outputs/ shipped with the submission) is replaced: only the files this
    # program writes (predictions.csv and the 7 class folders) are removed; anything else is left alone.
    old = [output / "predictions.csv"] + [output / label for label in CLASSES]
    if any(p.exists() for p in old):
        print(f"Note: replacing previous results in {output}", flush=True)
    for p in old:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    for label in CLASSES:
        (output / label).mkdir()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(["filename", "class"])
        times = []
        for index, path in enumerate(images, 1):
            start = time.perf_counter()
            try:
                prediction, n_beans, votes = predict_image(model, path)
            except Exception as exc:
                raise RuntimeError(f"Failed on {path.name}: {exc}") from exc
            times.append(time.perf_counter() - start)  # read image -> class (excludes copy/csv)
            shutil.copy2(path, output / prediction / path.name)
            writer.writerow([path.name, prediction])
            csv_file.flush()
            vote_text = " ".join(f"{k}:{v}" for k, v in votes.most_common())
            print(f"[{index}/{len(images)}] {path.name}: {prediction} ({n_beans} beans; {vote_text}) "
                  f"{times[-1]:.2f} s", flush=True)

    # check the written file: header + one row per image, unique, valid class
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    if rows[0] != ["filename", "class"] or len(rows) - 1 != len(images) \
            or len({r[0] for r in rows[1:]}) != len(images) or any(r[1] not in CLASSES for r in rows[1:]):
        raise RuntimeError("predictions.csv failed the format check")
    print(f"Time per image: mean {sum(times) / len(times):.2f} s, max {max(times):.2f} s "
          f"(model loading excluded)")
    print(f"Saved {len(images)} predictions to {output / 'predictions.csv'}")


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
