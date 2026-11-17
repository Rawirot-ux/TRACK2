# TRACK2 — Day 1 Coffee Bean Image Classification

Predicts one class per image (`good, shell, black, husk, broken, parchment, immature`) on a
**Raspberry Pi 4 CPU, Pi OS 32-bit, Python 3.13**. Uses only `ncnn`, `numpy`, `pillow`
(no PyTorch / Ultralytics — they have no 32-bit ARM wheels).

## Install (on the Pi)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` adds `--extra-index-url https://www.piwheels.org/simple` because `numpy` and `pillow`
for armv7l / cp313 are published on piwheels; `ncnn` comes from PyPI.

## Run (standard command)

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

| Argument | Meaning |
|---|---|
| `--input` | folder of test images (searched recursively, `.jpg/.jpeg/.png`, any case) or one image |
| `--weights` | the `weights/` folder below |
| `--out` | new or empty output folder (created if missing) |
| `--threads` | optional, CPU threads (default 4) |

Output: `outputs/predictions.csv` (`filename,class`, UTF-8, one row per image) and a copy of each image in
`outputs/<class>/`. Original images are never moved or changed. Any error (missing input/weights, no images,
unreadable image, non-empty output folder, bad csv) prints `Error: ...` and exits with code 1.

## weights/ (all files are required except config.json)

| File | What it is |
|---|---|
| `bean_finder_ncnn_model/model.ncnn.param` + `.bin` | bean finder: YOLO26n detection, 1 class, input 1024×1024 |
| `cls_ncnn_model/model.ncnn.param` + `.bin` | bean classifier: YOLO26n-cls, 7 classes, input 224×224 |
| `classes.txt` | classifier output order: `black, broken, good, husk, immature, parchment, shell` |
| `config.json` | optional preprocessing values (if missing, the defaults in `infer.py` are used) |
| `*/metadata.yaml` | export information from Ultralytics (not read by `infer.py`) |

Both models were trained with Ultralytics 8.4.208 and exported with `yolo export format=ncnn`.

## Preprocessing / pipeline (fixed, no random or test-time augmentation)

1. Open image, apply EXIF orientation, convert to RGB.
2. **Normalize** (same as training data): resize ×0.75 (LANCZOS) → JPEG quality 40 once in memory → decode.
   Reason: some organizer training images were already compressed; every image is brought to the same quality
   so the model cannot use compression as a class cue.
3. **Bean finder**: letterbox to 1024×1024 (bilinear like OpenCV, gray padding 114), RGB/255 → boxes with
   score > 0.25 → NMS IoU 0.5 → boxes mapped back to the image.
4. Drop boxes whose area is outside 0.2–3.0 × median area or whose long/short side > 4.
5. Crop each bean with 10 % padding per side → resize short side to 224 (bilinear) → center crop 224 → RGB/255
   (no mean/std) → classifier → class per bean.
6. **Majority vote** of bean classes = image class (tie → class seen first). If no bean is found, the whole
   image is classified so every image still gets a valid class.

ncnn is forced to fp32 (ncnn's ARM default fp16 changed probabilities slightly in our tests).

## Verified (MacBook, ncnn 1.0.20261020, numpy 2.5.4, pillow 13.0.0)

- Same predictions, bean counts and votes as the original Ultralytics `.pt` pipeline on our 7 validation images.
- Error cases above exit with code 1; normal runs exit with code 0.
- Not yet tested on a real Pi OS 32-bit board.
