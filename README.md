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
| `--out` | output folder (created if missing; results of a previous run there are replaced) |
| `--threads` | optional, CPU threads (default 4) |

Output: `outputs/predictions.csv` (`filename,class`, UTF-8, one row per image) and a copy of each image in
`outputs/<class>/`. Original images are never moved or changed. If `--out` already holds results, only
`predictions.csv` and the 7 class folders are replaced (other files are kept). The time per image (read image ->
class, model loading excluded) is printed for every image plus the mean; no timing file is written.
Any error (missing input/weights, no images, unreadable image, `--out` inside the input folder, bad csv) prints
`Error: ...` and exits with code 1.

## weights/ (all files are required except config.json)

| File | What it is |
|---|---|
| `bean_finder_ncnn_model/model.ncnn.param` + `.bin` | bean finder: YOLO26n detection, 1 class, input 1024×1024 |
| `cls_ncnn_model/model.ncnn.param` + `.bin` | bean classifier: YOLO26n-cls, 7 classes, input 224×224 · model **"E8"**: started from our earlier coffee-bean classifier (`coffee_cls.pt`, trained on a public coffee-bean dataset) and fine-tuned on bean crops from the organizer's circle/square images plus crops from simulated piles made from those same images |
| `classes.txt` | classifier output order: `black, broken, good, husk, immature, parchment, shell` |
| `config.json` | optional preprocessing values (if missing, the defaults in `infer.py` are used) |
| `*/metadata.yaml` | export information from Ultralytics · `infer.py` reads its `task:` line to tell the finder (`detect`) from the classifier (`classify`) |

Both models were trained with Ultralytics 8.4.208 and exported with `yolo export format=ncnn`
(`yolo export model=best.pt format=ncnn imgsz=224`, `yolo export model=bean_finder.pt format=ncnn imgsz=1024`).
The PyTorch `.pt` files are not in this repository (PyTorch cannot be installed on Pi OS 32-bit; `infer.py` runs
the NCNN files above).

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
6. **Vote** over beans = image class. `weights/config.json` `"vote": "probsum"` (used in this submission) picks the
   class with the highest summed bean probability; `"majority"` (code default) counts bean votes (tie → class seen
   first). If no bean is found, the whole image is classified so every image still gets a valid class.

ncnn is forced to fp32 (ncnn's ARM default fp16 changed probabilities slightly in our tests).

## Verified (MacBook, ncnn 1.0.20261020, numpy 2.5.4, pillow 13.0.0)

- NCNN models give the same bean counts, predictions and votes as the Ultralytics `.pt` models on our 7 held-out
  validation images (`grid_*`, not used for training).
- Held-out results (model E8): 7/7 validation images; 70/70 simulated piles built only from validation beans
  (random positions with gaps, 9–20 beans); 69/70 harder simulated piles (beans touching, shadows, brightness and
  size changes); 5/5 re-compressed validation images. Bean finder found 1017/1017 beans in the 70 piles.
- Error cases above exit with code 1; normal runs exit with code 0.
- Not yet tested on a real Pi OS 32-bit board.

## Training (`train.py`, on the training PC — not needed on the Pi)

`train.py` rebuilds the submitted classifier from the organizer images (`<data>/<class>/*.jpg`, 21 images).
Training needs `pip install ultralytics==8.4.208` (PyTorch, OpenCV); it is not part of `requirements.txt`.

```bash
# submitted model E8 (bean_finder.pt and coffee_cls.pt are our own earlier PyTorch weights, not in this repo)
python train.py --data day1_data --out runs_train/e8 --finder bean_finder.pt \
    --base-model coffee_cls.pt --weights weights
```

Steps: normalize every image (as in `infer.py`) → bean finder crops, split by image (`grid_*` = validation) →
simulated piles (beans re-placed at random with gaps, from training images only) → crops from those piles through
the `infer.py` pipeline → YOLO26n-cls training (imgsz 224, 50 epochs, patience 15, batch 32, lr0 0.01,
weight_decay 0.0005, hsv_h 0, hsv_s 0.3, degrees 180, flipud 0.5, seed 0) → NCNN export.
Result: `<out>/weights_new/` (same layout as `weights/`); test it with `infer.py --weights <out>/weights_new`
before replacing `weights/`. Hyper-parameters can be changed with `--epochs --patience --batch --imgsz --lr0
--weight-decay --dropout --base-model --piles-per-image --freeze --optimizer`. With the default `--optimizer auto`
Ultralytics picks AdamW and its own learning rate and ignores `--lr0`; set `--optimizer AdamW` (or `SGD`) to use
`--lr0`. GPU/MPS training is not bit-exact between machines.
