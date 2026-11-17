"""Classify test images with a locally trained Ultralytics YOLO classifier."""

import argparse
import csv
import shutil
import sys
from pathlib import Path

CLASSES = ("black", "broken", "good", "husk", "immature", "parchment", "shell")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Test image or folder (searched recursively)")
    parser.add_argument("--weights", required=True, type=Path, help="Local .pt file or folder containing best.pt")
    parser.add_argument("--out", required=True, type=Path, help="Empty or new output folder")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size; defaults to model's trained size")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.threads < 1 or (args.imgsz is not None and args.imgsz < 1):
        parser.error("--threads and --imgsz must be positive")
    return args


def run(args):
    source = args.input.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if weights.is_dir():
        weights = weights / "best.pt"
    if not weights.is_file() or weights.suffix.lower() != ".pt":
        raise ValueError(f"Expected a local .pt checkpoint: {weights}")
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if source.is_dir() and (output == source or source in output.parents or output in source.parents):
        raise ValueError("Input and output folders must be separate and must not contain one another")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty folder to avoid overwriting earlier results")
    images = sorted(
        (p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if source.is_dir() else ([source] if source.suffix.lower() in IMAGE_EXTENSIONS else [])
    )
    if not images:
        raise ValueError(f"No supported test images found in {source}")

    filenames = [path.name for path in images]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Input images must have unique filenames, including across subfolders")

    try:
        import torch
        from PIL import Image, ImageOps
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: python -m pip install -r requirements.txt") from exc

    torch.set_num_threads(args.threads)
    model = YOLO(str(weights))
    if model.task != "classify":
        raise ValueError("Weights must be from an Ultralytics classification model (task=classify)")
    names = model.names
    labels = {index: str(name).strip().lower() for index, name in names.items()}
    if len(labels) != len(CLASSES) or set(labels.values()) != set(CLASSES):
        raise ValueError(f"Model labels must be {CLASSES}; found {labels}. Use weights trained on these classes.")
    options = {"device": "cpu", "verbose": False, "save": False, "augment": False}
    if args.imgsz is not None:
        options["imgsz"] = args.imgsz

    output.mkdir(parents=True, exist_ok=True)
    for label in CLASSES:
        (output / label).mkdir()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "class"])
        for index, path in enumerate(images, 1):
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            results = model.predict(source=image, **options)
            if len(results) != 1 or results[0].probs is None:
                raise ValueError(f"Expected one classification result for {path.name}")
            probs = results[0].probs
            if len(probs.data) != len(CLASSES) or not bool(torch.isfinite(probs.data).all()):
                raise ValueError(f"Invalid class probabilities for {path.name}")
            class_id = int(probs.top1)
            if class_id not in labels:
                raise ValueError(f"Invalid predicted class for {path.name}")
            prediction = labels[class_id]
            destination = output / prediction / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            writer.writerow([path.name, prediction])
            csv_file.flush()
            print(f"[{index}/{len(images)}] {path.name}: {prediction}", flush=True)
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
