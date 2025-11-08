import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import joblib
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
Standalone educational demo: Food detection (classification) using a simple ML pipeline.

What it does
- Loads a folder dataset structured as:
    dataset_root/
        class_name_1/ *.jpg, *.png
        class_name_2/ *.jpg, *.png
        ...
- Extracts color histogram features from each image (simple, lightweight feature).
- Trains a linear SVM classifier to recognize the food class.
- Optionally joins predictions with a nutrition CSV mapping class -> calories, protein, carbs, fat.
- Provides CLI commands to train, evaluate, and predict on new images.

This is a prototype for learning and presentation purposes. It is not
intended to reflect a production system or any external API use.
"""


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


@dataclass
class NutritionInfo:
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0


def list_images_in_dir(root: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS and img_path.is_file():
                items.append((class_name, img_path))
    return items


def extract_color_histogram(image_bgr: np.ndarray, bins_per_channel: int = 32) -> np.ndarray:
    # Convert to HSV for more robust color description
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Compute histogram for each channel and concatenate
    hist_h = cv2.calcHist([hsv], [0], None, [bins_per_channel], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins_per_channel], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins_per_channel], [0, 256])
    # Normalize histograms and flatten
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    feature = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
    return feature.astype(np.float32)


def load_dataset_features(dataset_root: Path, image_size: int = 224) -> Tuple[np.ndarray, List[str]]:
    data: List[np.ndarray] = []
    labels: List[str] = []
    items = list_images_in_dir(dataset_root)
    if not items:
        raise RuntimeError(f"No images found under {dataset_root}. Ensure it has subfolders per class.")
    for label, img_path in items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        feat = extract_color_histogram(img)
        data.append(feat)
        labels.append(label)
    return np.vstack(data), labels


def train_model(X: np.ndarray, y: List[str]) -> Tuple[svm.LinearSVC, StandardScaler, List[str]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = svm.LinearSVC(random_state=42)
    model.fit(X_scaled, y)
    classes = sorted(list(set(y)))
    return model, scaler, classes


def evaluate_model(model: svm.LinearSVC, scaler: StandardScaler, X: np.ndarray, y: List[str]) -> str:
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return f"Classification Report\n{report}\nConfusion Matrix\n{cm}"


def predict_image(model: svm.LinearSVC, scaler: StandardScaler, image_path: Path) -> Tuple[str, float]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    feat = extract_color_histogram(img)[None, :]
    feat_scaled = scaler.transform(feat)
    proba = _linear_svc_pseudo_proba(model, feat_scaled)[0]
    idx = int(np.argmax(proba))
    label = model.classes_[idx]
    confidence = float(proba[idx])
    return label, confidence


def _linear_svc_pseudo_proba(model: svm.LinearSVC, X_scaled: np.ndarray) -> np.ndarray:
    # LinearSVC does not provide calibrated probabilities.
    # We map decision_function outputs to [0,1] via a softmax for demonstration only.
    scores = model.decision_function(X_scaled)
    if scores.ndim == 1:
        scores = scores[:, None]
    # softmax
    e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    proba = e / np.sum(e, axis=1, keepdims=True)
    return proba


def load_nutrition_csv(csv_path: Path) -> Dict[str, NutritionInfo]:
    nutrition: Dict[str, NutritionInfo] = {}
    if not csv_path.exists():
        return nutrition
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("food") or row.get("name") or "").strip()
            if not name:
                continue
            def _flt(key: str) -> float:
                try:
                    return float(row.get(key, 0) or 0)
                except Exception:
                    return 0.0
            nutrition[name.lower()] = NutritionInfo(
                calories=_flt("calories"),
                protein_g=_flt("protein_g"),
                carbs_g=_flt("carbs_g"),
                fat_g=_flt("fat_g"),
            )
    return nutrition


def pretty_nutrition_for(food_name: str, table: Dict[str, NutritionInfo]) -> str:
    info = table.get(food_name.lower())
    if not info:
        return "Nutrition: (not found in CSV)"
    return (
        f"Nutrition (per serving) — Calories: {info.calories:.0f}, "
        f"Protein: {info.protein_g:.1f}g, Carbs: {info.carbs_g:.1f}g, Fat: {info.fat_g:.1f}g"
    )


def save_artifacts(model: svm.LinearSVC, scaler: StandardScaler, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")


def load_artifacts(out_dir: Path) -> Tuple[svm.LinearSVC, StandardScaler]:
    model = joblib.load(out_dir / "model.joblib")
    scaler = joblib.load(out_dir / "scaler.joblib")
    return model, scaler


def cmd_train(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset).resolve()
    X, y = load_dataset_features(dataset_root)
    model, scaler, classes = train_model(X, y)
    save_artifacts(model, scaler, Path(args.out))
    print(f"Trained classes: {classes}")
    print(f"Artifacts saved to: {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset).resolve()
    X, y = load_dataset_features(dataset_root)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model, scaler, _ = train_model(X_train, y_train)
    report = evaluate_model(model, scaler, X_test, y_test)
    print(report)


def cmd_predict(args: argparse.Namespace) -> None:
    model_dir = Path(args.model)
    image_path = Path(args.image)
    model, scaler = load_artifacts(model_dir)
    label, conf = predict_image(model, scaler, image_path)
    nutrition_table = load_nutrition_csv(Path(args.nutrition_csv)) if args.nutrition_csv else {}
    print(f"Prediction: {label} ({conf*100:.1f}% confidence)")
    if nutrition_table:
        print(pretty_nutrition_for(label, nutrition_table))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple food detection demo using color histograms + SVM")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model on a folder dataset")
    p_train.add_argument("--dataset", required=True, help="Path to dataset root folder")
    p_train.add_argument("--out", default="artifacts", help="Output directory for model artifacts")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate on a folder dataset (hold-out split)")
    p_eval.add_argument("--dataset", required=True, help="Path to dataset root folder")
    p_eval.set_defaults(func=cmd_eval)

    p_pred = sub.add_parser("predict", help="Predict a single image using saved artifacts")
    p_pred.add_argument("--model", default="artifacts", help="Directory with model.joblib and scaler.joblib")
    p_pred.add_argument("--image", required=True, help="Path to image file")
    p_pred.add_argument("--nutrition_csv", help="CSV mapping food->calories,protein_g,carbs_g,fat_g")
    p_pred.set_defaults(func=cmd_predict)

    return p


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


