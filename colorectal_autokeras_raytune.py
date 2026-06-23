"""
colorectal_autokeras_raytune_output_auto.py

A modified version of the user's PyTorch/timm ViT colorectal histology experiment.
It adds two optional optimisation/baseline routes and extra analysis outputs:

1) Ray Tune mode: tune training hyperparameters for one fixed PyTorch + timm ViT architecture.
2) AutoKeras mode: run a separate Keras/AutoKeras image-classification baseline.
3) Final-train mode now saves ROC curves, ROC-AUC values, misclassified patch grids,
   and selected approximate ViT attention-rollout maps.
4) ViT runs can ablate IMAGE_SIZE, patch size (16/32), and tiny/small/base capacity.
5) Both multiclass and patch-level binary tumor-vs-non-tumor tasks are supported.
6) Binary outputs include tumor-focused precision/recall/F1, PR-AUC, PR curves,
   and tumor false-positive/false-negative counts for imbalanced evaluation.

Recommended workflow:
    python colorectal_autokeras_raytune_enhanced.py
    python colorectal_autokeras_raytune_enhanced.py --mode auto_train --task multiclass --image-size 224 --patch-size 16 --vit-depth base
    python colorectal_autokeras_raytune_enhanced.py --mode auto_train --task binary --image-size 384 --patch-size 32 --vit-depth small --use-class-weights
    python colorectal_autokeras_raytune_enhanced.py --mode ray_tune --task multiclass --image-size 128 --patch-size 16 --vit-depth tiny --num-samples 6 --tune-epochs 6
    python colorectal_autokeras_raytune_enhanced.py --mode final_train --task multiclass --image-size 224 --patch-size 16 --vit-depth base --config-json output/colorectal_exp_auto/multiclass/image224_patch16_base/best_ray_config.json
    python colorectal_autokeras_raytune_enhanced.py --mode autokeras --task multiclass --ak-max-trials 3 --ak-epochs 10

Install examples:
    pip install timm scikit-learn matplotlib numpy torch
    pip install "ray[tune]"
    pip install autokeras tensorflow

Notes:
- PyTorch/Ray Tune and AutoKeras are kept in separate modes to avoid mixing two training frameworks.
- Ray Tune optimises validation macro-F1 by default.
- The test set is used only after final model selection.
- Input data is read from data/ and ViT outputs are kept under output/colorectal_exp_auto/<task>/image<image_size>_patch<patch_size>_<depth>/.
- Default auto_train mode asks for binary vs multiclass, IMAGE_SIZE, patch size, and ViT depth before running Ray Tune followed by final training automatically.
- Binary classification is implemented as patch-level tumor vs non-tumor, not patient-level diseased vs healthy.
"""

import os
import time
import json
import inspect
import random
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import timm


# -----------------------------------------------------------------------------
# AMP helpers, kept from the original PyTorch version
# -----------------------------------------------------------------------------
def get_autocast():
    try:
        from torch.amp import autocast as ac
        sig = inspect.signature(ac)
        if "device_type" in sig.parameters:
            def _ac(enabled: bool):
                return ac(device_type="cuda", enabled=enabled)
            return _ac
        else:
            def _ac(enabled: bool):
                return ac(enabled=enabled)
            return _ac
    except Exception:
        from torch.cuda.amp import autocast as ac
        sig = inspect.signature(ac)
        if "device_type" in sig.parameters:
            def _ac(enabled: bool):
                return ac(device_type="cuda", enabled=enabled)
            return _ac
        else:
            def _ac(enabled: bool):
                return ac(enabled=enabled)
            return _ac


def get_grad_scaler(device: torch.device):
    enabled = device.type == "cuda"
    try:
        from torch.amp import GradScaler as GS
        sig = inspect.signature(GS)
        if "device_type" in sig.parameters:
            return GS(device_type="cuda", enabled=enabled)
        return GS(enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as GS
        return GS(enabled=enabled)


autocast_ctx = get_autocast()


# -----------------------------------------------------------------------------
# Base config, adapted from the uploaded script
# -----------------------------------------------------------------------------
SEED = 42
NPZ_PATH = "data/colorectal_histology.npz"
MULTICLASS_CLASS_NAMES = [
    "tumor",
    "stroma",
    "complex",
    "lympho",
    "debris",
    "mucosa",
    "adipose",
    "empty",
]

# Binary task mapping for patch-level cancer detection:
# positive class = tumor patch, negative class = all non-tumor tissue/background classes.
# This is NOT patient-level "diseased vs healthy" classification, because the dataset is patch-level.
BINARY_CLASS_NAMES = ["non_tumor", "tumor"]
TUMOR_CLASS_INDEX = 0


def get_class_names(task: str) -> List[str]:
    if task == "binary":
        return BINARY_CLASS_NAMES
    return MULTICLASS_CLASS_NAMES


def get_num_classes(task: str) -> int:
    return len(get_class_names(task))


def get_task_output_dir(
    task: str,
    image_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    vit_depth: Optional[str] = None,
) -> str:
    """Return an isolated output directory for one task/ViT ablation."""
    task_dir = os.path.join(OUTPUT_DIR, task)
    if image_size is None and patch_size is None and vit_depth is None:
        return task_dir
    if image_size is None or patch_size is None or vit_depth is None:
        raise ValueError("image_size, patch_size, and vit_depth must be provided together.")
    return os.path.join(task_dir, f"image{image_size}_patch{patch_size}_{vit_depth}")


DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_VIT_DEPTH = "base"
# Keep this alias as a safe fallback for older saved configurations.
IMAGE_SIZE = DEFAULT_IMAGE_SIZE
VAL_SIZE = 0.1
TEST_SIZE = 0.1
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
PRINT_EVERY = 20

DEFAULT_FINAL_EPOCHS = 30
DEFAULT_TUNE_EPOCHS = 6
DEFAULT_BATCH_SIZE = 16
DEFAULT_ACCUM_STEPS = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_MODEL_NAME = "vit_base_patch16_224"

OUTPUT_DIR = "output/colorectal_exp_auto"
RAY_DIR = OUTPUT_DIR
AK_DIR = OUTPUT_DIR
FINAL_DIR = OUTPUT_DIR


def get_vit_model_name(vit_depth: str, patch_size: int) -> str:
    """Map the architecture-ablation choices to the corresponding timm ViT name."""
    valid_depths = {"tiny", "small", "base"}
    if vit_depth not in valid_depths:
        raise ValueError(f"Unknown ViT depth '{vit_depth}'. Choose tiny, small, or base.")
    if patch_size not in {16, 32}:
        raise ValueError(f"Unsupported patch size {patch_size}. Choose 16 or 32.")
    return f"vit_{vit_depth}_patch{patch_size}_224"


def validate_vit_architecture(image_size: int, patch_size: int, vit_depth: str):
    if image_size <= 0:
        raise ValueError("IMAGE_SIZE must be a positive integer.")
    if image_size % patch_size != 0:
        raise ValueError(
            f"IMAGE_SIZE={image_size} must be divisible by patch_size={patch_size}."
        )
    get_vit_model_name(vit_depth, patch_size)


def get_vit_output_dir(args) -> str:
    return get_task_output_dir(args.task, args.image_size, args.patch_size, args.vit_depth)


# -----------------------------------------------------------------------------
# Reproducibility and image processing helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_bilinear_uint8_hwc(img_uint8_hwc: np.ndarray, out_size: int) -> np.ndarray:
    t = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()
    return t.numpy()


def random_small_rotation(img: np.ndarray) -> np.ndarray:
    k = np.random.choice([0, 1, 3], p=[0.5, 0.25, 0.25])
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def random_crop_and_resize(img: np.ndarray, out_size: int, min_crop_ratio: float = 0.9) -> np.ndarray:
    h, w, _ = img.shape
    crop_ratio = np.random.uniform(min_crop_ratio, 1.0)
    crop_h = max(16, int(h * crop_ratio))
    crop_w = max(16, int(w * crop_ratio))
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    cropped = img[top:top + crop_h, left:left + crop_w]
    return resize_bilinear_uint8_hwc(cropped, out_size)


def random_color_jitter(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    brightness = np.random.uniform(0.9, 1.1)
    contrast = np.random.uniform(0.9, 1.1)
    channel_scale = np.random.uniform(0.95, 1.05, size=(1, 1, 3))
    mean = img_f.mean(axis=(0, 1), keepdims=True)
    img_f = (img_f - mean) * contrast + mean
    img_f = img_f * brightness
    img_f = img_f * channel_scale
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


class ColorectalDataset(Dataset):
    """PyTorch Dataset for the timm ViT route."""

    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool, out_size: int):
        self.x = x
        self.y = y
        self.train = train
        self.out_size = out_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        label = int(self.y[i])

        if self.train:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
            if np.random.rand() < 0.5:
                img = random_small_rotation(img)
            if np.random.rand() < 0.5:
                img = random_color_jitter(img)
            img = random_crop_and_resize(img, self.out_size, min_crop_ratio=0.9)
        else:
            img = resize_bilinear_uint8_hwc(img, self.out_size)

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# Data loading and splitting
# -----------------------------------------------------------------------------
def prepare_splits(
    x_all: np.ndarray,
    y_all: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_all,
        y_all,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all,
    )

    val_ratio_within_train_val = VAL_SIZE / (1.0 - TEST_SIZE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio_within_train_val,
        random_state=SEED,
        stratify=y_train_val,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_npz_data(npz_path: str = NPZ_PATH, task: str = "multiclass"):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found. Put the .npz file under data/ or pass --npz-path.")

    data = np.load(npz_path)
    x_all = data["x"].astype(np.uint8)
    y_all = data["y"].astype(np.int64)

    if task == "binary":
        # Patch-level binary cancer detection: tumor vs non-tumor.
        # This should be described as patch-level tumor detection, not patient-level
        # diseased vs healthy classification.
        y_all = (y_all == TUMOR_CLASS_INDEX).astype(np.int64)
    elif task != "multiclass":
        raise ValueError(f"Unknown task: {task}. Use 'multiclass' or 'binary'.")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    return prepare_splits(x_all, y_all)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_two_curves(train_values, val_values, ylabel: str, title: str, out_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_values, label=f"train_{ylabel}")
    plt.plot(val_values, label=f"val_{ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_one_curve(values, ylabel: str, title: str, out_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(values, label=ylabel)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = os.path.join(out_dir, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names, out_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    path = os.path.join(out_dir, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()



def compute_roc_metrics(y_true, y_score, num_classes: int) -> Dict:
    """Compute ROC-AUC values for binary or multiclass classification."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    metrics = {}
    try:
        if num_classes == 2:
            positive_score = y_score[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive_score))
        else:
            metrics["roc_auc_macro_ovr"] = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            )
            metrics["roc_auc_weighted_ovr"] = float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
            )
    except Exception as e:
        metrics["roc_auc_error"] = str(e)
    return metrics


def compute_pr_metrics(y_true, y_score, num_classes: int, class_names: List[str]) -> Dict:
    """Compute PR-AUC / Average Precision metrics.

    For binary tumor detection, this emphasises the positive tumor class.
    For multiclass, macro and weighted one-vs-rest average precision are reported.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    metrics = {}

    try:
        if num_classes == 2:
            positive_score = y_score[:, 1]
            metrics["pr_auc_tumor"] = float(average_precision_score(y_true, positive_score))
            precision, recall, _ = precision_recall_curve(y_true, positive_score)
            metrics["pr_curve_auc_tumor"] = float(auc(recall, precision))
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            metrics["average_precision_macro_ovr"] = float(
                average_precision_score(y_bin, y_score, average="macro")
            )
            metrics["average_precision_weighted_ovr"] = float(
                average_precision_score(y_bin, y_score, average="weighted")
            )
            per_class_ap = {}
            for i, name in enumerate(class_names):
                per_class_ap[name] = float(average_precision_score(y_bin[:, i], y_score[:, i]))
            metrics["average_precision_per_class"] = per_class_ap
    except Exception as e:
        metrics["pr_auc_error"] = str(e)

    return metrics


def compute_binary_focus_metrics(y_true, y_pred, y_score, class_names: List[str]) -> Dict:
    """Return tumor-focused binary metrics for patch-level tumor detection.

    Assumes labels: 0 = non_tumor, 1 = tumor.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    metrics = {}
    if len(class_names) != 2:
        return metrics

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    metrics.update({
        "negative_class": class_names[0],
        "positive_class": class_names[1],
        "tn_non_tumor_correct": int(tn),
        "fp_non_tumor_predicted_as_tumor": int(fp),
        "fn_tumor_predicted_as_non_tumor": int(fn),
        "tp_tumor_correct": int(tp),
        "tumor_precision": float(per_precision[1]),
        "tumor_recall": float(per_recall[1]),
        "tumor_f1": float(per_f1[1]),
        "tumor_support": int(per_support[1]),
        "non_tumor_precision": float(per_precision[0]),
        "non_tumor_recall": float(per_recall[0]),
        "non_tumor_f1": float(per_f1[0]),
        "non_tumor_support": int(per_support[0]),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
    })

    if y_score.ndim == 2 and y_score.shape[1] >= 2:
        metrics.update(compute_pr_metrics(y_true, y_score, 2, class_names))

    return metrics


def make_class_weight_tensor(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Balanced class weights for CrossEntropyLoss.

    weight_c = N / (num_classes * count_c)
    This is useful for the binary tumor-vs-non-tumor task because tumor is about
    one eighth of the original balanced 8-class dataset after merging labels.
    """
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_keras_class_weight_dict(y_train: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Balanced class weights for Keras/AutoKeras fit(class_weight=...)."""
    counts = np.bincount(y_train.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return {int(i): float(w) for i, w in enumerate(weights)}


def plot_roc_curves(y_true, y_score, class_names, out_name: str, out_dir: str):
    """Save binary or one-vs-rest multiclass ROC curves."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = len(class_names)

    plt.figure(figsize=(7, 6))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"tumor ROC-AUC = {roc_auc:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for i, name in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}", linewidth=1.2)
            except Exception:
                continue

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(True)
    path = os.path.join(out_dir, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_pr_curves(y_true, y_score, class_names, out_name: str, out_dir: str):
    """Save binary or one-vs-rest multiclass precision-recall curves."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    num_classes = len(class_names)

    plt.figure(figsize=(7, 6))
    if num_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_score[:, 1])
        ap = average_precision_score(y_true, y_score[:, 1])
        plt.plot(recall, precision, label=f"tumor AP / PR-AUC = {ap:.3f}")
        baseline = float(np.mean(y_true == 1))
        plt.axhline(baseline, linestyle="--", linewidth=1, label=f"tumor prevalence = {baseline:.3f}")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for i, name in enumerate(class_names):
            try:
                precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                ap = average_precision_score(y_bin[:, i], y_score[:, i])
                plt.plot(recall, precision, label=f"{name} AP={ap:.3f}", linewidth=1.2)
            except Exception:
                continue

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(fontsize=8, loc="lower left")
    plt.grid(True)
    path = os.path.join(out_dir, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_misclassified_patches(
    x_raw: np.ndarray,
    y_true,
    y_pred,
    y_score,
    class_names,
    out_name: str,
    out_dir: str,
    max_images: int = 16,
):
    """Save a grid of misclassified raw patches with true/predicted labels."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    wrong_idx = np.where(y_true != y_pred)[0]

    summary_path = os.path.join(out_dir, out_name.replace(".png", "_summary.json"))
    summary = {
        "num_test_samples": int(len(y_true)),
        "num_misclassified": int(len(wrong_idx)),
        "saved_examples": int(min(len(wrong_idx), max_images)),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", summary_path)

    if len(wrong_idx) == 0:
        print("No misclassified patches to plot.")
        return

    chosen = wrong_idx[:max_images]
    n = len(chosen)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(3.2 * cols, 3.6 * rows))
    for plot_i, idx in enumerate(chosen, start=1):
        img = x_raw[idx]
        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])
        conf = float(np.max(y_score[idx])) if y_score.ndim == 2 else float("nan")
        plt.subplot(rows, cols, plot_i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"T: {class_names[true_label]}\nP: {class_names[pred_label]} ({conf:.2f})",
            fontsize=9,
        )
    plt.tight_layout()
    path = os.path.join(out_dir, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def _preprocess_one_image_for_vit(img_uint8_hwc: np.ndarray, out_size: int) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = resize_bilinear_uint8_hwc(img_uint8_hwc, out_size)
    img_f = img.astype(np.float32) / 255.0
    img_chw = img_f.transpose(2, 0, 1)
    img_chw = (img_chw - mean[:, None, None]) / std[:, None, None]
    return torch.from_numpy(img_chw).unsqueeze(0)


def save_vit_attention_map(
    model: nn.Module,
    x_raw: np.ndarray,
    y_true,
    y_pred,
    class_names,
    out_dir: str,
    image_size: int,
    out_name: str,
    image_index: int,
    selection_title: str,
):
    """Save one approximate ViT attention-rollout map for a selected test patch.

    The map is a qualitative self-attention rollout reconstructed from timm ViT
    QKV tensors. It is not a calibrated clinical explanation.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not hasattr(model, "blocks"):
        print("Attention map skipped: model has no ViT blocks attribute.")
        return False

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if image_index < 0 or image_index >= len(x_raw):
        raise IndexError(f"Invalid attention-map image index: {image_index}")

    attentions = []
    hooks = []

    def make_hook(attn_module):
        def hook(module, inputs, output):
            # output shape: [B, N, 3*C]
            batch_size, num_tokens, three_channels = output.shape
            num_heads = attn_module.num_heads
            head_dim = three_channels // 3 // num_heads
            qkv = output.reshape(batch_size, num_tokens, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            attention = (q @ k.transpose(-2, -1)) * attn_module.scale
            attentions.append(attention.softmax(dim=-1).detach().cpu())
        return hook

    try:
        for block in model.blocks:
            if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
                hooks.append(block.attn.qkv.register_forward_hook(make_hook(block.attn)))

        device = next(model.parameters()).device
        model.eval()
        xb = _preprocess_one_image_for_vit(x_raw[image_index], image_size).to(device)
        with torch.no_grad():
            _ = model(xb)
    finally:
        for hook_handle in hooks:
            hook_handle.remove()

    if not attentions:
        print("Attention map skipped: no attention tensors captured.")
        return False

    rollout = None
    for attention in attentions:
        # [B, heads, tokens, tokens] -> [tokens, tokens]
        attention_mean = attention[0].mean(dim=0)
        attention_augmented = attention_mean + torch.eye(attention_mean.size(0))
        attention_augmented = attention_augmented / attention_augmented.sum(dim=-1, keepdim=True)
        rollout = attention_augmented if rollout is None else attention_augmented @ rollout

    class_attention = rollout[0, 1:].numpy()
    grid_size = int(np.sqrt(class_attention.shape[0]))
    if grid_size * grid_size != class_attention.shape[0]:
        print("Attention map skipped: patch token count is not square.")
        return False

    attention_map = class_attention.reshape(grid_size, grid_size)
    attention_map = torch.tensor(attention_map)[None, None, :, :].float()
    attention_map = F.interpolate(
        attention_map,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    attention_map = attention_map.squeeze().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

    raw_resized = resize_bilinear_uint8_hwc(x_raw[image_index], image_size)
    true_label = int(y_true[image_index])
    pred_label = int(y_pred[image_index])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(raw_resized)
    plt.axis("off")
    plt.title("Original patch")

    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap="inferno")
    plt.axis("off")
    plt.title("Attention rollout")

    plt.subplot(1, 3, 3)
    plt.imshow(raw_resized)
    plt.imshow(attention_map, cmap="inferno", alpha=0.45)
    plt.axis("off")
    plt.title(f"{selection_title}\nT: {class_names[true_label]} | P: {class_names[pred_label]}")

    plt.tight_layout()
    output_path = os.path.join(out_dir, out_name)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved:", output_path)
    plt.close()
    return True


def save_vit_attention_examples(
    model: nn.Module,
    x_raw: np.ndarray,
    y_true,
    y_pred,
    class_names: List[str],
    out_dir: str,
    image_size: int,
):
    """Save tumor/error examples plus one representative attention map per class."""
    attention_dir = os.path.join(out_dir, "attention_maps")
    os.makedirs(attention_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    records = []

    def save_selection(name: str, criteria: str, candidates: np.ndarray):
        candidates = np.asarray(candidates, dtype=int)
        record = {
            "name": name,
            "criteria": criteria,
            "status": "not_available",
            "test_index": None,
            "true_label": None,
            "predicted_label": None,
        }
        if candidates.size == 0:
            print(f"Attention example unavailable: {name} ({criteria})")
            records.append(record)
            return

        image_index = int(candidates[0])
        record.update({
            "status": "saved",
            "test_index": image_index,
            "true_label": class_names[int(y_true[image_index])],
            "predicted_label": class_names[int(y_pred[image_index])],
        })
        saved = save_vit_attention_map(
            model=model,
            x_raw=x_raw,
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            out_dir=attention_dir,
            image_size=image_size,
            out_name=f"attention_{name}.png",
            image_index=image_index,
            selection_title=name.replace("_", " "),
        )
        if not saved:
            record["status"] = "skipped"
        records.append(record)

    if "tumor" in class_names:
        tumor_index = class_names.index("tumor")
        tumor_true = y_true == tumor_index
        tumor_pred = y_pred == tumor_index
        prediction_error = y_true != y_pred
        save_selection(
            "correct_tumor_patch",
            "true=tumor and predicted=tumor",
            np.flatnonzero(tumor_true & tumor_pred),
        )
        save_selection(
            "misclassified_tumor_patch",
            "prediction is wrong and tumor is either the true or predicted class",
            np.flatnonzero(prediction_error & (tumor_true | tumor_pred)),
        )
        save_selection(
            "false_positive_non_tumor_patch",
            "true=non_tumor and predicted=tumor",
            np.flatnonzero((~tumor_true) & tumor_pred),
        )
        save_selection(
            "false_negative_tumor_patch",
            "true=tumor and predicted=non_tumor",
            np.flatnonzero(tumor_true & (~tumor_pred)),
        )

    for class_index, class_name in enumerate(class_names):
        correct_candidates = np.flatnonzero((y_true == class_index) & (y_pred == class_index))
        # Prefer a correct prediction for a class-level example; fall back to a true sample.
        candidates = correct_candidates if correct_candidates.size else np.flatnonzero(y_true == class_index)
        save_selection(
            f"class_{class_name}",
            f"representative true={class_name} patch (correct prediction preferred)",
            candidates,
        )

    summary_path = os.path.join(attention_dir, "attention_examples_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump({"image_size": image_size, "examples": records}, summary_file, indent=2)
    print("Saved:", summary_path)


# -----------------------------------------------------------------------------
# PyTorch / timm / Ray Tune route
# -----------------------------------------------------------------------------
def build_model(config: Dict, device: torch.device, num_classes: int):
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
    patch_size = int(config.get("patch_size", DEFAULT_PATCH_SIZE))
    vit_depth = str(config.get("vit_depth", DEFAULT_VIT_DEPTH))
    validate_vit_architecture(image_size, patch_size, vit_depth)
    model_name = config.get("model_name", get_vit_model_name(vit_depth, patch_size))
    return timm.create_model(
        model_name,
        pretrained=True,
        num_classes=int(num_classes),
        img_size=image_size,
        drop_rate=config.get("drop_rate", 0.0),
        drop_path_rate=config.get("drop_path_rate", 0.0),
    ).to(device)


@torch.no_grad()
def evaluate_pytorch_model(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    model.eval()
    total = 0
    loss_sum = 0.0
    y_true = []
    y_pred = []
    y_score = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast_ctx(enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = criterion(logits, yb)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        loss_sum += loss.item() * xb.size(0)
        total += xb.size(0)
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_score.extend(probs.cpu().numpy().tolist())

    y_score = np.asarray(y_score)
    num_classes = len(class_names)
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cls_report = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    roc_metrics = compute_roc_metrics(y_true, y_score, num_classes)
    pr_metrics = compute_pr_metrics(y_true, y_score, num_classes, class_names)
    binary_focus_metrics = compute_binary_focus_metrics(y_true, y_pred, y_score, class_names)

    return {
        "loss": loss_sum / total,
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_metrics": roc_metrics,
        "pr_metrics": pr_metrics,
        "binary_focus_metrics": binary_focus_metrics,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score.tolist(),
        "classification_report": cls_report,
    }


def report_to_ray(metrics: Dict):
    """Report metrics to Ray Tune across Ray versions.

    Newer Ray versions expect report(metrics) with a single dictionary.
    Older Ray Tune versions used report(**metrics). Try the new form first.
    """
    from ray import tune

    try:
        tune.report(metrics)
    except TypeError:
        tune.report(**metrics)


def train_vit_one_config(
    config: Dict,
    npz_path: str = NPZ_PATH,
    out_dir: str = FINAL_DIR,
    save_outputs: bool = False,
    report_ray: bool = False,
    task: str = "multiclass",
):
    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    class_names = get_class_names(task)
    num_classes = len(class_names)
    x_train, x_val, x_test, y_train, y_val, y_test = load_npz_data(npz_path, task=task)

    train_counts = np.bincount(y_train.astype(int), minlength=num_classes)
    val_counts = np.bincount(y_val.astype(int), minlength=num_classes)
    test_counts = np.bincount(y_test.astype(int), minlength=num_classes)
    class_distribution = {
        "train": {class_names[i]: int(train_counts[i]) for i in range(num_classes)},
        "val": {class_names[i]: int(val_counts[i]) for i in range(num_classes)},
        "test": {class_names[i]: int(test_counts[i]) for i in range(num_classes)},
    }

    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
    train_ds = ColorectalDataset(x_train, y_train, train=True, out_size=image_size)
    val_ds = ColorectalDataset(x_val, y_val, train=False, out_size=image_size)
    test_ds = ColorectalDataset(x_test, y_test, train=False, out_size=image_size)

    batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    accum_steps = int(config.get("accum_steps", DEFAULT_ACCUM_STEPS))
    num_epochs = int(config.get("num_epochs", DEFAULT_TUNE_EPOCHS))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = build_model(config, device, num_classes=num_classes)
    use_class_weights = bool(config.get("use_class_weights", False))
    class_weight_tensor = make_class_weight_tensor(y_train, num_classes, device) if use_class_weights else None
    if use_class_weights:
        print("Using class weights:", class_weight_tensor.detach().cpu().numpy().tolist())
    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=float(config.get("label_smoothing", DEFAULT_LABEL_SMOOTHING)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", DEFAULT_LR)),
        weight_decay=float(config.get("weight_decay", DEFAULT_WEIGHT_DECAY)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = get_grad_scaler(device)

    history = {
        "loss": [], "acc": [], "val_loss": [], "val_acc": [],
        "val_f1_macro": [], "lr": []
    }
    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(out_dir, "checkpoint_raytuned_vit_best.pt")

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Config:", json.dumps(config, indent=2))
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    print("Class distribution:", json.dumps(class_distribution, indent=2))
    print(f"Effective batch = {batch_size} * {accum_steps} = {batch_size * accum_steps}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with autocast_ctx(enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(1) == yb).sum().item()
            running_total += xb.size(0)

            loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if len(train_loader) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = evaluate_pytorch_model(model, val_loader, criterion, device, class_names)
        scheduler.step()

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            if save_outputs:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_f1_macro": best_val_f1,
                    "history": history,
                    "class_names": class_names,
                    "task": task,
                }, best_ckpt_path)

        ray_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
            "val_f1_macro": float(val_metrics["f1_macro"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"time={time.time() - t0:.1f}s"
        )

        if report_ray:
            report_to_ray(ray_metrics)

    final_results = {
        "task": task,
        "class_names": class_names,
        "config": config,
        "best_epoch": best_epoch,
        "best_val_f1_macro": float(best_val_f1),
        "history": history,
        "class_distribution": class_distribution,
        "use_class_weights": bool(config.get("use_class_weights", False)),
    }

    if save_outputs:
        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = evaluate_pytorch_model(model, test_loader, criterion, device, class_names)
        print("\nFinal test results from Ray-tuned ViT:")
        print(f"test_loss            : {test_metrics['loss']:.4f}")
        print(f"test_accuracy        : {test_metrics['acc']:.4f}")
        print(f"test_precision_macro : {test_metrics['precision_macro']:.4f}")
        print(f"test_recall_macro    : {test_metrics['recall_macro']:.4f}")
        print(f"test_f1_macro        : {test_metrics['f1_macro']:.4f}")
        print("test_confusion_matrix:")
        print(test_metrics["cm"])
        if task == "binary":
            print("\nBinary tumor-focused metrics:")
            print(json.dumps(test_metrics["binary_focus_metrics"], indent=2))

        print("\nClassification report:")
        print(classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        ))

        final_results.update({
            "test_loss": float(test_metrics["loss"]),
            "test_accuracy": float(test_metrics["acc"]),
            "test_precision_macro": float(test_metrics["precision_macro"]),
            "test_recall_macro": float(test_metrics["recall_macro"]),
            "test_f1_macro": float(test_metrics["f1_macro"]),
            "test_roc_metrics": test_metrics["roc_metrics"],
            "test_pr_metrics": test_metrics["pr_metrics"],
            "test_binary_focus_metrics": test_metrics["binary_focus_metrics"],
            "test_confusion_matrix": test_metrics["cm"].tolist(),
            "classification_report": test_metrics["classification_report"],
        })

        result_path = os.path.join(out_dir, "results_raytuned_vit_final.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2)
        print("Saved:", result_path)

        plot_two_curves(history["loss"], history["val_loss"], "loss", "Ray-tuned ViT Train vs Val Loss", "raytuned_loss.png", out_dir)
        plot_two_curves(history["acc"], history["val_acc"], "acc", "Ray-tuned ViT Train vs Val Accuracy", "raytuned_accuracy.png", out_dir)
        plot_one_curve(history["val_f1_macro"], "val_f1_macro", "Ray-tuned ViT Validation Macro-F1", "raytuned_val_f1_macro.png", out_dir)
        plot_one_curve(history["lr"], "lr", "Ray-tuned ViT Learning Rate", "raytuned_lr.png", out_dir)
        plot_confusion_matrix(test_metrics["cm"], class_names, "raytuned_confusion_matrix.png", out_dir)
        plot_roc_curves(test_metrics["y_true"], test_metrics["y_score"], class_names, "raytuned_roc_curve.png", out_dir)
        plot_pr_curves(test_metrics["y_true"], test_metrics["y_score"], class_names, "raytuned_pr_curve.png", out_dir)
        plot_misclassified_patches(
            x_test,
            test_metrics["y_true"],
            test_metrics["y_pred"],
            test_metrics["y_score"],
            class_names,
            "raytuned_misclassified_patches.png",
            out_dir,
            max_images=16,
        )
        save_vit_attention_examples(
            model=model,
            x_raw=x_test,
            y_true=test_metrics["y_true"],
            y_pred=test_metrics["y_pred"],
            class_names=class_names,
            out_dir=out_dir,
            image_size=image_size,
        )

    return final_results


def run_ray_tune(args):
    from ray import tune
    from ray.tune import RunConfig
    from ray.tune.schedulers import ASHAScheduler

    out_dir = get_vit_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    # Ray 2.x/3.x compatibility: when passing RunConfig to tune.Tuner,
    # import it from ray.tune and set verbose as an integer.
    # This avoids AttributeError: 'str' object has no attribute 'value'.
    run_config = RunConfig(
        storage_path=os.path.abspath(out_dir),
        name="ray_tune_trials",
        verbose=1,
    )

    # Architecture is fixed for a single ablation run. Ray Tune searches only
    # optimization/regularization settings, so results are comparable across models.
    search_space = {
        "model_name": args.model_name,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "vit_depth": args.vit_depth,
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "batch_size": tune.choice([8, 16]),
        "accum_steps": tune.choice([4, 8]),
        "label_smoothing": tune.choice([0.0, 0.05, 0.1]),
        "drop_rate": tune.choice([0.0, 0.1, 0.2]),
        "drop_path_rate": tune.choice([0.0, 0.05, 0.1]),
        "num_epochs": args.tune_epochs,
        "use_class_weights": args.use_class_weights,
    }

    scheduler = ASHAScheduler(
        metric="val_f1_macro",
        mode="max",
        max_t=args.tune_epochs,
        grace_period=max(1, min(3, args.tune_epochs)),
        reduction_factor=2,
    )

    def trainable(config):
        train_vit_one_config(
            config=config,
            npz_path=args.npz_path,
            out_dir=out_dir,
            save_outputs=False,
            report_ray=True,
            task=args.task,
        )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=args.num_samples,
        ),
        run_config=run_config,
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val_f1_macro", mode="max")
    best_config = dict(best_result.config)

    best_path = os.path.join(out_dir, "best_ray_config.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)

    summary_path = os.path.join(out_dir, "best_ray_result_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_config": best_config,
            "best_metrics": best_result.metrics,
        }, f, indent=2, default=str)

    print("\nBest Ray Tune config:")
    print(json.dumps(best_config, indent=2))
    print("Saved:", best_path)
    print("Saved:", summary_path)


def run_final_train(args):
    out_dir = get_vit_output_dir(args)
    if args.config_json is None:
        args.config_json = os.path.join(out_dir, "best_ray_config.json")

    if not os.path.exists(args.config_json):
        raise FileNotFoundError(f"Config JSON not found: {args.config_json}. Run --mode ray_tune first.")

    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)

    requested_architecture = {
        "model_name": args.model_name,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "vit_depth": args.vit_depth,
    }
    for key, expected_value in requested_architecture.items():
        saved_value = config.get(key)
        if saved_value is None:
            config[key] = expected_value
        elif saved_value != expected_value:
            raise ValueError(
                f"Saved config {key}={saved_value!r} does not match the requested "
                f"architecture value {expected_value!r}. Use the matching options."
            )

    config["num_epochs"] = args.final_epochs
    # Allow --use-class-weights to enable class weighting even when the JSON
    # config was produced by an older run without this field.
    if args.use_class_weights:
        config["use_class_weights"] = True
    train_vit_one_config(
        config=config,
        npz_path=args.npz_path,
        out_dir=out_dir,
        save_outputs=True,
        report_ray=False,
        task=args.task,
    )


# -----------------------------------------------------------------------------
# AutoKeras baseline route
# -----------------------------------------------------------------------------
def resize_images_for_keras(x: np.ndarray, image_size: int):
    """Resize HWC uint8 images for AutoKeras using TensorFlow."""
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    x_tf = tf.image.resize(x_tf, (image_size, image_size), method="bilinear")
    x_tf = tf.clip_by_value(x_tf / 255.0, 0.0, 1.0)
    return x_tf.numpy().astype("float32")


def run_autokeras(args):
    # Import inside the function so PyTorch/Ray modes do not require TensorFlow/AutoKeras.
    import tensorflow as tf
    import autokeras as ak

    # Compatibility patch for AutoKeras + Keras 3.
    # Some AutoKeras versions pass numpy integer class counts into Dense(units=...),
    # which Keras 3 may reject even when it prints as units=8.
    try:
        import keras
        import numpy as _np

        _OriginalDense = keras.layers.Dense

        def _SafeDense(units, *dense_args, **dense_kwargs):
            if isinstance(units, _np.integer):
                units = int(units)
            return _OriginalDense(units, *dense_args, **dense_kwargs)

        keras.layers.Dense = _SafeDense
        try:
            import autokeras.blocks.heads as _ak_heads
            _ak_heads.layers.Dense = _SafeDense
        except Exception:
            pass
    except Exception:
        pass

    set_seed(SEED)
    tf.random.set_seed(SEED)
    out_dir = get_task_output_dir(args.task)
    class_names = get_class_names(args.task)
    num_classes = len(class_names)
    os.makedirs(out_dir, exist_ok=True)

    x_train, x_val, x_test, y_train, y_val, y_test = load_npz_data(args.npz_path, task=args.task)

    # Optional quick subset for first debugging runs.
    if args.ak_max_samples is not None and args.ak_max_samples > 0:
        n = min(args.ak_max_samples, len(x_train))
        x_train = x_train[:n]
        y_train = y_train[:n]
        print(f"AutoKeras quick mode: using first {n} training samples.")

    print("Resizing images for AutoKeras...")
    x_train_ak = resize_images_for_keras(x_train, args.ak_image_size)
    x_val_ak = resize_images_for_keras(x_val, args.ak_image_size)
    x_test_ak = resize_images_for_keras(x_test, args.ak_image_size)

    print("AutoKeras input shapes:")
    print("x_train:", x_train_ak.shape, "y_train:", y_train.shape)
    print("x_val  :", x_val_ak.shape, "y_val  :", y_val.shape)
    print("x_test :", x_test_ak.shape, "y_test :", y_test.shape)

    clf = ak.ImageClassifier(
        num_classes=int(num_classes),
        objective="val_accuracy",
        max_trials=args.ak_max_trials,
        overwrite=True,
        seed=SEED,
        directory=out_dir,
        project_name="autokeras_colorectal_baseline",
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.ak_patience,
        restore_best_weights=True,
    )

    class_weight = make_keras_class_weight_dict(y_train, num_classes) if args.use_class_weights else None
    if class_weight is not None:
        print("Using AutoKeras class weights:", class_weight)

    clf.fit(
        x_train_ak,
        y_train,
        epochs=args.ak_epochs,
        validation_data=(x_val_ak, y_val),
        callbacks=[early_stop],
        batch_size=args.ak_batch_size,
        class_weight=class_weight,
    )

    eval_result = clf.evaluate(x_test_ak, y_test)
    print("\nAutoKeras test evaluation:")
    print(eval_result)

    model = clf.export_model()
    raw_score = model.predict(x_test_ak, verbose=0)
    raw_score = np.asarray(raw_score)

    if raw_score.ndim > 1 and raw_score.shape[-1] == num_classes:
        y_score = raw_score
        # Convert logits to probabilities if needed.
        row_sums = y_score.sum(axis=1)
        if np.any(y_score < 0) or not np.allclose(row_sums, 1.0, atol=1e-2):
            exp_score = np.exp(y_score - np.max(y_score, axis=1, keepdims=True))
            y_score = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        y_pred = np.argmax(y_score, axis=-1).reshape(-1)
    else:
        raw_pred = clf.predict(x_test_ak)
        raw_pred = np.asarray(raw_pred)
        y_pred = raw_pred.reshape(-1).astype(int)
        y_score = np.zeros((len(y_pred), num_classes), dtype=np.float32)
        y_score[np.arange(len(y_pred)), y_pred] = 1.0

    roc_metrics = compute_roc_metrics(y_test, y_score, num_classes)
    pr_metrics = compute_pr_metrics(y_test, y_score, num_classes, class_names)
    binary_focus_metrics = compute_binary_focus_metrics(y_test, y_pred, y_score, class_names)

    report = classification_report(
        y_test, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

    print("\nAutoKeras classification report:")
    print(report)
    print("AutoKeras confusion matrix:")
    print(cm)
    if args.task == "binary":
        print("\nAutoKeras binary tumor-focused metrics:")
        print(json.dumps(binary_focus_metrics, indent=2))

    model_path = os.path.join(out_dir, "autokeras_colorectal_best.keras")
    model.save(model_path)
    print("Saved exported AutoKeras model:", model_path)

    results = {
        "dataset": "colorectal_histology",
        "model": "AutoKeras ImageClassifier baseline",
        "task": args.task,
        "class_names": class_names,
        "ak_image_size": args.ak_image_size,
        "ak_max_trials": args.ak_max_trials,
        "ak_epochs": args.ak_epochs,
        "ak_batch_size": args.ak_batch_size,
        "eval_result": str(eval_result),
        "roc_metrics": roc_metrics,
        "pr_metrics": pr_metrics,
        "binary_focus_metrics": binary_focus_metrics,
        "use_class_weights": bool(args.use_class_weights),
        "classification_report_text": report,
        "confusion_matrix": cm.tolist(),
    }
    result_path = os.path.join(out_dir, "results_autokeras_baseline.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", result_path)

    plot_confusion_matrix(cm, class_names, "autokeras_confusion_matrix.png", out_dir)
    plot_roc_curves(y_test, y_score, class_names, "autokeras_roc_curve.png", out_dir)
    plot_pr_curves(y_test, y_score, class_names, "autokeras_pr_curve.png", out_dir)
    plot_misclassified_patches(
        x_test,
        y_test,
        y_pred,
        y_score,
        class_names,
        "autokeras_misclassified_patches.png",
        out_dir,
        max_images=16,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def prompt_task_choice() -> str:
    options = [("binary", "binary"), ("multiclass", "multiclass")]
    print("\nSelect classification type:")
    for index, (_, label) in enumerate(options, start=1):
        print(f"  {index}. {label}")

    valid_by_index = {str(i): value for i, (value, _) in enumerate(options, start=1)}
    valid_by_value = {value.lower(): value for value, _ in options}
    while True:
        raw = input("Please enter the option number or name: ").strip().lower()
        if raw in valid_by_index:
            return valid_by_index[raw]
        if raw in valid_by_value:
            return valid_by_value[raw]
        print("Invalid input, please try again.")


def prompt_image_size() -> int:
    while True:
        raw = input(
            f"IMAGE_SIZE [default {DEFAULT_IMAGE_SIZE}; examples: 128, 224, 384]: "
        ).strip()
        if not raw:
            return DEFAULT_IMAGE_SIZE
        try:
            image_size = int(raw)
        except ValueError:
            print("IMAGE_SIZE must be an integer.")
            continue
        if image_size > 0:
            return image_size
        print("IMAGE_SIZE must be positive.")


def prompt_architecture_choice(label: str, choices: List, default):
    print(f"\nSelect {label}:")
    for index, choice in enumerate(choices, start=1):
        suffix = " (default)" if choice == default else ""
        print(f"  {index}. {choice}{suffix}")

    values_by_index = {str(i): value for i, value in enumerate(choices, start=1)}
    values_by_name = {str(value).lower(): value for value in choices}
    while True:
        raw = input(f"Enter option number or value [default {default}]: ").strip().lower()
        if not raw:
            return default
        if raw in values_by_index:
            return values_by_index[raw]
        if raw in values_by_name:
            return values_by_name[raw]
        print("Invalid input, please try again.")


def resolve_task(args) -> str:
    return args.task if args.task is not None else prompt_task_choice()


def resolve_vit_architecture(args):
    """Resolve interactive/CLI ViT ablation values and keep them on args."""
    if args.image_size is None:
        args.image_size = prompt_image_size()
    if args.patch_size is None:
        args.patch_size = prompt_architecture_choice(
            "patch size", [16, 32], DEFAULT_PATCH_SIZE
        )
    if args.vit_depth is None:
        args.vit_depth = prompt_architecture_choice(
            "ViT scale (tiny/small/base)", ["tiny", "small", "base"], DEFAULT_VIT_DEPTH
        )

    args.image_size = int(args.image_size)
    args.patch_size = int(args.patch_size)
    args.vit_depth = str(args.vit_depth)
    validate_vit_architecture(args.image_size, args.patch_size, args.vit_depth)
    args.model_name = get_vit_model_name(args.vit_depth, args.patch_size)
    print(
        "Selected ViT architecture: "
        f"{args.model_name} | IMAGE_SIZE={args.image_size}"
    )


def save_pipeline_selection(args, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    selection_path = os.path.join(out_dir, "pipeline_selection.json")
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "task": args.task,
            "architecture": {
                "model_name": args.model_name,
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "vit_depth": args.vit_depth,
            },
            "use_class_weights": bool(args.use_class_weights),
            "num_samples": args.num_samples,
            "tune_epochs": args.tune_epochs,
            "final_epochs": args.final_epochs,
        }, f, indent=2)
    print("Saved:", selection_path)


def run_auto_train_pipeline(args):
    args.task = resolve_task(args)
    resolve_vit_architecture(args)
    if args.task == "binary" and not args.use_class_weights:
        args.use_class_weights = True
        print("Enabled --use-class-weights automatically for binary classification.")

    out_dir = get_vit_output_dir(args)
    save_pipeline_selection(args, out_dir)
    print(f"\nStarting auto training pipeline for task: {args.task}")
    print("Stage 1/2: Ray Tune hyperparameter search")
    run_ray_tune(args)
    args.config_json = os.path.join(out_dir, "best_ray_config.json")
    print("Stage 2/2: Final full training with the best Ray Tune config")
    run_final_train(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Colorectal histology ViT tuning with Ray Tune and AutoKeras baseline.")
    parser.add_argument("--mode", type=str, default="auto_train", choices=["auto_train", "ray_tune", "final_train", "autokeras"],
                        help="auto_train: choose task and ViT ablation options then run Ray Tune and final training automatically; ray_tune: tune one PyTorch ViT architecture; final_train: train the matching best Ray config fully; autokeras: run AutoKeras baseline")
    parser.add_argument("--npz-path", type=str, default=NPZ_PATH)
    parser.add_argument("--task", type=str, default=None, choices=["multiclass", "binary"],
                        help="multiclass: original 8-class task; binary: patch-level tumor vs non-tumor task. If omitted in auto_train mode, the script will ask interactively.")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="Use balanced class weights. Recommended for binary tumor-vs-non-tumor because tumor is the minority class.")

    # ViT architecture ablation options. Omit them in auto_train mode to choose interactively.
    parser.add_argument("--image-size", type=int, default=None,
                        help="ViT input size. Must be divisible by patch size, e.g. 128, 224, or 384.")
    parser.add_argument("--patch-size", type=int, default=None, choices=[16, 32],
                        help="ViT patch size: 16 or 32.")
    parser.add_argument("--vit-depth", type=str, default=None, choices=["tiny", "small", "base"],
                        help="ViT scale/capacity ablation (tiny, small, or base).")

    # Ray Tune options
    parser.add_argument("--num-samples", type=int, default=4, help="Number of Ray Tune trials. Increase if you have more time/GPU.")
    parser.add_argument("--tune-epochs", type=int, default=DEFAULT_TUNE_EPOCHS, help="Epochs per Ray Tune trial.")
    parser.add_argument("--cpus-per-trial", type=int, default=2)
    parser.add_argument("--gpus-per-trial", type=float, default=1.0 if torch.cuda.is_available() else 0.0)

    # Final training options
    parser.add_argument("--config-json", type=str, default=None, help="Path to best_ray_config.json from Ray Tune.")
    parser.add_argument("--final-epochs", type=int, default=DEFAULT_FINAL_EPOCHS)

    # AutoKeras options
    parser.add_argument("--ak-max-trials", type=int, default=3)
    parser.add_argument("--ak-epochs", type=int, default=10)
    parser.add_argument("--ak-batch-size", type=int, default=32)
    parser.add_argument("--ak-image-size", type=int, default=128,
                        help="Use 128 for quick runs; 224 is closer to the ViT image size but uses more memory.")
    parser.add_argument("--ak-patience", type=int, default=3)
    parser.add_argument("--ak-max-samples", type=int, default=None,
                        help="Optional training subset for quick AutoKeras debugging, e.g. --ak-max-samples 1000")

    args = parser.parse_args()
    if args.mode != "auto_train" and args.task is None:
        args.task = "multiclass"
    return args


def main():
    args = parse_args()
    if args.mode == "auto_train":
        run_auto_train_pipeline(args)
    elif args.mode == "ray_tune":
        resolve_vit_architecture(args)
        run_ray_tune(args)
    elif args.mode == "final_train":
        resolve_vit_architecture(args)
        run_final_train(args)
    elif args.mode == "autokeras":
        run_autokeras(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
