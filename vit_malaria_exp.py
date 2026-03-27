import os
import time
import json
import inspect
import random
from typing import Dict, Tuple

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
    roc_auc_score,
    classification_report,
)

import timm


# ===========================
# AMP compatibility helpers
# ===========================
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


# ===========================
# Config
# ===========================
SEED = 42
NPZ_PATH = "malaria.npz"

NUM_CLASSES = 2
CLASS_NAMES = ["parasitized", "uninfected"]

# Run modes: "full", "mid", "quick"
RUN_MODE = "full"

MODE_SAMPLE_LIMITS = {
    "full": {"train": None, "val": None, "test": None},
    "mid": {"train": 8000, "val": 1200, "test": 1200},
    "quick": {"train": 2000, "val": 400, "test": 400},
}

NUM_EPOCHS = 8
BATCH_SIZE = 16
ACCUM_STEPS = 1
LR = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
MODEL_NAME = "vit_tiny_patch16_224"

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

VAL_SIZE = 0.1
TEST_SIZE = 0.1

USE_SCHEDULER = True
LABEL_SMOOTHING = 0.1
PATIENCE = 4
PRINT_EVERY = 100

PLOT_DIR = "malaria_exp_full"
RESULT_JSON = os.path.join(PLOT_DIR, "results_malaria_exp_full.json")
CKPT_PATH = os.path.join(PLOT_DIR, "checkpoint_malaria_vit_full_best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Reproducibility
# ===========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# Image helpers
# ===========================
def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def resize_bilinear_uint8_hwc(img_uint8_hwc: np.ndarray, out_size: int) -> np.ndarray:
    t = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()
    return t.numpy()


def random_rotate_90(img: np.ndarray) -> np.ndarray:
    k = np.random.randint(0, 4)
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=k))


def random_crop_and_resize(
    img: np.ndarray,
    out_size: int,
    min_crop_ratio: float = 0.9
) -> np.ndarray:
    h, w, _ = img.shape
    crop_ratio = np.random.uniform(min_crop_ratio, 1.0)
    crop_h = max(8, int(h * crop_ratio))
    crop_w = max(8, int(w * crop_ratio))
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


# ===========================
# Dataset
# ===========================
class MalariaDataset(Dataset):
    def __init__(self, x, y, train: bool, out_size: int):
        self.x = x
        self.y = y
        self.train = train
        self.out_size = out_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = ensure_rgb(self.x[i]).astype(np.uint8)
        label = int(self.y[i])

        if self.train:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
            if np.random.rand() < 0.5:
                img = random_rotate_90(img)
            if np.random.rand() < 0.5:
                img = random_color_jitter(img)
            img = random_crop_and_resize(img, self.out_size, min_crop_ratio=0.9)
        else:
            img = resize_bilinear_uint8_hwc(img, self.out_size)

        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# ===========================
# Plot helpers
# ===========================
def plot_two_curves(train_values, val_values, ylabel: str, title: str, out_name: str):
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(train_values, label=f"train_{ylabel}")
    plt.plot(val_values, label=f"val_{ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_one_curve(values, ylabel: str, title: str, out_name: str):
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(values, label=ylabel)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names, out_name: str):
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(5, 4))
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
    path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


# ===========================
# Split / subset helpers
# ===========================
def apply_mode_limit(x, y, max_samples):
    if max_samples is None or len(x) <= max_samples:
        return x, y

    x_sub, _, y_sub, _ = train_test_split(
        x,
        y,
        train_size=max_samples,
        random_state=SEED,
        stratify=y,
    )
    return x_sub, y_sub


def prepare_splits(
    x_all: np.ndarray,
    y_all: np.ndarray
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

    limits = MODE_SAMPLE_LIMITS[RUN_MODE]
    x_train, y_train = apply_mode_limit(x_train, y_train, limits["train"])
    x_val, y_val = apply_mode_limit(x_val, y_val, limits["val"])
    x_test, y_test = apply_mode_limit(x_test, y_test, limits["test"])

    return x_train, x_val, x_test, y_train, y_val, y_test


# ===========================
# Main
# ===========================
def main():
    set_seed(SEED)
    os.makedirs(PLOT_DIR, exist_ok=True)
    if RUN_MODE not in MODE_SAMPLE_LIMITS:
        raise ValueError(f"RUN_MODE must be one of {list(MODE_SAMPLE_LIMITS.keys())}")

    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"{NPZ_PATH} not found. Run export_tfds_malaria.py first.")

    data = np.load(NPZ_PATH, allow_pickle=True)
    x_all = data["x"]
    y_all = data["y"].astype(np.int64)

    print("Loaded:", NPZ_PATH)
    print("Total samples:", len(x_all))
    print("x dtype:", x_all.dtype)
    print("y shape:", y_all.shape)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_all))
    x_all = x_all[idx]
    y_all = y_all[idx]

    x_train, x_val, x_test, y_train, y_val, y_test = prepare_splits(x_all, y_all)

    print(f"Run mode: {RUN_MODE}")
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    train_ds = MalariaDataset(x_train, y_train, train=True, out_size=IMAGE_SIZE)
    val_ds = MalariaDataset(x_val, y_val, train=False, out_size=IMAGE_SIZE)
    test_ds = MalariaDataset(x_test, y_test, train=False, out_size=IMAGE_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES,
        img_size=IMAGE_SIZE,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = get_grad_scaler(DEVICE)

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        if USE_SCHEDULER else None
    )

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)
    print("Image size:", IMAGE_SIZE)
    print(f"Effective batch = {BATCH_SIZE} * {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS}")

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> Dict:
        model.eval()
        total = 0
        loss_sum = 0.0
        y_true = []
        y_pred = []
        y_prob = []

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            with autocast_ctx(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)

            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = None

        cls_report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        )

        return {
            "loss": loss_sum / total,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "cm": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "classification_report": cls_report,
        }

    history = {
        "loss": [],
        "acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            with autocast_ctx(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            running_loss += loss.item() * xb.size(0)
            running_correct += (logits.argmax(1) == yb).sum().item()
            running_total += xb.size(0)

            loss_scaled = loss / ACCUM_STEPS
            scaler.scale(loss_scaled).backward()

            if step % PRINT_EVERY == 0 or step == len(train_loader):
                avg_loss_so_far = running_loss / running_total
                avg_acc_so_far = running_correct / running_total
                print(
                    f"Epoch {epoch:02d} | Step {step}/{len(train_loader)} "
                    f"| loss={avg_loss_so_far:.4f} | acc={avg_acc_so_far:.4f}"
                )

            if step % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if len(train_loader) % ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = evaluate(val_loader)

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(
            float(val_metrics["roc_auc"]) if val_metrics["roc_auc"] is not None else None
        )
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_metrics["acc"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_metrics["acc"],
                    "val_f1": val_metrics["f1"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "model_name": MODEL_NAME,
                    "image_size": IMAGE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "accum_steps": ACCUM_STEPS,
                    "run_mode": RUN_MODE,
                    "seed": SEED,
                },
                CKPT_PATH,
            )
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()

        roc_auc_text = (
            f"{val_metrics['roc_auc']:.4f}"
            if val_metrics["roc_auc"] is not None else "N/A"
        )

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_roc_auc={roc_auc_text} "
            f"lr={history['lr'][-1]:.6f} time={time.time() - t0:.1f}s"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print("Saved best checkpoint:", CKPT_PATH)

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(test_loader)

    print("\nFinal test results:")
    print(f"test_loss      : {test_metrics['loss']:.4f}")
    print(f"test_accuracy  : {test_metrics['acc']:.4f}")
    print(f"test_precision : {test_metrics['precision']:.4f}")
    print(f"test_recall    : {test_metrics['recall']:.4f}")
    print(f"test_f1        : {test_metrics['f1']:.4f}")
    if test_metrics["roc_auc"] is not None:
        print(f"test_roc_auc   : {test_metrics['roc_auc']:.4f}")
    else:
        print("test_roc_auc   : N/A")
    print("test_confusion_matrix:")
    print(test_metrics["cm"])

    print("\nClassification report:")
    report_text = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print(report_text)

    results = {
        "dataset": "malaria",
        "run_mode": RUN_MODE,
        "seed": SEED,
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "device": str(DEVICE),
        "num_epochs_requested": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * ACCUM_STEPS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "patience": PATIENCE,
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["acc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_roc_auc": (
            float(test_metrics["roc_auc"]) if test_metrics["roc_auc"] is not None else None
        ),
        "test_confusion_matrix": test_metrics["cm"].tolist(),
        "classification_report": test_metrics["classification_report"],
        "history": history,
    }

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved:", RESULT_JSON)

    plot_two_curves(
        history["loss"],
        history["val_loss"],
        "loss",
        "Malaria Train vs Val Loss",
        "malaria_loss.png",
    )
    plot_two_curves(
        history["acc"],
        history["val_acc"],
        "acc",
        "Malaria Train vs Val Accuracy",
        "malaria_accuracy.png",
    )
    plot_one_curve(
        history["val_f1"],
        "val_f1",
        "Malaria Validation F1",
        "malaria_val_f1.png",
    )

    valid_auc_values = [x for x in history["val_roc_auc"] if x is not None]
    if len(valid_auc_values) > 0:
        plot_one_curve(
            history["val_roc_auc"],
            "val_roc_auc",
            "Malaria Validation ROC-AUC",
            "malaria_val_roc_auc.png",
        )

    plot_one_curve(
        history["lr"],
        "lr",
        "Malaria Learning Rate",
        "malaria_lr.png",
    )
    plot_confusion_matrix(
        test_metrics["cm"],
        CLASS_NAMES,
        "malaria_confusion_matrix.png",
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()