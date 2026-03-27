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
NPZ_PATH = "colorectal_histology.npz"
NUM_CLASSES = 8
CLASS_NAMES = [
    "tumor",
    "stroma",
    "complex",
    "lympho",
    "debris",
    "mucosa",
    "adipose",
    "empty",
]

RUN_MODE = "full"

NUM_EPOCHS = 30
BATCH_SIZE = 16
ACCUM_STEPS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224
MODEL_NAME = "vit_base_patch16_224"
# MODEL_NAME = "vit_small_patch16_224"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

VAL_SIZE = 0.1
TEST_SIZE = 0.1
USE_SCHEDULER = True
LABEL_SMOOTHING = 0.1
PATIENCE = 8
PRINT_EVERY = 20

PLOT_DIR = "colorectal_exp_full"
RESULT_JSON = os.path.join(PLOT_DIR, "results_colorectal_exp_full.json")
BEST_CKPT_PATH = os.path.join(PLOT_DIR, "checkpoint_colorectal_vit_full_best.pt")
LATEST_CKPT_PATH = os.path.join(PLOT_DIR, "checkpoint_colorectal_vit_full_latest.pt")


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


# ===========================
# Dataset
# ===========================
class ColorectalDataset(Dataset):
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
    path = os.path.join(PLOT_DIR, out_name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close()


# ===========================
# Split helper
# ===========================
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

    return x_train, x_val, x_test, y_train, y_val, y_test


# ===========================
# Checkpoint helpers
# ===========================
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_acc: float,
    best_epoch: int,
    patience_counter: int,
    history: dict,
    val_metrics: dict,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "history": history,
        "last_val_loss": float(val_metrics["loss"]),
        "last_val_acc": float(val_metrics["acc"]),
        "last_val_f1_macro": float(val_metrics["f1_macro"]),
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "run_mode": RUN_MODE,
        "seed": SEED,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "use_scheduler": USE_SCHEDULER,
    }
    torch.save(ckpt, path)


def empty_history():
    return {
        "loss": [],
        "acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "lr": [],
    }


def load_checkpoint_if_exists(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
):
    if not os.path.exists(path):
        print(f"No checkpoint found at: {path}")
        return 1, -1.0, -1, 0, empty_history()

    print(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt.get("best_val_acc", ckpt.get("last_val_acc", -1.0))
    best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", -1))
    patience_counter = ckpt.get("patience_counter", 0)
    history = ckpt.get("history", empty_history())

    print(
        f"Resume info -> next epoch: {start_epoch}, "
        f"best_val_acc: {best_val_acc:.4f}, best_epoch: {best_epoch}"
    )

    return start_epoch, best_val_acc, best_epoch, patience_counter, history


# ===========================
# Main
# ===========================
def main():
    set_seed(SEED)
    os.makedirs(PLOT_DIR, exist_ok=True)

    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"{NPZ_PATH} not found. Run export script first.")

    data = np.load(NPZ_PATH)
    x_all = data["x"].astype(np.uint8)
    y_all = data["y"].astype(np.int64)

    print("Loaded:", NPZ_PATH, "x:", x_all.shape, "y:", y_all.shape)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    x_train, x_val, x_test, y_train, y_val, y_test = prepare_splits(x_all, y_all)

    print(f"Run mode: {RUN_MODE}")
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    train_ds = ColorectalDataset(x_train, y_train, train=True, out_size=IMAGE_SIZE)
    val_ds = ColorectalDataset(x_val, y_val, train=False, out_size=IMAGE_SIZE)
    test_ds = ColorectalDataset(x_test, y_test, train=False, out_size=IMAGE_SIZE)

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

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            with autocast_ctx(enabled=(DEVICE.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            preds = logits.argmax(dim=1)
            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

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
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "cm": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "classification_report": cls_report,
        }

    start_epoch = 1
    best_val_acc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = empty_history()

    start_epoch, best_val_acc, best_epoch, patience_counter, history = load_checkpoint_if_exists(
        LATEST_CKPT_PATH,
        model,
        optimizer,
        scheduler,
        scaler,
        DEVICE,
    )

    if start_epoch > NUM_EPOCHS:
        print(f"Training already completed up to epoch {start_epoch - 1}. Skipping training loop.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
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
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        improved = val_metrics["acc"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            LATEST_CKPT_PATH,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_acc,
            best_epoch,
            patience_counter,
            history,
            val_metrics,
        )

        if improved:
            save_checkpoint(
                BEST_CKPT_PATH,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_acc,
                best_epoch,
                patience_counter,
                history,
                val_metrics,
            )

        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} lr={history['lr'][-1]:.6f} "
            f"time={time.time() - t0:.1f}s"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print("Saved best checkpoint:", BEST_CKPT_PATH)
    print("Saved latest checkpoint:", LATEST_CKPT_PATH)

    if os.path.exists(BEST_CKPT_PATH):
        checkpoint = torch.load(BEST_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    elif os.path.exists(LATEST_CKPT_PATH):
        print("Best checkpoint not found, using latest checkpoint for test.")
        checkpoint = torch.load(LATEST_CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError("No checkpoint found for final evaluation.")

    test_metrics = evaluate(test_loader)

    print("\nFinal test results:")
    print(f"test_loss            : {test_metrics['loss']:.4f}")
    print(f"test_accuracy        : {test_metrics['acc']:.4f}")
    print(f"test_precision_macro : {test_metrics['precision_macro']:.4f}")
    print(f"test_recall_macro    : {test_metrics['recall_macro']:.4f}")
    print(f"test_f1_macro        : {test_metrics['f1_macro']:.4f}")
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
        "dataset": "colorectal_histology",
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
        "test_precision_macro": float(test_metrics["precision_macro"]),
        "test_recall_macro": float(test_metrics["recall_macro"]),
        "test_f1_macro": float(test_metrics["f1_macro"]),
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
        "Colorectal Train vs Val Loss",
        "colorectal_loss.png",
    )
    plot_two_curves(
        history["acc"],
        history["val_acc"],
        "acc",
        "Colorectal Train vs Val Accuracy",
        "colorectal_accuracy.png",
    )
    plot_one_curve(
        history["val_f1_macro"],
        "val_f1_macro",
        "Colorectal Validation Macro-F1",
        "colorectal_val_f1_macro.png",
    )
    plot_one_curve(
        history["lr"],
        "lr",
        "Colorectal Learning Rate",
        "colorectal_lr.png",
    )
    plot_confusion_matrix(
        test_metrics["cm"],
        CLASS_NAMES,
        "colorectal_confusion_matrix.png",
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()