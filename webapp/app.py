from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from PIL import Image
import torch
import timm

app = Flask(__name__, template_folder="templates", static_folder="static")

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
BINARY_CLASS_NAMES = ["non_tumor", "tumor"]
TASKS = {"multiclass", "binary"}
IMAGE_SIZES = (128, 224, 384)
PATCH_SIZES = (16, 32)
VIT_SCALES = ("tiny", "small", "base")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
EXPERIMENT_PATTERN = re.compile(r"^image(?P<image_size>\d+)_patch(?P<patch_size>\d+)_(?P<vit_depth>tiny|small|base)$")
MODEL_PATTERN = re.compile(r"^vit_(?P<vit_depth>tiny|small|base)_patch(?P<patch_size>16|32)_")

WEBAPP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEBAPP_DIR.parent
TRAIN_SCRIPT = PROJECT_DIR / "colorectal_autokeras_raytune.py"
BASE_OUTPUT_DIR = PROJECT_DIR / "output" / "colorectal_exp_auto"
SAMPLE_DIR = PROJECT_DIR / "data" / "samples"
LOG_DIR = BASE_OUTPUT_DIR / "web_training_logs"

_MODEL_CACHE: Dict[Tuple[str, str, float], Tuple[torch.nn.Module, List[str], int]] = {}
_TRAINING_LOCK = threading.Lock()
_TRAINING_JOB: Dict[str, Any] | None = None


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _model_name(vit_depth: str, patch_size: int) -> str:
    return f"vit_{vit_depth}_patch{patch_size}_224"


def _task_class_names(task: str) -> List[str]:
    return MULTICLASS_CLASS_NAMES if task == "multiclass" else BINARY_CLASS_NAMES


def _validate_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError("task must be 'multiclass' or 'binary'.")
    return task


def _task_dir(task: str) -> Path:
    return BASE_OUTPUT_DIR / _validate_task(task)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _metadata_for_experiment(task: str, experiment_id: str, directory: Path) -> Dict[str, Any]:
    pipeline = _safe_json(directory / "pipeline_selection.json")
    architecture = pipeline.get("architecture", {}) if isinstance(pipeline.get("architecture"), dict) else {}
    best_config = _safe_json(directory / "best_ray_config.json")
    config = {**best_config, **architecture}
    match = EXPERIMENT_PATTERN.match(experiment_id)

    image_size = config.get("image_size")
    patch_size = config.get("patch_size")
    vit_depth = config.get("vit_depth")
    if match:
        image_size = image_size or int(match.group("image_size"))
        patch_size = patch_size or int(match.group("patch_size"))
        vit_depth = vit_depth or match.group("vit_depth")

    model_name = config.get("model_name")
    model_match = MODEL_PATTERN.match(str(model_name)) if model_name else None
    if model_match:
        patch_size = patch_size or int(model_match.group("patch_size"))
        vit_depth = vit_depth or model_match.group("vit_depth")
    if not model_name and vit_depth in VIT_SCALES and patch_size in PATCH_SIZES:
        model_name = _model_name(vit_depth, int(patch_size))

    image_size = int(image_size) if image_size else 224
    patch_size = int(patch_size) if patch_size else 16
    vit_depth = str(vit_depth) if vit_depth else "base"
    label = (
        f"{image_size}px / patch {patch_size} / {vit_depth}"
        if experiment_id != "legacy"
        else f"Legacy checkpoint ({model_name or 'ViT'})"
    )
    return {
        "id": experiment_id,
        "label": label,
        "image_size": image_size,
        "patch_size": patch_size,
        "vit_depth": vit_depth,
        "model_name": model_name or _model_name(vit_depth, patch_size),
        "checkpoint": str(directory / "checkpoint_raytuned_vit_best.pt"),
        "task": task,
    }


def list_vit_experiments(task: str) -> List[Dict[str, Any]]:
    task_dir = _task_dir(task)
    if not task_dir.is_dir():
        return []

    experiments: List[Dict[str, Any]] = []
    legacy_checkpoint = task_dir / "checkpoint_raytuned_vit_best.pt"
    if legacy_checkpoint.is_file():
        experiments.append(_metadata_for_experiment(task, "legacy", task_dir))

    for child in sorted(task_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or not (child / "checkpoint_raytuned_vit_best.pt").is_file():
            continue
        experiments.append(_metadata_for_experiment(task, child.name, child))
    return experiments


def _find_experiment(task: str, experiment_id: str) -> Dict[str, Any]:
    for experiment in list_vit_experiments(task):
        if experiment["id"] == experiment_id:
            return experiment
    raise FileNotFoundError(
        "No saved ViT checkpoint matches this task and architecture. "
        "Train it first or choose an available experiment."
    )


def _pil_to_vit_tensor(image: Image.Image, out_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((out_size, out_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    array = array.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy((array - mean) / std).unsqueeze(0)


def get_execution_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.cuda.current_device()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def load_vit_from_checkpoint(task: str, experiment_id: str) -> Tuple[torch.nn.Module, List[str], int]:
    experiment = _find_experiment(task, experiment_id)
    checkpoint_path = Path(experiment["checkpoint"])
    cache_key = (task, experiment_id, checkpoint_path.stat().st_mtime)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    device = get_execution_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    class_names = checkpoint.get("class_names", _task_class_names(task)) if isinstance(checkpoint, dict) else _task_class_names(task)
    image_size = int(config.get("image_size", experiment["image_size"]))
    model_name = config.get("model_name", experiment["model_name"])
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names),
        img_size=image_size,
        drop_rate=config.get("drop_rate", 0.0),
        drop_path_rate=config.get("drop_path_rate", 0.0),
    )
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval().to(device)
    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = (model, class_names, image_size)
    return model, class_names, image_size


def load_autokeras_model(task: str):
    cache_key = ("autokeras", task, 0.0)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], cached[1]

    try:
        import tensorflow as tf
    except Exception as error:
        raise RuntimeError("TensorFlow is not available for AutoKeras prediction.") from error

    model_path = _task_dir(task) / "autokeras_colorectal_best.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"AutoKeras model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    class_names = _task_class_names(task)
    _MODEL_CACHE[cache_key] = (model, class_names, 128)
    return model, class_names


def list_sample_images(limit: int = 4) -> List[str]:
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(
        path.name for path in SAMPLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[:limit]


def list_visualization_images() -> Dict[str, List[Dict[str, str]]]:
    images: Dict[str, List[Dict[str, str]]] = {}
    for task in sorted(TASKS):
        task_dir = _task_dir(task)
        if not task_dir.is_dir():
            images[task] = []
            continue

        task_images: List[Dict[str, str]] = []
        for image_path in sorted(task_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(task_dir)
            if "ray_tune_trials" in relative.parts:
                continue
            experiment_id = relative.parts[0] if len(relative.parts) > 1 else "legacy"
            task_images.append({
                "filename": relative.as_posix(),
                "caption": relative.name,
                "experiment_id": experiment_id,
            })
        images[task] = task_images
    return images


def _read_log_tail(log_path: Path, limit: int = 5000) -> str:
    if not log_path.is_file():
        return "Waiting for training output..."
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return text[-limit:]
    except OSError:
        return "Training log is temporarily unavailable."


def _training_status() -> Dict[str, Any]:
    global _TRAINING_JOB
    with _TRAINING_LOCK:
        if _TRAINING_JOB is None:
            return {"state": "idle", "log_tail": ""}

        process = _TRAINING_JOB["process"]
        return_code = process.poll()
        if return_code is None:
            state = "running"
        elif return_code == 0:
            state = "completed"
        else:
            state = "failed"

        log_handle = _TRAINING_JOB.get("log_handle")
        if return_code is not None and log_handle is not None and not log_handle.closed:
            log_handle.close()
        return {
            "state": state,
            "return_code": return_code,
            "started_at": _TRAINING_JOB["started_at"],
            "command": _TRAINING_JOB["command"],
            "log_tail": _read_log_tail(_TRAINING_JOB["log_path"]),
        }


def _parse_training_request(data: Dict[str, Any]) -> Dict[str, Any]:
    task = _validate_task(str(data.get("task", "multiclass")))
    image_size = int(data.get("image_size", 224))
    patch_size = int(data.get("patch_size", 16))
    vit_depth = str(data.get("vit_depth", "base"))
    num_samples = int(data.get("num_samples", 4))
    tune_epochs = int(data.get("tune_epochs", 6))
    final_epochs = int(data.get("final_epochs", 30))

    if image_size not in IMAGE_SIZES:
        raise ValueError("image_size must be one of 128, 224, or 384.")
    if patch_size not in PATCH_SIZES:
        raise ValueError("patch_size must be 16 or 32.")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")
    if vit_depth not in VIT_SCALES:
        raise ValueError("vit_depth must be tiny, small, or base.")
    if not 1 <= num_samples <= 20:
        raise ValueError("num_samples must be between 1 and 20.")
    if not 1 <= tune_epochs <= 50:
        raise ValueError("tune_epochs must be between 1 and 50.")
    if not 1 <= final_epochs <= 200:
        raise ValueError("final_epochs must be between 1 and 200.")

    model_name = _model_name(vit_depth, patch_size)
    if not timm.is_model(model_name):
        raise ValueError(
            f"Installed timm does not provide {model_name}. Update timm or choose another scale/patch pair."
        )
    return {
        "task": task,
        "image_size": image_size,
        "patch_size": patch_size,
        "vit_depth": vit_depth,
        "num_samples": num_samples,
        "tune_epochs": tune_epochs,
        "final_epochs": final_epochs,
    }


@app.get("/")
def index():
    return render_template("index.html", sample_images=list_sample_images())


@app.get("/api/experiments/<task>")
def experiments(task: str):
    try:
        return jsonify({"experiments": list_vit_experiments(_validate_task(task))})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/training/start")
def start_training():
    global _TRAINING_JOB
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        settings = _parse_training_request(data)
        if not TRAIN_SCRIPT.is_file():
            raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

        with _TRAINING_LOCK:
            if _TRAINING_JOB is not None and _TRAINING_JOB["process"].poll() is None:
                return jsonify({"error": "A training job is already running."}), 409

            LOG_DIR.mkdir(parents=True, exist_ok=True)
            job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = LOG_DIR / f"{job_id}_{settings['task']}_image{settings['image_size']}_patch{settings['patch_size']}_{settings['vit_depth']}.log"
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--mode", "auto_train",
                "--task", settings["task"],
                "--image-size", str(settings["image_size"]),
                "--patch-size", str(settings["patch_size"]),
                "--vit-depth", settings["vit_depth"],
                "--num-samples", str(settings["num_samples"]),
                "--tune-epochs", str(settings["tune_epochs"]),
                "--final-epochs", str(settings["final_epochs"]),
            ]
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_DIR,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _TRAINING_JOB = {
                "process": process,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "log_path": log_path,
                "log_handle": log_handle,
            }
        return jsonify({"message": "Training started.", "status": _training_status()}), 202
    except (TypeError, ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/training/status")
def training_status():
    return jsonify(_training_status())


@app.get("/visualize")
def visualize():
    return render_template(
        "visualize.html",
        images=list_visualization_images(),
        experiments={task: list_vit_experiments(task) for task in sorted(TASKS)},
    )


@app.get("/visualize_image/<task>/<path:filename>")
def visualize_image(task: str, filename: str):
    try:
        task_dir = _task_dir(task)
    except ValueError:
        abort(404)
    candidate = (task_dir / filename).resolve()
    if not _is_relative_to(candidate, task_dir) or not candidate.is_file():
        abort(404)
    return send_from_directory(task_dir, filename)


@app.get("/sample_image/<path:filename>")
def sample_image(filename: str):
    candidate = (SAMPLE_DIR / filename).resolve()
    if not _is_relative_to(candidate, SAMPLE_DIR) or not candidate.is_file():
        abort(404)
    return send_from_directory(SAMPLE_DIR, filename)


@app.post("/predict")
def predict():
    try:
        model_type = request.form.get("model_type", "vit")
        task = _validate_task(request.form.get("task", "multiclass"))
        if model_type not in {"vit", "autokeras"}:
            raise ValueError("model_type must be 'vit' or 'autokeras'.")
        if "image" not in request.files:
            return jsonify({"error": "Choose an image before predicting."}), 400

        image = Image.open(io.BytesIO(request.files["image"].read()))
        if model_type == "vit":
            experiment_id = request.form.get("experiment_id", "")
            model, class_names, image_size = load_vit_from_checkpoint(task, experiment_id)
            device = get_execution_device()
            batch = _pil_to_vit_tensor(image, image_size).to(device)
            with torch.no_grad():
                probabilities = torch.softmax(model(batch), dim=1).cpu().numpy()[0]
            model_label = next(item["label"] for item in list_vit_experiments(task) if item["id"] == experiment_id)
        else:
            model, class_names = load_autokeras_model(task)
            resized = image.convert("RGB").resize((128, 128), Image.BILINEAR)
            batch = np.expand_dims(np.asarray(resized, dtype=np.float32) / np.float32(255.0), axis=0)
            raw = np.asarray(model.predict(batch, verbose=0))
            if raw.ndim > 1 and raw.shape[-1] == len(class_names):
                probabilities = raw[0]
            else:
                probabilities = np.zeros(len(class_names), dtype=np.float32)
                probabilities[int(raw.reshape(-1)[0])] = 1.0
            model_label = "AutoKeras (128px input)"

        prediction_index = int(np.argmax(probabilities))
        confidence = float(probabilities[prediction_index])
        return jsonify({
            "pred_idx": prediction_index,
            "pred_name": class_names[prediction_index],
            "score": confidence,
            "model_label": model_label,
            "text": f"Prediction: {class_names[prediction_index]} ({confidence:.1%})",
        })
    except (FileNotFoundError, ValueError, OSError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.getenv("FLASK_DEBUG") == "1")
