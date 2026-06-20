from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import io
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import timm

app = Flask(__name__, template_folder="templates", static_folder="static")

# Class names used by the project
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

IMAGE_SIZE = 224
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "colorectal_exp_auto")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
VISUALIZE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# Simple model cache
_MODEL_CACHE = {}


def _pil_to_vit_tensor(img: Image.Image, out_size: int = IMAGE_SIZE) -> torch.Tensor:
    img = img.convert("RGB")
    img = img.resize((out_size, out_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / np.float32(255.0)
    # CHW
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std
    t = torch.from_numpy(arr).unsqueeze(0)
    return t


def get_execution_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.cuda.current_device()
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def load_vit_from_checkpoint(task: str) -> Tuple[torch.nn.Module, list]:
    key = ("vit", task)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    task_dir = os.path.join(BASE_OUTPUT_DIR, task)
    ckpt_path = os.path.join(task_dir, "checkpoint_raytuned_vit_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ViT checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    class_names = ckpt.get("class_names", MULTICLASS_CLASS_NAMES if task == "multiclass" else BINARY_CLASS_NAMES)
    num_classes = len(class_names)
    model_name = config.get("model_name", "vit_base_patch16_224")

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, img_size=IMAGE_SIZE)
    state = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state)
    except Exception:
        # Try to handle if checkpoint contains nested dict
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

    model.eval()
    model.to(device)
    _MODEL_CACHE[key] = (model, class_names)
    return model, class_names


def load_autokeras_model(task: str):
    key = ("ak", task)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow not available for AutoKeras model loading") from e

    task_dir = os.path.join(BASE_OUTPUT_DIR, task)
    model_path = os.path.join(task_dir, "autokeras_colorectal_best.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AutoKeras model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    class_names = MULTICLASS_CLASS_NAMES if task == "multiclass" else BINARY_CLASS_NAMES
    _MODEL_CACHE[key] = (model, class_names)
    return model, class_names


def list_sample_images(limit: int = 2):
    if not os.path.isdir(SAMPLE_DIR):
        return []
    images = sorted([
        fname for fname in os.listdir(SAMPLE_DIR)
        if os.path.isfile(os.path.join(SAMPLE_DIR, fname)) and fname.split('.')[-1].lower() in VISUALIZE_EXTENSIONS
    ])
    return images[:limit]


@app.route("/", methods=["GET"])
def index():
    sample_images = list_sample_images(limit=2)
    return render_template("index.html", sample_images=sample_images)


def list_visualization_images():
    tasks = ["multiclass", "binary"]
    images = {}
    for task in tasks:
        task_dir = os.path.join(BASE_OUTPUT_DIR, task)
        if not os.path.isdir(task_dir):
            images[task] = []
            continue
        images[task] = sorted([
            fname for fname in os.listdir(task_dir)
            if os.path.isfile(os.path.join(task_dir, fname)) and fname.split('.')[-1].lower() in VISUALIZE_EXTENSIONS
        ])
    return images


@app.route("/visualize", methods=["GET"])
def visualize():
    images = list_visualization_images()
    return render_template("visualize.html", tasks=list(images.keys()), images=images)


@app.route("/visualize_image/<task>/<path:filename>", methods=["GET"])
def visualize_image(task, filename):
    task_dir = os.path.join(BASE_OUTPUT_DIR, task)
    if not os.path.isdir(task_dir):
        return jsonify({"error": "task not found"}), 404
    safe_name = os.path.basename(filename)
    return send_from_directory(task_dir, safe_name)


@app.route("/sample_image/<path:filename>", methods=["GET"])
def sample_image(filename):
    safe_name = os.path.basename(filename)
    return send_from_directory(SAMPLE_DIR, safe_name)


@app.route("/predict", methods=["POST"])
def predict():
    # params: model_type: vit|autokeras, task: multiclass|binary, lang: cn|en
    try:
        model_type = request.form.get("model_type", "vit")
        task = request.form.get("task", "multiclass")
        lang = request.form.get("lang", "cn")

        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        f = request.files["image"]
        img = Image.open(io.BytesIO(f.read()))

        if model_type == "vit":
            try:
                model, class_names = load_vit_from_checkpoint(task)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

            device = get_execution_device()
            xb = _pil_to_vit_tensor(img).to(device)
            with torch.no_grad():
                try:
                    logits = model(xb)
                except Exception as e:
                    if device.type == "cuda":
                        device = torch.device("cpu")
                        model.to(device)
                        xb = xb.to(device)
                        logits = model(xb)
                    else:
                        raise
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            score = float(probs[pred_idx])

        else:  # autokeras
            try:
                model, class_names = load_autokeras_model(task)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

            img_rgb = img.convert("RGB")
            img_resized = img_rgb.resize((128, 128), Image.BILINEAR)
            arr = np.asarray(img_resized).astype(np.float32) / np.float32(255.0)
            xb = np.expand_dims(arr, axis=0)
            raw = model.predict(xb)
            if raw.ndim > 1 and raw.shape[-1] == len(class_names):
                probs = raw[0]
                pred_idx = int(np.argmax(probs))
                pred_name = class_names[pred_idx]
                score = float(probs[pred_idx])
            else:
                # some AutoKeras exports return class indices
                pred_idx = int(raw.reshape(-1)[0])
                probs = None
                pred_name = class_names[pred_idx]
                score = 1.0

        if lang == "cn":
            text = f"此图片在 {task} 分类下的分类是：{pred_name}（置信度 {score:.2f}）"
        else:
            text = f"This image is classified as {pred_name} for {task} (confidence {score:.2f})"

        return jsonify({"pred_idx": pred_idx, "pred_name": pred_name, "score": score, "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
