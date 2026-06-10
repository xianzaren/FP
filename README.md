# Colorectal Histology Classification: AutoKeras and Ray Tune

## 1. Project overview

This project trains and evaluates image classification models on colorectal histology image patches.

The code supports three main modes:

- `ray_tune`: searches for better ViT hyperparameters using Ray Tune.
- `final_train`: uses the best Ray Tune configuration to train the final ViT model.
- `autokeras`: trains an AutoKeras image-classification baseline for comparison.

The code supports two tasks:

- `multiclass`: original 8-class tissue classification.
- `binary`: patch-level `tumor` vs `non_tumor` classification.

The binary task is patch-level tumor detection. It is not patient-level disease diagnosis.

---

## 2. Dataset preparation

The dataset is prepared directly with Python, rather than manually downloading it from Kaggle or another website.

This code expects the processed dataset to be saved as:

```text
data/colorectal_histology.npz
```

The `.npz` file should contain two arrays:

```text
x: image data, shape similar to (N, H, W, 3)
y: integer labels, shape similar to (N,)
```

Use the following Python code to download the colorectal histology dataset through TensorFlow Datasets and save it as the `.npz` file required by this project:

```bash
python - <<'PY'
import os
import numpy as np
import tensorflow_datasets as tfds

os.makedirs("data", exist_ok=True)

ds = tfds.load("colorectal_histology", split="train", as_supervised=True)

x_list = []
y_list = []

for image, label in tfds.as_numpy(ds):
    x_list.append(image)
    y_list.append(label)

x = np.stack(x_list).astype("uint8")
y = np.array(y_list).astype("int64")

np.savez("data/colorectal_histology.npz", x=x, y=y)

print("Saved: data/colorectal_histology.npz")
print("x shape:", x.shape)
print("y shape:", y.shape)
print("labels:", np.unique(y, return_counts=True))
PY
```

After running this command, the code should be able to load the dataset from:

```text
data/colorectal_histology.npz
```

---

## 3. Multiclass running commands

Run Ray Tune first to search for the best ViT hyperparameters:

```bash
python colorectal_autokeras_raytune.py --mode ray_tune --task multiclass --num-samples 2 --tune-epochs 3 --npz-path data/colorectal_histology.npz
```

Then run final training using the best Ray Tune configuration:

```bash
python colorectal_autokeras_raytune.py --mode final_train --task multiclass --final-epochs 30 --npz-path data/colorectal_histology.npz
```

The multiclass outputs will be saved in:

```text
output/colorectal_exp_auto/multiclass/
```

---

## 4. Binary running commands

The binary task maps:

```text
tumor -> positive class
all other classes -> non_tumor
```

Run Ray Tune for binary classification:

```bash
python colorectal_autokeras_raytune.py --mode ray_tune --task binary --use-class-weights --num-samples 2 --tune-epochs 3 --npz-path data/colorectal_histology.npz
```

Then run final binary training:

```bash
python colorectal_autokeras_raytune.py --mode final_train --task binary --use-class-weights --final-epochs 30 --npz-path data/colorectal_histology.npz
```

The binary outputs will be saved in:

```text
output/colorectal_exp_auto/binary/
```

---

## 5. AutoKeras baseline running commands

Run AutoKeras baseline for multiclass classification:

```bash
python colorectal_autokeras_raytune.py --mode autokeras --task multiclass --ak-max-trials 3 --ak-epochs 8 --npz-path data/colorectal_histology.npz
```

Run AutoKeras baseline for binary classification:

```bash
python colorectal_autokeras_raytune.py --mode autokeras --task binary --use-class-weights --ak-max-trials 3 --ak-epochs 8 --npz-path data/colorectal_histology.npz
```

AutoKeras is used as a baseline model for comparison with the Ray Tune-optimised ViT.

---

## 6. Output files

For the Ray Tune and final ViT model, the main output files include:

```text
best_ray_config.json
best_ray_result_summary.json
results_raytuned_vit_final.json
checkpoint_raytuned_vit_best.pt
raytuned_loss.png
raytuned_accuracy.png
raytuned_val_f1_macro.png
raytuned_lr.png
raytuned_confusion_matrix.png
raytuned_roc_curve.png
raytuned_pr_curve.png
raytuned_misclassified_patches.png
raytuned_attention_map.png
```

For AutoKeras, the main output files include:

```text
results_autokeras_baseline.json
autokeras_colorectal_best.keras
autokeras_confusion_matrix.png
autokeras_roc_curve.png
autokeras_pr_curve.png
autokeras_misclassified_patches.png
```

Multiclass results are saved under:

```text
output/colorectal_exp_auto/multiclass/
```

Binary results are saved under:

```text
output/colorectal_exp_auto/binary/
```
