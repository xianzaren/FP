# Colorectal Histology Classification: ViT, Ray Tune, and AutoKeras

This project trains and evaluates image classifiers for colorectal histology patch images. It supports the original 8-class tissue classification task and a patch-level binary tumor vs non-tumor task.

The main ViT workflow uses Ray Tune to search hyperparameters, then trains a final PyTorch ViT model with the selected configuration. AutoKeras is kept as a baseline route.

## Main Entry Points

- `download_colorectal.py`: downloads the TensorFlow Datasets colorectal histology dataset and writes `data/colorectal_histology.npz`.
- `colorectal_autokeras_raytune.py`: single-experiment worker for `auto_train`, `ray_tune`, `final_train`, and `autokeras` modes.
- `run_batches_experiments.py`: interactive batch launcher for multiple task/image-size/patch-size/ViT-scale combinations.
- `compare_experiment_outputs.py`: summarizes completed ViT experiments into task-specific Excel workbooks and a Markdown report.
- `webapp/app.py`: Flask app for prediction, training launch, and result visualization.

## Environment

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

PyTorch, Ray Tune, TensorFlow, AutoKeras, and OpenPyXL are used by different parts of the project. GPU is used automatically by the ViT training code when CUDA is available.

## Dataset Preparation

The training code expects:

```text
data/colorectal_histology.npz
```

The `.npz` file must contain:

```text
x: image data, shape similar to (N, H, W, 3)
y: integer labels, shape similar to (N,)
```

Prepare it with:

```bash
python download_colorectal.py
```

## Tasks

- `multiclass`: original 8-class tissue classification.
- `binary`: patch-level `tumor` vs `non_tumor`; this is not patient-level diagnosis.

For binary training, use `--use-class-weights` unless you have a specific reason not to, because tumor is the minority class.

## Single ViT Experiment

The recommended single-run mode is `auto_train`, which runs Ray Tune and then final training for one selected architecture.

Example multiclass run:

```bash
python colorectal_autokeras_raytune.py --mode auto_train --task multiclass --image-size 224 --patch-size 16 --vit-depth small --num-samples 4 --tune-epochs 6 --final-epochs 30
```

Example binary run:

```bash
python colorectal_autokeras_raytune.py --mode auto_train --task binary --use-class-weights --image-size 224 --patch-size 16 --vit-depth small --num-samples 4 --tune-epochs 6 --final-epochs 30
```

You can also run the two stages manually:

```bash
python colorectal_autokeras_raytune.py --mode ray_tune --task multiclass --image-size 224 --patch-size 16 --vit-depth small --num-samples 4 --tune-epochs 6
python colorectal_autokeras_raytune.py --mode final_train --task multiclass --image-size 224 --patch-size 16 --vit-depth small --final-epochs 30
```

## Batch Experiments

Use the batch launcher when comparing many architecture combinations:

```bash
python run_batches_experiments.py
```

It prompts for task, image size, patch size, and ViT scale. Press Enter in a prompt to include all choices for that category.

Current batch options are:

```text
tasks: binary, multiclass
image sizes: 128, 160, 224, 384
patch sizes: 16, 32
ViT scales: tiny, small
```

The launcher skips invalid combinations where `image_size` is not divisible by `patch_size`. It also skips `patch32 + tiny`, matching the current worker-script constraints.

Batch logs are written under:

```text
output/batch_runs/
```

The actual model results are still written by the worker script under `output/colorectal_exp_auto/...`.

## Output Layout

ViT outputs are organized by task and architecture:

```text
output/colorectal_exp_auto/<task>/image<image_size>_patch<patch_size>_<vit_depth>/
```

Example:

```text
output/colorectal_exp_auto/binary/image224_patch16_small/
```

Important files in a completed ViT experiment include:

```text
best_ray_config.json
best_ray_result_summary.json
results_raytuned_vit_final.json
checkpoint_raytuned_vit_best.pt
pipeline_selection.json
raytuned_loss.png
raytuned_accuracy.png
raytuned_val_f1_macro.png
raytuned_confusion_matrix.png
raytuned_roc_curve.png
raytuned_pr_curve.png
failure_analysis/
attention_maps/
```

Some plots or metrics depend on the task. For example, binary outputs include tumor-focused PR/ROC metrics, while multiclass outputs focus on multiclass classification metrics.

## Comparing Experiments

After experiments finish, generate comparison workbooks with:

```bash
python compare_experiment_outputs.py
```

Default outputs are:

```text
output/experiment_comparison/binary_comparison.xlsx
output/experiment_comparison/multiclass_comparison.xlsx
output/experiment_comparison/comparison_report.md
```

Each workbook contains:

```text
notes
summary
patch_size
token_trends
vit_depth
```

The comparison sheets use long format: one row is one experiment. Rows with the same `comparison_id` belong to the same comparison group.

Interpretation notes:

- `summary`: all completed experiments for one task, sorted by the selected metric.
- `patch_size`: fixes image size and ViT scale; patch size and token count change together.
- `token_trends`: fixes patch size and ViT scale; image size and token count change together.
- `vit_depth`: fixes image size, patch size, and token count; ViT scale changes.

Important caveat: patch size and token count are mathematically linked:

```text
num_tokens = (image_size / patch_size)^2
```

Therefore, the current patch-size table is not a pure patch-size ablation. It compares combinations such as `patch16 + more tokens` vs `patch32 + fewer tokens` at the same image size and ViT scale. Also, Ray Tune searches hyperparameters separately per architecture, so these are best-result comparisons rather than strict one-variable ablations.

If an Excel file is open in another program on Windows, the script may fail to overwrite it. Close the workbook and rerun the script.

## AutoKeras Baseline

Run AutoKeras baseline training with:

```bash
python colorectal_autokeras_raytune.py --mode autokeras --task multiclass --ak-max-trials 3 --ak-epochs 8
python colorectal_autokeras_raytune.py --mode autokeras --task binary --use-class-weights --ak-max-trials 3 --ak-epochs 8
```

AutoKeras outputs are saved under the task root in `output/colorectal_exp_auto/<task>/`.

## WebApp

Run the Flask app from the project root:

```bash
python webapp/app.py
```

Open:

```text
http://127.0.0.1:5000
```

The WebApp can:

- list completed ViT checkpoints by task and architecture,
- predict uploaded patch images with a saved ViT checkpoint or AutoKeras baseline,
- start one training job at a time,
- show result galleries and visualization artifacts at `/visualize`.

WebApp-managed logs are written to:

```text
output/colorectal_exp_auto/web_training_logs/
```

## Practical Reporting Guidance

When writing conclusions from the current experiment grid, use careful wording:

- Say `patch16 with more tokens often performs better than patch32 with fewer tokens`, not `patch size alone caused the gain`.
- Say `token count is not monotonically better`, because higher token counts do not always produce better F1.
- Say `tiny vs small has task- and image-size-dependent behavior`, because `small` does not consistently dominate `tiny` in the current results.

For strict ablation claims, rerun experiments with fixed hyperparameters and only one changed variable, ideally with repeated seeds and mean/std reporting.
