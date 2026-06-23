# Colorectal ViT Lab WebApp

Run the Flask application from the `FP` directory:

```bash
pip install flask pillow
python webapp/app.py
```

Open `http://127.0.0.1:5000`.

## Prediction

For ViT prediction, select the task and the exact architecture used during training:

- input size: `128`, `224`, or `384`
- patch size: `16` or `32`
- ViT scale: `tiny`, `small`, or `base`

The WebApp lists only completed checkpoints. A ViT prediction is enabled only when a checkpoint exists for the selected task/architecture. The checkpoint's saved `model_name` and `image_size` are used to construct the model, so the WebApp does not silently use a `224px` model for another experiment.

AutoKeras remains a separate baseline and uses its exported model from the task root.

## Training

The training panel starts the existing `colorectal_autokeras_raytune.py` auto-training pipeline with the selected task, architecture, Ray Tune trial count, tuning epochs, and final-training epochs. It runs in the same Python environment as the WebApp and writes a log to:

```text
output/colorectal_exp_auto/web_training_logs/
```

Only one WebApp-managed training job can run at a time. Training never starts until the **Start training run** button is pressed.

## Results

The gallery at `/visualize` recursively displays final evaluation artifacts, including configuration-specific `attention_maps/` files. It filters by task and saved experiment directory, for example:

```text
output/colorectal_exp_auto/multiclass/image224_patch16_base/
```
