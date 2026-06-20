# Webapp for Colorectal Classifier

Run a small Flask app to upload an image and get predictions from trained models.

Requirements (additionally to project requirements):

```
pip install flask pillow
```

Run the app from the `FP` folder:

```bash
python webapp/app.py
```

Open http://localhost:5000 in your browser. Select language (CN/EN), task (multiclass/binary), model (ViT or AutoKeras), upload an image and click Predict.

Notes:
- The app expects model files under `output/colorectal_exp_auto/<task>/`:
  - ViT checkpoint: `checkpoint_raytuned_vit_best.pt`
  - AutoKeras export: `autokeras_colorectal_best.keras`
- If models are stored elsewhere, edit `webapp/app.py` to change `BASE_OUTPUT_DIR`.
