"""
Unified ViT (sample-code style) for TFDS datasets (malaria / blood_cells)

Fully mirrors original Keras CIFAR example logic:
1) Build x_train/y_train and x_test/y_test first (like load_data() provides)
2) model.fit(..., validation_split=0.1) to create a validation set from x_train
3) model.evaluate(x_test, y_test) for final test evaluation

Fix: Some TFDS image datasets may have variable H/W -> resize BEFORE np.stack.

Also prints:
- DATASET_NAME, TRAIN_SPLIT, TEST_SPLIT
- num_epochs, batch_size, image_size, patch_size
- num_patches, projection_dim, num_heads, transformer_layers, mlp_head_units
- train_pool size, val size, actual train used, test size
"""

# ---------------------------
# Setup
# ---------------------------
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Dataset (change here)
# ---------------------------
DATASET_NAME = "blood_cells"   # "malaria" or "blood_cells"


# ---------------------------
# Requested parameters (fixed)
# ---------------------------
TRAIN_SPLIT = "train"
TEST_SPLIT  = "test[:10%]"  # printed; used only if TFDS provides 'test'

num_epochs = 10
batch_size = 128
image_size = 96
patch_size = 8

# ViT core params (CPU friendly)
projection_dim = 64
num_heads = 4
transformer_layers = 4
mlp_head_units = [512, 256]


# ---------------------------
# Print run config
# ---------------------------
print("=== Run Config ===")
print(f'DATASET_NAME: "{DATASET_NAME}"')
print(f'TRAIN_SPLIT: "{TRAIN_SPLIT}"')
print(f'TEST_SPLIT:  "{TEST_SPLIT}"')
print(f"num_epochs: {num_epochs}")
print(f"batch_size: {batch_size}")
print(f"image_size: {image_size}")
print(f"patch_size: {patch_size}")
print("==================\n")


# ---------------------------
# TFDS cache directory
# ---------------------------
DATA_DIR = r"C:\tfds_small"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------
# Helper: find dataset builder robustly (handles name mismatch)
# ---------------------------
def get_builder_with_fallback(name, data_dir):
    candidates = [name]
    # common alternates (safe to try)
    if name == "blood_cells":
        candidates += ["blood_cell", "blood_cells_dataset"]
    if name == "malaria":
        candidates += ["malaria_dataset"]

    last_err = None
    for n in candidates:
        try:
            b = tfds.builder(n, data_dir=data_dir)
            # touching info triggers better errors early
            _ = b.info
            if n != name:
                print(f'NOTE: Using TFDS dataset name "{n}" (fallback from "{name}")')
            return b
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f'Cannot find TFDS dataset for "{name}". Tried: {candidates}. '
        f"Last error: {last_err}"
    )


# ---------------------------
# Helper: resize -> numpy arrays (so np.stack works)
# ---------------------------
def ds_to_numpy_resized(ds, target_hw):
    """
    Convert (img, label) dataset to numpy arrays, resizing each image to target_hw=(H,W).
    """
    x_list, y_list = [], []
    for img, label in tfds.as_numpy(ds):
        img_tf = tf.convert_to_tensor(img)  # uint8, HxWx3, H/W may vary
        img_rs = tf.image.resize(img_tf, target_hw, method="bilinear")
        img_rs = tf.cast(tf.round(img_rs), tf.uint8)  # keep uint8-like
        x_list.append(img_rs.numpy())
        y_list.append(label)

    x = np.stack(x_list).astype("uint8")
    y = np.array(y_list).astype("int32").reshape(-1, 1)
    return x, y


# ---------------------------
# Prepare the data (CIFAR-like: build x_train/x_test first)
# ---------------------------
builder = get_builder_with_fallback(DATASET_NAME, DATA_DIR)

# Make sure data exists (download if needed)
builder.download_and_prepare()
info = builder.info

print("Available splits:", [str(s) for s in info.splits.keys()])

# num_classes from TFDS if available
try:
    num_classes = info.features["label"].num_classes
except Exception:
    # fallback: malaria=2; blood_cells commonly 4
    num_classes = 2 if DATASET_NAME == "malaria" else 4

print("num_classes:", num_classes)

# Force fixed model input shape
input_shape = (image_size, image_size, 3)
print("model input_shape:", input_shape)

SEED = 42
rng = np.random.default_rng(SEED)

has_test = ("test" in info.splits) and ("train" in info.splits)

if has_test:
    # True CIFAR-like: TFDS provides test
    ds_train_raw = builder.as_dataset(split=TRAIN_SPLIT, as_supervised=True)
    ds_test_raw  = builder.as_dataset(split=TEST_SPLIT,  as_supervised=True)

    x_train, y_train = ds_to_numpy_resized(ds_train_raw, (image_size, image_size))
    x_test,  y_test  = ds_to_numpy_resized(ds_test_raw,  (image_size, image_size))
else:
    # TFDS has train only -> manual 90/10 split to create x_test
    print("No TFDS test split found -> manually split train into 90% train_pool and 10% test.")
    ds_pool = builder.as_dataset(split=TRAIN_SPLIT, as_supervised=True)
    x_all, y_all = ds_to_numpy_resized(ds_pool, (image_size, image_size))

    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    N = x_all.shape[0]
    test_size = max(1, int(0.1 * N))

    x_train = x_all[:-test_size]
    y_train = y_all[:-test_size]
    x_test  = x_all[-test_size:]
    y_test  = y_all[-test_size:]

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape} - y_test shape:  {y_test.shape}\n")


# ---------------------------
# Hyperparameters (rest)
# ---------------------------
learning_rate = 0.001
weight_decay  = 0.0001

num_patches = (image_size // patch_size) ** 2
transformer_units = [projection_dim * 2, projection_dim]  # [128, 64]

print("=== Model Params ===")
print(f"num_patches={num_patches}, projection_dim={projection_dim}, num_heads={num_heads}")
print(f"transformer_layers={transformer_layers}, mlp_head_units={mlp_head_units}")
print("====================\n")


# ---------------------------
# Data augmentation (sample-code style)
# ---------------------------
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(x_train)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape_ = ops.shape(images)
        batch_size_ = input_shape_[0]
        height = input_shape_[1]
        width = input_shape_[2]
        channels = input_shape_[3]

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size_,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected = self.projection(patch)
        return projected + self.position_embedding(positions)


def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    augmented = data_augmentation(inputs)

    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)  # sample-code style
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    return keras.Model(inputs=inputs, outputs=logits)


def run_experiment(model):
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    k = min(5, num_classes)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k, name=f"top-{k}-accuracy"),
        ],
    )

    ckpt = f"./checkpoint_{DATASET_NAME}_vit.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        ckpt,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # EXACTLY like the sample code: validation_split=0.1 from x_train
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
        shuffle=True,
    )

    # show implied split sizes (important for your report)
    train_pool = len(x_train)
    val_size = int(np.ceil(0.1 * train_pool))
    actual_train_used = train_pool - val_size
    print("\n=== Effective Split Sizes ===")
    print(f"train_pool = {train_pool}")
    print(f"val_size ≈ {val_size}")
    print(f"actual_train_used ≈ {actual_train_used}")
    print(f"test_size = {len(x_test)}")
    print("=============================\n")

    model.load_weights(ckpt)
    results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    loss = results[0]
    accuracy = results[1]
    topk = results[2]
    print(f"Test accuracy: {accuracy*100:.2f}%")
    print(f"Test top-{k} accuracy: {topk*100:.2f}%")
    return history


vit = create_vit_classifier()
history = run_experiment(vit)


def plot_history(item, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title(f"Train and Validation {item} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()

    out_path = os.path.join(out_dir, f"{DATASET_NAME}_{item}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.show()
    plt.close()


plot_history("loss")
plot_history("accuracy")
