"""
ViT patch classification on TFDS 'patch_camelyon' (PCam) - CPU friendly
Fix truncated/corrupted TFDS cache by:
- Using a NEW isolated TFDS data_dir
- Forcing re-download + re-prepare
- Locking download_dir + extract_dir inside the same data_dir

Keeps sample-code style:
- x_train/y_train/x_test/y_test are numpy arrays
- data_augmentation.layers[0].adapt(x_train)
- model.fit(..., validation_split=0.1)
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow_datasets as tfds
import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# 1) Put TFDS cache in a clean directory (C: exists)
# ---------------------------
DATA_DIR = r"C:\tfds_pcam_clean"
os.makedirs(DATA_DIR, exist_ok=True)

# CPU quick-check splits
TRAIN_SPLIT = "train[:0.1%]"   # after it works, try "train[:1%]" or "train[:5%]"
TEST_SPLIT  = "test[:1%]"
SEED = 42


# ---------------------------
# 2) Force re-download + re-prepare (avoid corrupted .h5 reuse)
# ---------------------------
print("TFDS data_dir:", DATA_DIR)
print("Forcing TFDS re-download & re-prepare...")

download_config = tfds.download.DownloadConfig(
    download_dir=os.path.join(DATA_DIR, "downloads", "raw"),
    extract_dir=os.path.join(DATA_DIR, "downloads", "extracted"),
    manual_dir=os.path.join(DATA_DIR, "manual"),
    download_mode=tfds.GenerateMode.FORCE_REDOWNLOAD,
)

builder = tfds.builder("patch_camelyon", data_dir=DATA_DIR)
builder.download_and_prepare(download_config=download_config)
info = builder.info

print("Dataset prepared.")
print("Available splits:", list(info.splits.keys()))
print("Total train:", info.splits["train"].num_examples)
print("Total test :", info.splits["test"].num_examples)


# ---------------------------
# 3) TFDS -> numpy arrays (sample-code style)
# ---------------------------
num_classes = 2
input_shape = (96, 96, 3)

ds_train = builder.as_dataset(split=TRAIN_SPLIT, as_supervised=True)
ds_test  = builder.as_dataset(split=TEST_SPLIT,  as_supervised=True)

def ds_to_numpy(ds):
    x_list, y_list = [], []
    for img, label in tfds.as_numpy(ds):
        x_list.append(img)
        y_list.append(label)
    x = np.stack(x_list).astype("uint8")
    y = np.array(y_list).astype("int32").reshape(-1, 1)
    return x, y

x_train, y_train = ds_to_numpy(ds_train)
x_test,  y_test  = ds_to_numpy(ds_test)

# Shuffle once so validation_split is effectively random
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(x_train))
x_train, y_train = x_train[perm], y_train[perm]

print(f"TRAIN_SPLIT={TRAIN_SPLIT} -> x_train {x_train.shape}, y_train {y_train.shape}")
print(f"TEST_SPLIT ={TEST_SPLIT}  -> x_test  {x_test.shape},  y_test  {y_test.shape}")


# ---------------------------
# 4) Hyperparameters (CPU friendly)
# ---------------------------
learning_rate = 1e-3
weight_decay  = 1e-4
batch_size    = 128
num_epochs    = 10

image_size = 96
patch_size = 8
num_patches = (image_size // patch_size) ** 2

projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]  # [128, 64]
transformer_layers = 4
mlp_head_units = [256, 128]


# ---------------------------
# 5) Data augmentation (same style)
# ---------------------------
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
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
        b = ops.shape(images)[0]
        h = ops.shape(images)[1]
        w = ops.shape(images)[2]
        c = ops.shape(images)[3]
        nh = h // self.patch_size
        nw = w // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        return ops.reshape(patches, (b, nh * nw, self.patch_size * self.patch_size * c))


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(0, self.num_patches, 1), axis=0)
        projected = self.projection(patch)
        return projected + self.position_embedding(positions)


def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    x = data_augmentation(inputs)

    x = Patches(patch_size)(x)
    x = PatchEncoder(num_patches, projection_dim)(x)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attn, x])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, dropout_rate=0.1)
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Keep sample-code style: Flatten
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    x = mlp(x, mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(x)
    return keras.Model(inputs=inputs, outputs=logits)


def run_experiment(model):
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    ckpt = "./checkpoint_pcam_vit.weights.h5"
    cb = keras.callbacks.ModelCheckpoint(
        ckpt, monitor="val_accuracy", save_best_only=True, save_weights_only=True
    )

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        shuffle=True,
        callbacks=[cb],
    )

    model.load_weights(ckpt)
    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print(f"Test accuracy: {acc*100:.2f}%")
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
    plt.title(f"Train vs Val {item}", fontsize=14)
    plt.legend()
    plt.grid()

    out_path = os.path.join(out_dir, f"pcam_{item}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print("Saved:", out_path)

    plt.show()
    plt.close()

plot_history("loss")
plot_history("accuracy")
