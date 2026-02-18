"""
ViT patch classification on TFDS 'malaria' - sample-code style split (CIFAR-like)

Fully mirrors the original Keras CIFAR example split logic:
1) Build x_train/y_train and x_test/y_test first (like load_data() provides)
2) model.fit(..., validation_split=0.1) to create a validation set from x_train
3) model.evaluate(x_test, y_test) for final test evaluation

Fix: malaria images may have different H/W -> resize BEFORE np.stack.

Also prints:
TRAIN_SPLIT, TEST_SPLIT, num_epochs, batch_size, image_size, patch_size
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
# Dataset
# ---------------------------
DATASET_NAME = "malaria"  # keep malaria here


# ---------------------------
# Requested parameters (fixed as you asked)
# NOTE: For malaria on your machine, TFDS shows only Split('train').
# So TEST_SPLIT is printed but may not be used.
# ---------------------------
TRAIN_SPLIT = "train"
TEST_SPLIT  = "test[:10%]" 

num_epochs = 10
batch_size = 128
image_size = 96
patch_size = 8

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
# Helper: resize -> numpy arrays (so np.stack works)
# ---------------------------
def ds_to_numpy_resized(ds, target_size_hw):
    """
    Convert (img, label) dataset to numpy arrays, resizing each image to target_size_hw.
    target_size_hw: (H, W)
    """
    x_list, y_list = [], []
    for img, label in tfds.as_numpy(ds):
        img_tf = tf.convert_to_tensor(img)  # uint8, HxWx3, H/W may vary
        img_rs = tf.image.resize(img_tf, target_size_hw, method="bilinear")
        img_rs = tf.cast(tf.round(img_rs), tf.uint8)  # keep uint8-like
        x_list.append(img_rs.numpy())
        y_list.append(label)

    x = np.stack(x_list).astype("uint8")
    y = np.array(y_list).astype("int32").reshape(-1, 1)
    return x, y


# ---------------------------
# Prepare the data (CIFAR-like: get x_train/x_test first)
# ---------------------------
# Load info for splits + num_classes
_, ds_info = tfds.load(
    DATASET_NAME,
    split="train[:1%]",
    with_info=True,
    data_dir=DATA_DIR,
)

print("Available splits:", list(ds_info.splits.keys()))

# num_classes from TFDS if available
try:
    num_classes = ds_info.features["label"].num_classes
except Exception:
    num_classes = 2

print("num_classes:", num_classes)

# Force fixed input shape (malaria can be (None,None,3))
input_shape = (image_size, image_size, 3)
print("model input_shape:", input_shape)

SEED = 42
rng = np.random.default_rng(SEED)

if "train" in ds_info.splits and "test" in ds_info.splits:
    # If TFDS provides test, we use it directly (most CIFAR-like)
    ds_train_raw, ds_test_raw = tfds.load(
        DATASET_NAME,
        split=[TRAIN_SPLIT, TEST_SPLIT],
        as_supervised=True,
        data_dir=DATA_DIR,
    )
    x_train, y_train = ds_to_numpy_resized(ds_train_raw, (image_size, image_size))
    x_test, y_test   = ds_to_numpy_resized(ds_test_raw,  (image_size, image_size))

else:
    # Your environment: malaria only has 'train'.
    # We will create CIFAR-like (train, test) by manual 90/10 split.
    # IMPORTANT: We load TRAIN_SPLIT as the "full pool" (can be "train" or a subset).
    ds_pool = tfds.load(
        DATASET_NAME,
        split=TRAIN_SPLIT,   # change to "train" if you want all data
        as_supervised=True,
        data_dir=DATA_DIR,
    )
    x_all, y_all = ds_to_numpy_resized(ds_pool, (image_size, image_size))

    # Shuffle once BEFORE splitting (so the split is random)
    idx = rng.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    # CIFAR-like fixed test set (10%)
    N = x_all.shape[0]
    test_size = max(1, int(0.1 * N))

    x_train = x_all[:-test_size]
    y_train = y_all[:-test_size]
    x_test  = x_all[-test_size:]
    y_test  = y_all[-test_size:]

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape} - y_test shape:  {y_test.shape}\n")


# ---------------------------
# Hyperparameters
# ---------------------------
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256          # <-- was 128
num_epochs = 10           # keep

num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8     # <-- was 4
mlp_head_units = [2048, 1024] 


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
