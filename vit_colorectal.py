"""
Title: Image classification with Vision Transformer (ViT) on TFDS colorectal_histology
Based on: Keras example by Khalid Salama (2021/01/18)

Goal: Keep the original sample-code style:
- Use x_train/y_train/x_test/y_test as numpy arrays
- Use data_augmentation.layers[0].adapt(x_train)
- Use model.fit(..., validation_split=0.1)
"""

# ---------------------------
# Setup
# ---------------------------
import os

# For CPU + TFDS + Keras, TensorFlow backend is the most reliable.
# If you insist on "jax", you can switch, but TFDS/numpy pipeline is simplest with TF.
os.environ["KERAS_BACKEND"] = "tensorflow"  # ["tensorflow", "jax", "torch"]

import tensorflow as tf
import tensorflow_datasets as tfds

import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Prepare the data (TFDS -> numpy arrays, keep original style)
# ---------------------------
# colorectal_histology usually: 8 classes, 150x150x3 patches (TFDS provides train only on your machine)
num_classes = 8
input_shape = (150, 150, 3)

# Load the whole dataset (only "train" split exists in your TFDS build)
ds_all, ds_info = tfds.load(
    "colorectal_histology",
    split="train",
    as_supervised=True,   # yields (image, label)
    with_info=True
)

print("Available splits:", list(ds_info.splits.keys()))
print("Total examples:", ds_info.splits["train"].num_examples)

# Convert TFDS dataset to numpy arrays (5000 images -> feasible in RAM)
x_list, y_list = [], []
for img, label in tfds.as_numpy(ds_all):
    x_list.append(img)
    y_list.append(label)

x_all = np.stack(x_list).astype("uint8")                  # (N, 150, 150, 3)
y_all = np.array(y_list).astype("int32").reshape(-1, 1)   # (N, 1)

# Shuffle once to make splits random (recommended; original validation_split is not shuffled)
rng = np.random.default_rng(42)
idx = rng.permutation(len(x_all))
x_all, y_all = x_all[idx], y_all[idx]

# Manually create a test set (e.g., last 10%)
N = x_all.shape[0]
test_size = int(0.1 * N)

x_train = x_all[:-test_size]
y_train = y_all[:-test_size]
x_test = x_all[-test_size:]
y_test = y_all[-test_size:]

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape:  {x_test.shape} - y_test shape:  {y_test.shape}")


# ---------------------------
# Configure the hyperparameters
# ---------------------------
# ---------------------------
# Configure the hyperparameters (MATCH ORIGINAL KERAS EXAMPLE)
# ---------------------------
# Configure the hyperparameters (MATCH ORIGINAL KERAS EXAMPLE)
# ---------------------------
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256          # <-- was 128
num_epochs = 10
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8     # <-- was 4
mlp_head_units = [2048, 1024]  # <-- was [512, 256]

# ---------------------------
# Use data augmentation
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
# Compute mean/variance for normalization (same style as original code)
data_augmentation.layers[0].adapt(x_train)


# ---------------------------
# Implement multilayer perceptron (MLP)
# ---------------------------
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# ---------------------------
# Implement patch creation as a layer
# ---------------------------
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

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


# ---------------------------
# Display patches for a sample image (kept from original, now uses x_train)
# ---------------------------
plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")
plt.show()


# ---------------------------
# Implement the patch encoding layer
# ---------------------------
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


# ---------------------------
# Build the ViT model
# ---------------------------
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

    # Keep original "Flatten" style (you asked to keep sample code style)
    representation = layers.Flatten()(representation)

    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# ---------------------------
# Compile, train, and evaluate the model (keep original style)
# ---------------------------
def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "./checkpoint_colorectal_vit.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,     # SAME AS ORIGINAL
        callbacks=[checkpoint_callback],
        shuffle=True,            # ensure training data is shuffled each epoch
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top-5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


# ---------------------------
# Plot history (same style)
# ---------------------------
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

    out_path = os.path.join(out_dir, f"{item}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.show()
    plt.close()


plot_history("loss")
plot_history("top-5-accuracy")
