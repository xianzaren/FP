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