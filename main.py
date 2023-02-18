#!/usr/bin/env python3

import PIL
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

from keras.utils import image_dataset_from_directory

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data = Path(data)

# batch size
batch_size = 32

img_width = 180
img_height = 180

# dataset for training
train_ds = image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# dataset for validation
val_ds = image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

print(class_names)
