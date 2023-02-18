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

validation_split = 0.2

# dataset for training
train_ds = image_dataset_from_directory(
    data,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# dataset for validation
val_ds = image_dataset_from_directory(
    data,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

# keeps the images in memory after they're loaded off disk during the first epoch
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# overlaps data preprocessing and model execution while training
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
