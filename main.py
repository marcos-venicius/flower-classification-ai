#!/usr/bin/env python3
import PIL
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from utils import view_train_results

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data = Path(data)

EPOCHS = 10

checkpoint_path = "data/epochs"

# batch size
batch_size = 32

img_width = 180
img_height = 180

input_shape = (img_height, img_width, 3)
img_size = (img_height, img_width)

validation_split = 0.2

# dataset for training
train_ds = image_dataset_from_directory(
    data,
    validation_split=validation_split,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# dataset for validation
val_ds = image_dataset_from_directory(
    data,
    validation_split=validation_split,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

# keeps the images in memory after they're loaded off disk during the first epoch
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# overlaps data preprocessing and model execution while training
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=input_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ]
)

# create network model
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=input_shape),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name='outputs')
])

if os.path.isfile(checkpoint_path + ".index"):
	model.load_weights(checkpoint_path)

# compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

initial_epoch = model.optimizer.iterations.numpy() # steps per epoch

cp_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath=checkpoint_path,
	save_weights_only=True,
)

# run epochs
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
	initial_epoch=initial_epoch,
	batch_size=batch_size,
	callbacks=[cp_callback]
)

if os.getenv('VIEW_RESULTS') == '1':
	# view results
	view_train_results(history, EPOCHS)
