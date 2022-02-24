# import epoch as epoch
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2


ds = tfds.load("mnist", split="train", shuffle_files=True)
data_train, data_test = tf.keras.datasets.mnist.load_data()
(image_train, label_train) = data_train
(image_test, label_test) = data_test

inputs = Input(shape=(28,))
x = Dense(8, activation="relu")(inputs)
x = Dense(4, activation="relu")(x)
x = Dense(1, activation="softmax")(x)

model = Model(inputs, x)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(image_train, label_train,
          epochs=10, batch_size=100,
          validation_data=(image_test, label_test),
          callbacks=[tensorboard_callback])



