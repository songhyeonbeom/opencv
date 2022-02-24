import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_datasets as tfds

ds = tfds.load('mnist', split='train', shuffle_files=True)


