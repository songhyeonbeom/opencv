# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow_datasets as tfds
#
# ds = tfds.load('mnist', split='train', shuffle_files=True)
# print(ds)


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2


# images_train = cv2.cvtColor(images_train, cv2.IMREAD_GRAYSCALE)

data_train, data_test = tf.keras.datasets.mnist.load_data()
(images_train, labels_train) = data_train
(images_test, label_test) = data_test

print(images_train.shape)

X_train = images_train.reshape(images_train.shape[0],28,28,1)
cv2.imshow("1111", X_train[1].reshape(28,28))
cv2.waitKey(0)
