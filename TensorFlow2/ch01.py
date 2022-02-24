# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
#
# with tf.GradientTape() as tape:
#     tape.watch(model.input)
#     model_vals = model(v)


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

input_data = [1,2,3]
x = tf.placeholder(dtype=tf.float32)
y = x * 2

sess = tf.Session()
result = sess.run(y, feed_dict={x:input_data})

print(result)