import tensorflow as tf

# saving checkpoint
save_file = "./model.ckpt"

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Weights:")
    print(sess.run(weights))
    print("Bias:")
    print(sess.run(bias))

    saver.save(sess, save_file)