import numpy as np
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("./repo/MNIST", one_hot = True)
imgs = mnist.train.images

def weight_variable(shape):
    #initial = tf.constant(0.0, shape = shape)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

activation = tf.nn.relu
#activation = tf.sigmoid
learning_rate = 2e-4
	
x = tf.placeholder("float", shape=[None, 784])
# convert to a stack of images
x_image = tf.reshape(x, [-1, 28, 28, 1])

# conv1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = activation(conv2d(x_image, W_conv1) + b_conv1)

# conv_e1
W_conv_e1 = weight_variable([5, 5, 32, 32])
b_conv_e1 = bias_variable([32])
h_conv_e1 = activation(conv2d(h_conv1, W_conv_e1) + b_conv_e1)

# max-pool_e1
h_pool_e1 = max_pool_2x2(h_conv_e1)

# conv2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = activation(conv2d(h_pool_e1, W_conv2) + b_conv2)

# max-pool1
h_pool2 = max_pool_2x2(h_conv2)

# dense
final_size = 7*7*64
#final_size = 14*14*32
h_pool2_flat = tf.reshape(h_pool2, [-1, final_size])
#h_pool2_flat = tf.reshape(h_pool1, [-1, final_size])

W_fc1 = weight_variable([final_size, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = activation(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train
y_ = tf.placeholder("float", shape=[None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(2001):
	batch = mnist.train.next_batch(50)
  	if i%500 == 0:
		train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
batch = mnist.test.next_batch(500)
#plt.imshow(sess.run(h_pool2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})[0, 0:6, 0:6, 0])
print("test accuracy %g"%sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
