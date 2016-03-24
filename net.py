import os
import random
import sys
import argparse

from scipy import misc
import numpy as np

import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--use_saved', help='Continue from saved checkpoint')
args = parser.parse_args()

xs = []
ys = []

validation_xs = []
validation_ys = []

# Number of genres
num_genres = 2

# Read image into numpy array
for i, folder in enumerate(['classical', 'hiphop']):
  for j, f in enumerate(os.listdir('genres_png/%s' % folder)):
    img = misc.imread('genres_png/%s/%s' % (folder, f), flatten=True)
    img_split = np.split(img, [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608], axis=1)
    img_split2 = [x.reshape(512 * 100) for x in img_split[:8]]

    if j == 99:
      validation_xs.extend(img_split2)
    else:
      xs.extend(img_split2)

    for _ in xrange(len(img_split2)):
      new_array = [0] * num_genres
      new_array[i] = 1
      if j == 99:
        validation_ys.append(new_array)
      else:
        ys.append(new_array)
  print 'added %s ' % folder

# Height and width of input
height = 100  # number of frequency bins
width = 512  # number of frames (150 frames in one second)


x = tf.placeholder('float', [None, height * width])
y_ = tf.placeholder('float', [None, num_genres])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_1x4(x):
  return tf.nn.avg_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 1, 4, 1], padding='SAME')

num_features1 = 32

W_conv1 = weight_variable([1, 4, 1, num_features1])
b_conv1 = bias_variable([num_features1])

x_image = tf.reshape(x, [-1, height, width, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = avg_pool_1x4(h_conv1)

num_features2 = 64

W_conv2 = weight_variable([1, 4, num_features1, num_features2])
b_conv2 = bias_variable([num_features2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_1x4(h_conv2)

num_features3 = 32

W_conv3 = weight_variable([1, 4, num_features2, num_features3])
b_conv3 = bias_variable([num_features3])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = avg_pool_1x4(h_conv3)

nodes = 128

W_fc1 = weight_variable([width / 64 * 100 * num_features3, nodes])
b_fc1 = bias_variable([nodes])

h_pool2_flat = tf.reshape(h_pool3, [-1, width / 64 * 100 * num_features3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([nodes, nodes])
b_fc2 = bias_variable([nodes])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([nodes, 2])
b_fc3 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


saver = tf.train.Saver()

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = .01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
  if args.use_saved:
    saver.restore(sess, 'model.ckpt')
  else:
    sess.run(tf.initialize_all_variables())
  for i in range(100000):
    if i % 25 == 0:
      print('%s, ' % str(i))
    random_indices = random.sample(xrange(len(xs)), 20)
    batch_xs = [xs[index] for index in random_indices]
    batch_ys = [ys[index] for index in random_indices]
    if i % 1000 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      print "step %d, training accuracy %g"%(i, train_accuracy)
      save_path = saver.save(sess, './model.ckpt')
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    if i % 5000 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: validation_xs, y_: validation_ys, keep_prob: 1.0})
      save_path = saver.save(sess, './model%s.ckpt' % str(i))
      print "step %d, validation accuracy %g"%(i, train_accuracy)


