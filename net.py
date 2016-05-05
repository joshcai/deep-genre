import os
import random
import sys
import argparse

from scipy import misc
import numpy as np

import tensorflow as tf


import Image

parser = argparse.ArgumentParser()
parser.add_argument('--use_saved', help='Continue from saved checkpoint')
parser.add_argument('--reconstruct', help='Try to reconstruct song')
args = parser.parse_args()

# list of 100x512 arrays (inputs)
xs = []
# list of one-hot vectors with corresponding genre output as 1 (outputs)
ys = []

# list of 100x512 arrays for validation set
validation_xs = []
# corresponding outputs for validation
validation_ys = []

# Number of genres
num_genres = 10

# Read image into numpy array
for i, folder in enumerate(sorted(os.listdir('genres_png'))):
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


# creates a new weight variable with the shape specified
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# creates a new bias variable with the shape specified
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# performs a convolution on x with W as the filter
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# performas 1x4 average pooling 
def avg_pool_1x4(x):
  return tf.nn.avg_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 1, 4, 1], padding='SAME')

# number of features in first layer
num_features1 = 32

W_conv1 = weight_variable([1, 4, 1, num_features1])
b_conv1 = bias_variable([num_features1])

x_image = tf.reshape(x, [-1, height, width, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = avg_pool_1x4(h_conv1)

# number of features in second layer
num_features2 = 64

W_conv2 = weight_variable([1, 4, num_features1, num_features2])
b_conv2 = bias_variable([num_features2])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_1x4(h_conv2)

# number of features in third layer
num_features3 = 32

W_conv3 = weight_variable([1, 4, num_features2, num_features3])
b_conv3 = bias_variable([num_features3])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = avg_pool_1x4(h_conv3)

# number of nodes in fully-connected layer
nodes = 128

W_fc1 = weight_variable([width / 64 * 100 * num_features3, nodes])
b_fc1 = bias_variable([nodes])

h_pool2_flat = tf.reshape(h_pool3, [-1, width / 64 * 100 * num_features3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([nodes, nodes])
b_fc2 = bias_variable([nodes])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# dropout rate
keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([nodes, 2])
b_fc3 = bias_variable([2])

# performs softmax to get probabilities to sum to 1
y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


saver = tf.train.Saver()

# objective function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# set up for decaying learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = .01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
# tests to see if prediction was the same as correct output
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# get percentage of correct predictions in a test set
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# try to reconstruct a song from first layers
if args.reconstruct:
  content = xs[0]

  with tf.Session() as sess:
    saver.restore(sess, 'model.ckpt')
    content_features = h_conv1.eval(feed_dict={x: [content]})
    shape = (1,) + content.shape
    noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
    initial = tf.random_normal(shape) * 0.256
    image = tf.Variable(initial)

    image_reshaped = tf.reshape(image, [-1, height, width, 1])
    conv = tf.nn.conv2d(image_reshaped, W_conv1, strides=(1, 1, 1, 1),
            padding='SAME')
    conv2 = tf.nn.bias_add(conv, b_conv1)
    image_features = tf.nn.relu(conv2)
    content_loss = (2 * tf.nn.l2_loss(
            image_features - content_features) /
            content_features.size)
    train_step2 = tf.train.AdamOptimizer(.001).minimize(content_loss)
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, 'model.ckpt')
    for i in range(2000):
      last_step = (i == 1999)

      if i % 100 == 0 or last_step:
        print('loss at step %s: %s' % (str(i), str(content_loss.eval())))
      train_step2.run()

# train neural net to recognize genre
else:
  with tf.Session() as sess:
    if args.use_saved:
      # trains from saved checkpoint
      saver.restore(sess, 'model.ckpt')
    else:
      # starts training from new point
      sess.run(tf.initialize_all_variables())
    for i in range(100000):
      if i % 25 == 0:
        print('%s, ' % str(i))
      # gets random indices for mini-batch gradient descent to speed up training
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

