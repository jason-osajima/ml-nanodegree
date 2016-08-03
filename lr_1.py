# Modules to import for project

import tensorflow as tf
import h5py
import numpy as np

#Load the data
h5f = h5py.File('sat-6.h5','r')

X_train = h5f['X_train'][:]
y_train = h5f['y-train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_valid = h5f['X_valid'][:]
y_valid = h5f['y_valid'][:]

h5f.close()

#number of land_cover labels
land_cover = ['buildings', 'barren_land', 'trees', 'grassland', 'roads', 'water_bodies']
num_labels = len(land_cover)
image_size = 28
layers = 4

batch_size = 128

graph = tf.Graph()
with graph.as_default():

    # Input the data into constants that are attached to the graph. 

    tf_X_train = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size * layers))
    tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_X_valid = tf.constant(X_valid)
    tf_X_test = tf.constant(X_test)

    # These are the parameters that we are going to be training. We will initialize the
    # weight matrix using random valued following a truncated normal distribution. The
    # biases get initialized to zero.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size * layers, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy. We take the average of the cross-entropy
    # across all training examples which is our loss.
    logits = tf.matmul(tf_X_train, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_y_train))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_X_valid, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_X_test, weights) + biases)


num_steps = 3001


# Function that we can use to measure accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("Initialized")
    for step in xrange(num_steps):
        # Pick an offset within the training data, which has been randomized.
        offset = (step * batch_size) % (X_train.shape[0] - batch_size)
        
        # Generate a minibatch.
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        
        # Prepare a dictionary telling the session where to feed the minibatch. The key of the dictionary is the placeholder node of the graph to be fed, and the value is the numpy array to feed to it.
        feed_dict = {tf_X_train : batch_data, tf_y_train : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print ("Minibatch loss at step", step, ":", l)
            print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print ("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), y_valid))
    print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), y_test))