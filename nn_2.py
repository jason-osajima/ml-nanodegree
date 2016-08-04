# Modules to import for project
import tensorflow as tf
import h5py
import time
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
num_steps = 20001
batch_size = 128

# Function that we can use to measure accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 128
n_hidden_nodes = 100

# batch_size = 128
n_hidden_nodes = 100
beta = 0.01
alpha = 0.001

graph = tf.Graph()
with graph.as_default():
    
    #functions to initialize weights and biases.
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    #Placeholders for the input images and output target classes.
    X = tf.placeholder(tf.float32, shape=[None, 3136])
    y_ = tf.placeholder(tf.float32, shape=[None, 6])
    
    #A placeholder for the probability that a neuron's output is kept during dropout.
    keep_prob = tf.placeholder(tf.float32)
    
    
    # First hidden layer.
    # Set the variables.
    W_1 = weight_variable([3136, n_hidden_nodes])
    b_1 = bias_variable([1, n_hidden_nodes])
    
    #Perform matrix multiplication, add bias, and put it through the ReLu 
    z1 = tf.matmul(X, W_1)+b_1
    h1 = tf.nn.relu(z1)
    
    
    
    #A placeholder for the probability that a neuron's output is kept during dropout.
    keep_prob = tf.placeholder(tf.float32)
  
    # Variables. We initialize a set of weights and biases for the two layers of our neural network.
    # Weights_01 is a 3136 x 100 matrix.
    weights_01 = tf.Variable(tf.truncated_normal([image_size * image_size * layers, n_hidden_nodes]))
    # Weights_12 is a 100 x 6 matrix
    weights_12 = tf.Variable(tf.truncated_normal([n_hidden_nodes, num_labels]))
    # biases_01 is a 1 x 100 matrix
    biases_01 = tf.Variable(tf.zeros([n_hidden_nodes]))
    # biases_12 is a 1 x 6 matrix
    biases_12 = tf.Variable(tf.zeros([num_labels]))
    
    # Training computation.
    z_01= tf.matmul(tf_X_train, weights_01) + biases_01
    h1 = tf.nn.relu(z_01)
    #dropout
    h1_drop = tf.nn.dropout(h1, keep_prob)
    z_12 = tf.matmul(h1_drop, weights_12) + biases_12
    
    #Introduce regularization with parameter beta. 
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(z_12, tf_y_train))
    #loss with regularization
    loss = loss + beta * (tf.nn.l2_loss(weights_01) + tf.nn.l2_loss(weights_12) )
    
    #optimizer
    ada_optimizer = tf.train.AdagradOptimizer(alpha).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(z_12)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_X_valid, weights_01) + biases_01), weights_12) + biases_12)
    test_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf_X_test, weights_01) + biases_01), weights_12) + biases_12)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("Initialized")
    start = time.time()
    for step in xrange(num_steps):
        # Pick an offset within the training data, which has been randomized.
        offset = (step * batch_size) % (X_train_norm.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = X_train_norm[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_X_train : batch_data, tf_y_train : batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run(
          [ada_optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 1000 == 0):
            print ("Minibatch loss at step", step, ":", l)
            print ("Training accuracy: %.1f%%" % accuracy(train_prediction.eval({tf_X_train : batch_data, 
                                           tf_y_train : batch_labels, keep_prob: 1.0}), batch_labels))
            print ("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval({keep_prob: 1.0}), y_valid))
    end = time.time()
    print ("Training time (secs): {:.5f}".format(end - start))