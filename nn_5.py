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
num_steps = 100001
batch_size = 128

# Function that we can use to measure accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#Initiate an array for every 50 steps
steps = np.arange(0,num_steps,50)
#Initiate a list of arrays for loss for every 50 steps. 
nn_loss3 = np.zeros((num_steps-1)/50+1)

bbatch_size = 128
n_hidden_nodes_12 = 400
n_hidden_nodes_23 = 300
n_hidden_nodes_34 = 200
beta = 0.01
alpha = 0.1

graph = tf.Graph()
with graph.as_default():

    # Input data.
    tf_X_train = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size * layers))
    tf_y_train = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_X_valid = tf.constant(X_valid)
    tf_X_test = tf.constant(X_test)
    
    #A placeholder for the probability that a neuron's output is kept during dropout.
    keep_prob = tf.placeholder(tf.float32)
  
    # Variables. We initialize a set of weights and biases for the two layers of our neural network.
    # Weights_01 is a 3136 x 400 matrix.
    weights_01 = tf.Variable(tf.truncated_normal([image_size * image_size * layers, n_hidden_nodes_12]))
    # Weights_12 is a 400 x 300 matrix
    weights_12 = tf.Variable(tf.truncated_normal([n_hidden_nodes_12, n_hidden_nodes_23]))
    # Weights_23 is a 300 x 200 matrix
    weights_23 = tf.Variable(tf.truncated_normal([n_hidden_nodes_23, n_hidden_nodes_34]))
    # Weights_34 is a 200 x 6 matrix
    weights_34 = tf.Variable(tf.truncated_normal([n_hidden_nodes_34, num_labels]))
    
    # biases_01 is a 1 x 400 matrix
    biases_01 = tf.Variable(tf.zeros([n_hidden_nodes_12]))
    # biases_12 is a 1 x 300 matrix
    biases_12 = tf.Variable(tf.zeros([n_hidden_nodes_23]))
    # biases_23 is a 1 x 200 matrix
    biases_23 = tf.Variable(tf.zeros([n_hidden_nodes_34]))
    # biases_34 is a 1 x 6 matrix
    biases_34 = tf.Variable(tf.zeros([num_labels]))
    
    # Training computation.
    z_01= tf.matmul(tf_X_train, weights_01) + biases_01
    h1 = tf.nn.relu(z_01)
    #dropout
    h1_drop = tf.nn.dropout(h1, keep_prob)
    z_12 = tf.matmul(h1_drop, weights_12) + biases_12
    
    h2 = tf.nn.relu(z_12)
    #dropout
    h2_drop = tf.nn.dropout(h2, keep_prob)
    z_23 = tf.matmul(h2_drop, weights_23) + biases_23
    
    h3 = tf.nn.relu(z_23)
    #dropout
    h3_drop = tf.nn.dropout(h3, keep_prob)
    z_34 = tf.matmul(h3_drop, weights_34) + biases_34
    
    #Introduce regularization with parameter beta. 
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(z_34, tf_y_train))
    #loss with regularization
    loss = loss + beta * (tf.nn.l2_loss(weights_01) + 
                          tf.nn.l2_loss(weights_12) + tf.nn.l2_loss(weights_23) 
                          + tf.nn.l2_loss(weights_34))
    
    #optimizer
    ada_optimizer = tf.train.AdagradOptimizer(alpha).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(z_12)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(
                            tf.nn.relu(tf.matmul(tf_X_valid, weights_01) + biases_01), 
                            weights_12) + biases_12), weights_23)+biases_23), weights_34) + biases_34)
    test_prediction = tf.nn.softmax(
        tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(
                            tf.nn.relu(tf.matmul(tf_X_test, weights_01) + biases_01), 
                            weights_12) + biases_12), weights_23)+biases_23), weights_34) + biases_34)



    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print ("Initialized")
        start = time.time()
        for step in xrange(num_steps):
            # Pick an offset within the training data, which has been randomized.
            offset = (step * batch_size) % (X_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = X_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_X_train : batch_data, tf_y_train : batch_labels, keep_prob: 0.5}
            _, l, predictions = session.run(
              [ada_optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 1000 == 0):
                print ("Minibatch loss at step", step, ":", l)
                print ("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print ("Validation accuracy: %.1f%%" % accuracy(
                        valid_prediction.eval(), y_valid))
        end = time.time()
        print ("Training time (secs): {:.5f}".format(end - start))
        start = time.time()
        print ("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), y_test))
        end = time.time()
        print ("Prediction time (secs): {:.5f}".format(end - start))

