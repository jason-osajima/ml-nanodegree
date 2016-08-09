# Modules to import for project
import tensorflow as tf
import h5py
import time
import numpy as np

#Load the data
h5f = h5py.File('cnn-sat-6.h5','r')

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

beta = 0.01
alpha = 1e-4

graph = tf.Graph()
with graph.as_default():

    #functions to initialize weights and biases.
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    #functions to initialize convolutions and max pooling.
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    #Placeholders for the input images and output target classes.
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 4])
    y_ = tf.placeholder(tf.float32, shape=[None, 6])
    
    # First convolutional layer.
    # Reshape x to a 4d tensor.
    x_image = tf.reshape(X, [-1,28,28,4])
    # Set the variables.
    W_conv1 = weight_variable([5, 5, 4, 32])
    b_conv1 = bias_variable([32])
    # Perform convolution and max-pooling.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    
    # Second convolutional layer.
    # Set the variables.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # Perform convolution and max-pooling.
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    
    # Densely connected layer.
    # Set the variables.
    W_fc1 = weight_variable([7 * 7 * 64, 512])
    b_fc1 = bias_variable([512])
    #Perform matrix multiplication, add a bias, and put it through the ReLu.
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    
    #Add Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    
    #Add softmax.
    W_fc2 = weight_variable([512, 6])
    b_fc2 = bias_variable([6])
    z_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv=tf.nn.softmax(z_fc2)
    
    
    #Introduce regularization with parameter beta. 
    #Calculate the loss with regularization.
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(z_fc2, y_))
    loss = loss + beta * tf.nn.l2_loss(W_fc2)
    
    
    #optimizer
    ada_optimizer = tf.train.AdagradOptimizer(alpha).minimize(loss)
    
    #accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print ("Initialized")
    start = time.time()
    
    for i in xrange(num_steps):
        #Create an offset
        offset = (i * batch_size) % (X_train.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        if i%1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    X:batch_data, y_: batch_labels, keep_prob: 1.0})
            valid_accuracy = accuracy.eval(feed_dict={
                    X:X_valid, y_: y_valid, keep_prob: 1.0})
        
        print("step %d, training accuracy %g and loss %g"%(i, train_accuracy))
        print("step %d, cross validation accuracy %g"%(i, valid_accuracy))
        print("------------------------------------------")

        ada_optimizer.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

