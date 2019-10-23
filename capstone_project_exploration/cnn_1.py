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

#land_cover labels
land_cover = ['buildings', 'barren_land', 'trees', 'grassland', 'roads', 'water_bodies']

#Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

#Network Parameters
n_input = 3136 #Image data input (28*28*4)
n_classes = len(land_cover)
dropout = 0.75

#tf graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes]) 
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

#create wrappers
def conv2d(x, W, b, strides = 1):
    #conv2d wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    #maxpool 2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

#create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 4])
    #Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #max pooling
    conv1 = maxpool2d(conv1, k=2)

    #Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #max pooling
    conv2 = maxpool2d(conv2, k=2)

    #fully connected layer
    #reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    #apply droupout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    #output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out

#store layer weights and biases
weights = {
    #5 x 5 conv, 4 inputs, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5,5,4,32])),
    #5 x 5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    #fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    #1024 inputs, 6 outputs (class prediction)
    'out' : tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#construct model
pred = conv_net(x, weights, biases, keep_prob)

#define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Pick an offset within the training data, which has been randomized.
        offset = (step * batch_size) % (X_train.shape[0] - batch_size)
        
        # Generate a minibatch.
        batch_data = X_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_data, y: batch_labels,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_data, y: batch_labels, keep_prob: 1.})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X_valid,
                                      y: y_valid,
                                      keep_prob: 1.})