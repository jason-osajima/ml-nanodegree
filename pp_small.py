# Modules to import for project
import tensorflow as tf
import h5py
import time
import numpy as np

#Load the data
h5f = h5py.File('sat-6.h5','r')

X_train = h5f['X_train'][0:1500]
y_train = h5f['y-train'][0:1500]
X_test = h5f['X_test'][0:500]
y_test = h5f['y_test'][0:500]
X_valid = h5f['X_valid'][0:500]
y_valid = h5f['y_valid'][0:500]

h5f.close()

Xy_set = ['X_train', 'y-train', 'X_test', 'y_test', 'X_valid', 'y_valid']

Xy_cnn_set = [X_train, y_train, X_test, y_test, X_valid, y_valid]

h5f = h5py.File('sat-6_small.h5', 'w')
for i in range(len(Xy_set)):
    h5f.create_dataset(Xy_set[i], data=Xy_cnn_set[i])
    print(Xy_set[i] + ' sat-6_small.h5')

h5f.close()


