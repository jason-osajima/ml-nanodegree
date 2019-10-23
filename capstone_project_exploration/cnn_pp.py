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

cnn_X_train = X_train.reshape(243000, 28, 28, 4)
cnn_X_test = X_test.reshape(81000, 28, 28, 4)
cnn_X_valid = X_valid.reshape(81000, 28, 28, 4)

Xy_set = ['X_train', 'y-train', 'X_test', 'y_test', 'X_valid', 'y_valid']

Xy_cnn_set = [cnn_X_train, y_train, cnn_X_test, y_test, cnn_X_valid, y_valid]

h5f = h5py.File('cnn-sat-6.h5', 'w')
for i in range(len(Xy_set)):
    h5f.create_dataset(Xy_set[i], data=Xy_cnn_set[i])
    print(Xy_set[i] + ' successfully written to cnn-sat-6.h5')

h5f.close()


