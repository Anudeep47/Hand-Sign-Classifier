import numpy as np
import h5py
import math

def load_dataset():
    train_data = h5py.File('data/train_signs.h5', 'r')
    test_data = h5py.File('data/test_signs.h5', 'r')
    # train data
    train_set_x = np.array(train_data['train_set_x'][:])
    train_set_y = np.array(train_data['train_set_y'][:])
    # test data
    test_set_x = np.array(test_data['test_set_x'][:])
    test_set_y = np.array(test_data['test_set_y'][:])
    # classes
    classes = np.array(test_data['list_classes'][:])
    return train_set_x, train_set_y, test_set_x, test_set_y, classes

def random_minibatches(x, y, minibatch_size=64):
    m = x.shape[0]
    minibatches=[]
    # randomizing the data
    random_order = list(np.random.permutation(m))
    x_random = x[random_order, :, :, :]
    y_random = y[random_order, :]
    # slicing data into minibatches
    n_minibatches = math.ceil(float(m)/minibatch_size)
    for i in range(int(n_minibatches)):
        x_minibatch = x_random[i*minibatch_size : (i+1)*minibatch_size, :, :, :]
        y_minibatch = y_random[i*minibatch_size : (i+1)*minibatch_size, :]
        minibatch = (x_minibatch, y_minibatch)
        minibatches.append(minibatch)
    return minibatches, n_minibatches
