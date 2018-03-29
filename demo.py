import numpy as np
import tensorflow as tf
from model_utils import *
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
# %matplotlib inline

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    Z1 = tf.nn.conv2d(X, W1, [1,1,1,1], 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, [1,8,8,1], [1,8,8,1], 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, [1,1,1,1], 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, [1,4,4,1], [1,4,4,1], 'SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    return Z3

def cnn_model(x_train, y_train, x_test, y_test, learning_rate=0.009, epochs=100, minibatch_size=64, print_cost=True):
    # ops.reset_default_graph()
    (m, n_H0, n_W0, n_C0) = x_train.shape
    n_y = y_train.shape[1]
    costs=[]

    # creating placeholders
    x = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    y = tf.placeholder(tf.float32, shape=(None, n_y))

    # initializing parameters
    W1 = tf.get_variable('W1', [4,4,3,8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2,2,8,16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {'W1': W1, 'W2': W2}

    # forward propagation
    Z3 = forward_propagation(x, parameters)

    # computing cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=y))

    # back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # initialize variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            n_minibatch_cost = 0
            minibatches, n_minibatches = random_minibatches(x_train, y_train, minibatch_size)
            for minibatch in minibatches:
                x_minibatch, y_minibatch = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={x: x_minibatch, y: y_minibatch})
                n_minibatch_cost += minibatch_cost/n_minibatches
            costs.append(n_minibatch_cost)
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, n_minibatch_cost))
        predictions = tf.equal(tf.argmax(Z3, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(predictions, 'float'))
        train_accuracy = accuracy.eval({x: x_train, y: y_train})
        test_accuracy = accuracy.eval({x: x_test, y: y_test})
    return train_accuracy, test_accuracy, parameters, costs


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, classes = load_dataset()
    # index = 3
    # plt.imshow(x_train[index])
    # print(y_train[index])

    # normalizing
    x_train = x_train/255
    x_test = x_test/255
    # one hot encoding
    y_train = np.eye(6)[y_train]
    y_test = np.eye(6)[y_test]
    # train and test
    train_accuracy, test_accuracy, parameters, costs = cnn_model(x_train, y_train, x_test, y_test)
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    print('Train Accuracy: ', train_accuracy)
    print('Test Accuracy: ', test_accuracy)
