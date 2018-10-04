from utils.data import get_data
import tensorflow as tf
import math
import numpy as np

class ConvNet:
    def __init__(self):
        self.model = None

    def initialize_filters(self):
        '''Initialize weights on filters
        Input data has shape: [X_n, n_H, n_W, n_C = 1]'''

        #First conv2D layer has 8 filters of shape [2, 2]
        W1 = tf.get_variable(name = 'W1', shape = [2, 2, 1, 8],
                             initializer = tf.contrib.layers.xavier_initializer())

        #Second conv2D layer has 16 filters of shape [2, 2]
        W2 = tf.get_variable(name='W2', shape=[2, 2, 8, 16],
                             initializer = tf.contrib.layers.xavier_initializer())

        return W1, W2

    def forward_prop(self, X, W1, W2, num_classes):
        '''Forward propagation. The model has two conv2D layers
        followed by two fully connected layers.
        '''

        #First conv2D layer with maxpooling
        Z1 = tf.nn.conv2d(input = X, filter = W1, strides = [1, 1, 1, 1], padding = 'SAME')
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.max_pool(value = A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

        #Second conv2D layer with maxpooling
        Z2 = tf.nn.conv2d(input = P1, filter = W2, strides = [1, 1, 1, 1], padding = 'SAME')
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.max_pool(value = A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

        #Flatten
        P2 = tf.contrib.layers.flatten(inputs = P2)

        #First fully-connected layer
        A3 = tf.contrib.layers.fully_connected(inputs = P2, num_outputs = 100, activation_fn=tf.nn.relu)

        #Second fully-connected layer
        A4 = tf.contrib.layers.fully_connected(inputs = A3, num_outputs=50, activation_fn=tf.nn.relu)

        #Output layer
        Z5 = tf.contrib.layers.fully_connected(inputs = A4, num_outputs = num_classes, activation_fn = None)

        return Z5

    def get_minibatches(self, X, y, batch_size):
        num_batches = math.ceil(X.shape[0] / batch_size)
        X_batches = np.array_split(X, num_batches, axis=0)
        y_batches = np.array_split(y, num_batches, axis=0)
        return [(X_batch, y_batch) for X_batch, y_batch in zip(X_batches, y_batches)]

    def fit_model(self, X_train, y_train, X_test, y_test, learning_rate = 0.001, num_epochs = 5, minibatch_size = 64):

        #Create placeholders
        n_samples, n_H0, n_W0, n_C0 = X_train.shape
        n_samples, n_y = y_train.shape
        X = tf.placeholder(dtype=tf.float32, shape = [None, n_H0, n_W0, n_C0])
        y = tf.placeholder(dtype=tf.float32, shape = [None, n_y])

        #Initialize filters
        W1, W2 = self.initialize_filters()

        #Forward propagation
        Z5 = self.forward_prop(X, W1, W2, n_y)

        #Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        #Start a tf session
        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(num_epochs):
                epoch_cost = 0
                minibatches = self.get_minibatches(X_train, y_train, minibatch_size)
                num_batches = len(minibatches)
                for minibatch in minibatches:
                    batch_X, batch_y = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:batch_X, y:batch_y})
                    epoch_cost += minibatch_cost
                epoch_cost /= num_batches
                print('Cost for epoch {}: {}'.format(epoch, epoch_cost))

            pred = tf.nn.softmax(Z5)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), 'float'))
            print(accuracy.eval({X: X_test, y: y_test}))


    def predict(self): pass


if __name__ == '__main__':
    #Get the data
    X_train, X_test, y_train, y_test = get_data()
    model = ConvNet()
    model.fit_model(X_train, y_train, X_test, y_test)


