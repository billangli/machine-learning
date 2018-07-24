"""
network.py
===============

Created by Bill Ang Li
Jul. 7th, 2018

Based on Michael Nielsen's Neural Networks and Deep Learning book
"""

import random

import numpy as np


class Network(object):
    """A 3-layered neural network"""

    def __init__(self, sizes: list):
        """
        Initializer that takes in a list of number of neurons in each layer

        randn(m, n) returns m x n matrix of random numbers from a normal distribution
        zip(a, b) returns list of tuples with values from a, b
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Return the output of the network if <a> is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data: list, num_epochs: int, mini_batch_size: int, eta: float, test_data=None):
        """
        Train the neural net using mini-batch stochastic gradient descent
        If <test_data> is supplied, then the program will evaluate the network after each epoch

        training_data is a list of tuples (x, y) with x as input and y as expected output
        """
        n = len(training_data)

        for j in range(num_epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                n_test = len(list(test_data))
                print("Epoch {}: {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch: list, eta: float):
        """
        Update the biases and weights according to the mini-batch
        """
        # Create a set of biases and weights that are zeros
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Update gradients for biases and weights
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update biases and weights
        self.biases = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
        Each of <nabla_b> and <nabla_w> are lists of NumPy arrays
            similar in structure to <self.biases> and <self.weights>
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feed forward
        activation = x
        activations = [x]
        zs = []  # Store the weighted inputs of each layer (z before activation function)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Back propagate (l index starts with last layer being 1)
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result.
        Note that the neural network's output is assumed to be the index of whichever neuron in the final layer has the
            highest activation.
        """
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x / \partial a for the output activations

        Based on the quadratic cost function C(x) = 1/(2n) * (y - a^L(x))^2
            where x is input
            y is expected output
            a^L is the actual output
        """
        return (output_activations-y)


def sigmoid(z):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    import network
    net = network.Network([784, 100, 10])
    net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
