import math
import random
import time

import numpy as np
import scipy
from .matrix import Matrix

from .supervised_learner import SupervisedLearner


class PerceptronLearner(SupervisedLearner):
    def __init__(self):
        self.final_labels = []


    def train(self, features, labels):
        solution_found = False
        n_epochs_without_improvement = 0
        # instantiate learning rate, weight vector, and mse
        lr = .05
        self.w = np.random.random_sample((1, features.cols + 1))
        best_mse = math.inf
        best_weights = np.zeros(shape=(1, features.cols + 1))
        best_accuracy = 0

        first_epoch = True
        # epochs
        while n_epochs_without_improvement < 50 and not solution_found:
            for i, row in enumerate(features.data):
                # add bias input to row and convert to numpy array
                if first_epoch:
                    row.append(1)
                row = np.array(row)

                # get target
                target = labels.data[i][0]

                # get net and output
                net, output = self.compute_net_and_get_output(row, self.w)
                # compute point error
                e = target - output
                # update weights
                w_update_vector = lr * e * row
                self.w += w_update_vector

            # Track if accuracy has improved
            accuracy = self.measure_accuracy(features, labels)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = np.copy(self.w)
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            # If there is no error, a solution has been found. Exit the loop.
            if accuracy == 1:
                solution_found = True

            first_epoch = False

        self.w = best_weights
        self.mse = best_mse
        return self.w

    def compute_net_and_get_output(self, row, w):
        # compute net
        net = np.dot(w, row)
        # compute output
        if net > 0:
            output = 1
        else:
            output = 0
        return net, output

    def predict(self, features, labels):
        del labels[:]

        features = np.array(features)

        # compute net and output prediction
        net, output = self.compute_net_and_get_output(features, self.w)
        labels.append(output)
        self.final_labels.append(output)


class MultiClassPerceptron(PerceptronLearner):
    def __init__(self, weights):
        self.weights_dict = weights
        self.final_labels = []

    def train(self, features, labels):
        self.weights_dict.append(super().train(features, labels))

    def predict(self, features, labels):
        # if training is done
        del labels[:]

        best_net = -math.inf
        best_output = 0
        output_class = ""
        features.append(1)
        features = np.array(features)
        for key, value in self.weights_dict.items():

            net, output = self.compute_net_and_get_output(features, value)

            # if abs(output) >= abs(best_output):
            if output >= best_output:
                if net >= best_net:
                    best_net = net
                    best_output = output
                    output_class = key

        self.final_labels.append(output_class)
        labels.append(output_class)
