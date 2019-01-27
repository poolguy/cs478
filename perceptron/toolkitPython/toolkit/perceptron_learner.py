import math
import random
import time

import numpy as np
import scipy
from .matrix import Matrix

from .supervised_learner import SupervisedLearner


class PerceptronLearner(SupervisedLearner):

    def train(self, features, labels):
        solution_found = False
        n_epochs_without_improvement = 0
        # instantiate learning rate, weight vector, and mse
        lr = .05
        w = np.zeros(shape=(1, features.cols + 1))
        best_mse = math.inf
        best_weights = np.zeros(shape=(1, features.cols + 1))


        # todo: build in a bfsf to store weights with associated lowest mse

        first_epoch = True
        while n_epochs_without_improvement < 5 and not solution_found:
            sse = 0
            for i, row in enumerate(features.data):
                # add bias input to row and convert to numpy array
                if first_epoch:
                    row.append(1)
                row = np.array(row)

                # get target
                target = labels.data[i][0]

                # compute net
                net = np.dot(w, row)
                # compute output
                if net > 0:
                    output = 1
                else:
                    output = 0
                # compute point error
                e = target - output
                # update weights
                w_update_vector = lr * e * row
                w += w_update_vector

                # update sum squared error
                sse += e**2

            # compute mean squared error
            mse = sse / features.rows
            # Track if error has changed
            if mse < best_mse:
                best_mse = mse
                best_weights = np.copy(w)
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            # If there is no error, a solution has been found. Exit the loop.
            if mse == 0:
                solution_found = True

            first_epoch = False

        print(best_weights)
        print(best_mse)





    def predict(self, features, labels):
        del labels[:]
        labels += self.labels
        pass


