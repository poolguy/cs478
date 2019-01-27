import random
import time

from toolkit.matrix import Matrix
from toolkit.perceptron_learner import PerceptronLearner


class PerceptronTest:

    def main(self):
        learner = PerceptronLearner()
        learner_name = "Perceptron"
        file_name = "../datasets/voting.arff"
        # parse the command-line arguments
        # Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> | cross <num_folds>)
        eval_method = "training"
        eval_parameter = None
        # boolean: Print the confusion matrix and learner accuracy on individual class values
        print_confusion_matrix = False
        # boolean: Use normalized data
        normalize = False
        # string: Random seed
        random.seed(12)


        # load the ARFF file
        data = Matrix()
        data.load_arff(file_name)
        if normalize:
            print("Using normalized data")
            data.normalize()

        # print some stats
        print("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(file_name, data.rows, data.cols, learner_name, eval_method))

        if eval_method == "training":

            print("Calculating accuracy on training set...")

            features = Matrix(data, 0, 0, data.rows, data.cols - 1)
            labels = Matrix(data, 0, data.cols - 1, data.rows, 1)
            confusion = Matrix()
            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))
            accuracy = learner.measure_accuracy(features, labels, confusion)
            print("Training set accuracy: " + str(accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "static":

            print("Calculating accuracy on separate test set...")

            test_data = Matrix(arff=eval_parameter)
            if normalize:
                test_data.normalize()

            print("Test set name: {}".format(eval_parameter))
            print("Number of test instances: {}".format(test_data.rows))
            features = Matrix(data, 0, 0, data.rows, data.cols - 1)
            labels = Matrix(data, 0, data.cols - 1, data.rows, 1)

            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(features, labels)
            print("Training set accuracy: {}".format(train_accuracy))

            test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols - 1)
            test_labels = Matrix(test_data, 0, test_data.cols - 1, test_data.rows, 1)
            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "random":

            print("Calculating accuracy on a random hold-out set...")
            train_percent = float(eval_parameter)
            if train_percent < 0 or train_percent > 1:
                raise Exception("Percentage for random evaluation must be between 0 and 1")
            print("Percentage used for training: {}".format(train_percent))
            print("Percentage used for testing: {}".format(1 - train_percent))

            data.shuffle()

            train_size = int(train_percent * data.rows)
            train_features = Matrix(data, 0, 0, train_size, data.cols - 1)
            train_labels = Matrix(data, 0, data.cols - 1, train_size, 1)

            test_features = Matrix(data, train_size, 0, data.rows - train_size, data.cols - 1)
            test_labels = Matrix(data, train_size, data.cols - 1, data.rows - train_size, 1)

            start_time = time.time()
            learner.train(train_features, train_labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(train_features, train_labels)
            print("Training set accuracy: {}".format(train_accuracy))

            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "cross":

            print("Calculating accuracy using cross-validation...")

            folds = int(eval_parameter)
            if folds <= 0:
                raise Exception("Number of folds must be greater than 0")
            print("Number of folds: {}".format(folds))
            reps = 1
            sum_accuracy = 0.0
            elapsed_time = 0.0
            for j in range(reps):
                data.shuffle()
                for i in range(folds):
                    begin = int(i * data.rows / folds)
                    end = int((i + 1) * data.rows / folds)

                    train_features = Matrix(data, 0, 0, begin, data.cols - 1)
                    train_labels = Matrix(data, 0, data.cols - 1, begin, 1)

                    test_features = Matrix(data, begin, 0, end - begin, data.cols - 1)
                    test_labels = Matrix(data, begin, data.cols - 1, end - begin, 1)

                    train_features.add(data, end, 0, data.cols - 1)
                    train_labels.add(data, end, data.cols - 1, 1)

                    start_time = time.time()
                    learner.train(train_features, train_labels)
                    elapsed_time += time.time() - start_time

                    accuracy = learner.measure_accuracy(test_features, test_labels)
                    sum_accuracy += accuracy
                    print("Rep={}, Fold={}, Accuracy={}".format(j, i, accuracy))

            elapsed_time /= (reps * folds)
            print("Average time to train (in seconds): {}".format(elapsed_time))
            print("Mean accuracy={}".format(sum_accuracy / (reps * folds)))

        else:
            raise Exception("Unrecognized evaluation method '{}'".format(eval_method))


if __name__ == '__main__':
    PerceptronTest().main()
