import numpy as np

class SVMClassifier:
    def __init__(self, learning_rate=0.1, max_iterations=200, regularization=0.5, num_folds=5):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.num_folds = num_folds

    def initialize_parameters(self, features):
        self.weights = np.zeros(features)
        self.bias = 0

    def train(self, data, labels):
        data = np.array(data)
        num_samples, num_features = data.shape
        self.initialize_parameters(num_features)

        for _ in range(self.max_iterations):
            for index, sample in enumerate(data):
                margin = labels[index] * (np.dot(sample, self.weights) + self.bias)
                if margin >= 1:
                    dw = 2 * self.regularization * self.weights
                    db = 0
                else:
                    dw = 2 * self.regularization * self.weights - labels[index] * sample
                    db = labels[index]

                self.weights -= self.learning_rate * dw
                self.bias += self.learning_rate * db

    def predict(self, data):
        return np.sign(np.dot(data, self.weights) + self.bias)

    def optimize_lambda(self, data, labels, lambdas):
        best_lambda = None
        highest_accuracy = 0

        for lam in lambdas:
            self.regularization = lam
            accuracies = []
            for fold in range(self.num_folds):
                validation_start = fold * (len(data) // self.num_folds)
                validation_end = (fold + 1) * (len(data) // self.num_folds)
                training_indices = np.concatenate([np.arange(0, validation_start), np.arange(validation_end, len(data))])
                validation_indices = np.arange(validation_start, validation_end)

                self.train(data[training_indices], labels[training_indices])
                predictions = self.predict(data[validation_indices])
                accuracies.append(np.mean(predictions == labels[validation_indices]))

            average_accuracy = np.mean(accuracies)
            if average_accuracy > highest_accuracy:
                highest_accuracy = average_accuracy
                best_lambda = lam

        self.regularization = best_lambda
        self.train(data, labels)

class MultiClassSVM:
    def __init__(self):
        self.classifiers = {}

    def train(self, data, labels):
        unique_classes = np.unique(labels)
        for i in range(len(unique_classes) - 1):
            for j in range(i + 1, len(unique_classes)):
                relevant_indices = (labels == unique_classes[i]) | (labels == unique_classes[j])
                relevant_data = data[relevant_indices]
                relevant_labels = labels[relevant_indices]
                binary_labels = np.where(relevant_labels == unique_classes[i], -1, 1)

                svm = SVMClassifier()
                svm.train(relevant_data, binary_labels)
                self.classifiers[(unique_classes[i], unique_classes[j])] = svm

    def predict(self, data):
        votes = np.zeros((len(data), len(self.classifiers)))
        for (class1, class2), classifier in self.classifiers.items():
            predictions = classifier.predict(data)
            votes[:, class1] += (predictions == -1)
            votes[:, class2] += (predictions == 1)

        return np.argmax(votes, axis=1)
