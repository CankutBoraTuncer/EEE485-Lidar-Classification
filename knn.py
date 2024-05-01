import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1- x2)**2))


class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):  # We pass through the testing dataset.
        predictions = [self._predict(x) for x in X] # Helper function for each examples in the dataset
        return predictions

    def _predict(self, x):
        distances = []
        for idx, x_train in enumerate(self.X_train):
            if np.array_equal(x, x_train):
                continue  
            distance = euclidean_distance(x, x_train)
            distances.append((distance, idx))

        distances.sort()
        k_indices = [idx for _, idx in distances[:self.k]]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


    

