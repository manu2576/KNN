import numpy as np
from collections import Counter

def ecd_distance(point1, point2):
    # Compute the Euclidean distance between two points.
    return np.sqrt(np.sum((point1 - point2)**2))

class KNN():
    def __init__(self, k=3):
        # Initialize with number of neighbors.
        self.k = k

    def fit(self, X, y):

        # Fit the model with training data.
        self.x_train = X
        self.y_train = y

    def predict(self, X):

        # Predict labels for the data.
        return np.array([self.knn_predict(x) for x in X])

    def knn_predict(self, predict_x):

        # Predict label for a single data point.
        distance = [ecd_distance(predict_x, x_train) for x_train in self.x_train]
        k_nearest_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


