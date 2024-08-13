import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Generating a sample dataset
np.random.seed(0)
X_train = np.random.rand(50, 2) * 100  # 50 points with 2 features
y_train = np.random.choice([0, 1], 50)  # Binary labels

# Create and train the k-NN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predicting a new data point
X_new = np.array([[60, 30]])  # New data point with 2 features
y_pred = knn.predict(X_new)

# Plotting the training data
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=50, edgecolors='k')
plt.scatter(X_new[:, 0], X_new[:, 1], c='green', marker='*', s=200, label='New Data Point')

# Plotting the neighbors
distances, indices = knn.kneighbors(X_new)
neighbors = X_train[indices[0]]
plt.scatter(neighbors[:, 0], neighbors[:, 1], c='yellow', s=100, edgecolors='k', label='Neighbors')

# Annotating the plot
for i, neighbor in enumerate(neighbors):
    plt.annotate(f'{i+1}', (neighbor[0], neighbor[1]), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'k-NN Classification (k={k})\nPredicted Class: {y_pred[0]}')
plt.show()
