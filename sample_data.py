import numpy as np 
import pandas as pd
from numpy.random import RandomState
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cross_validation import train_test_split

nsize = 100
random_state = RandomState(123)
X, y = make_blobs(n_samples = nsize, centers = 2, n_features = 2, random_state = random_state)
X = pd.DataFrame(X, columns = ['x', 'y'])
X['z'] = np.random.normal(5, 5, nsize)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_state)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c = y_train, marker='o')
ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')

plt.show()