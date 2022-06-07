import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import special
from matplotlib import pyplot as plt


X = np.genfromtxt('data.csv', delimiter=',')


def plofs(X, k, _lambda):
  knn = NearestNeighbors(n_neighbors=k+1)
  knn.fit(X)

  p_dists = np.zeros(X.shape[0])
  plofs_arr = np.zeros(X.shape[0])

  for i in range(X.shape[0]):
    neighbor_indices = knn.kneighbors([X[i]], return_distance=False)
    standard_distance = np.sqrt(np.sum(np.linalg.norm(X[neighbor_indices] - X[i], axis=1) ** 2) / k)
    p_dists[i] = _lambda * standard_distance

  for i in range(X.shape[0]):
    neighbor_indices = knn.kneighbors([X[i]], return_distance=False)
    plofs_arr[i] = (p_dists[i] / np.mean(p_dists[neighbor_indices])) - 1

  return plofs_arr


def loop(X, k=20, _lambda=3):
  plofs_arr = plofs(X, k, _lambda)
  n_plof = _lambda * np.sqrt(np.mean(plofs_arr ** 2))
  loop_vec = np.maximum(0, special.erf(plofs_arr / (n_plof * np.sqrt(2))))

  return loop_vec


y = loop(X, k=20*10)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Blues')
plt.show()
