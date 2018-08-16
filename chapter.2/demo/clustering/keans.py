
print(__doc__)

import numpy as np

from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.datasets.samples_generator import make_moons
from sklearn.preprocessing import StandardScaler

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_moons(n_samples=1000, noise=0.1)

X = StandardScaler().fit_transform(X)

# #############################################################################
n_clusters_ = 2
kmean = KMeans(n_clusters=n_clusters_)
kmean.fit_predict(X)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

labels = kmean.labels_
centers = kmean.cluster_centers_
colors = ['r', 'b', 'y']

plt.figure(figsize=(6, 4), dpi=144)
plt.xticks(())
plt.yticks(())

for c in range(n_clusters_):
    cluster = X[labels == c]
    plt.scatter(cluster[:,0], cluster[:,1], marker='o', s=20, c=colors[c])

plt.scatter(centers[:,0], centers[:,1], marker='o', c="white", alpha=0.9, s=300)
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker='$%d$' % i, s=50, c=colors[i])

plt.show()
