# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('LabelEncoded.csv')
d = pd.read_csv('Total.csv')
x = data.iloc[:, 1:].values
city = d.iloc[:, 1:3].values
X = d.iloc[:,3:-4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])
X[:, 1] = le.fit_transform(X[:, 1])
x[:, 21] = le.fit_transform(x[:, 21])
oe = OneHotEncoder(categorical_features = [0, 21])
oe1 = OneHotEncoder(categorical_features = [1])
x = oe.fit_transform(x).toarray()
X = oe1.fit_transform(X).toarray()

X[356,2] = 0

from sklearn.decomposition import LatentDirichletAllocation
lda1 = LatentDirichletAllocation(n_topics=20, random_state=1)
lda1.fit(X)

t = lda1.transform(X)

temp = np.append(city, t, 1)
tdata = pd.DataFrame(temp)

from sklearn.cluster import KMeans
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(t)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss)

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(t)

