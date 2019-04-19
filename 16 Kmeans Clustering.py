#K means clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the data set with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#Using the elbow method to find the optimal number of clusters
#max_iter=300, n_init=10 and init='k-means++' are default. 
from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS(Inertia)')
plt.show()

#Applying the k-means to data set with designated number of clusters
kmeans=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

#visualising the clusters (this code is applicable only for two dimensions.)
#if you ve more than 2 dimension don't use this code unless you reduce the number of dimensions
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1], s=100, c='red', label='Careful')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1], s=100, c='blue', label='Standard')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1], s=100, c='green', label='Target')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1], s=100, c='magenta', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income k$')
plt.ylabel('Spending Score 0-100')
plt.legend()
plt.show()






