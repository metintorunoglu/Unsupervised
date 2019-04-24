#HR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the data set with pandas
dataset = pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values

#using dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrograms')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#fitting HC to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5)
y_hc=hc.fit_predict(X)

#Vısualising the clusters(Not appropriate for more than two dimensions)
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red', label='Careful')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='blue', label='Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='green', label='Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c='magenta', label='Sensible')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income k$')
plt.ylabel('Spending Score 0-100')
plt.legend()
plt.show()
