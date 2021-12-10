import numpy as np
import random
import sklearn.cluster as cluster
import statistics as stat


data = np.array([[random.uniform(5, 90) for i in range(2)] for j in range(50)])
arr = np.array([random.uniform(0, 20) for i in range(10)])

print("Centers of clusters (Kmeans): \n" + str(cluster.KMeans(3).fit(data).cluster_centers_))

bandwidth = cluster.estimate_bandwidth(data)
print("Centers of cluster (Mean Shift): \n" + str(cluster.MeanShift(bandwidth=bandwidth).fit(data).cluster_centers_))

print("Median:" + str(stat.median(arr)))
print("Mean:" + str(stat.mean(arr)))
print("harmonic Mean:" + str(stat.harmonic_mean(arr)))
print("standart deviation:" + str(stat.stdev(arr)))
