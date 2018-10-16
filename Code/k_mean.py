import numpy as np
from math import sqrt
import random

data = np.genfromtxt("cho.txt", delimiter='\t')
expect = data[:, 1]
genes = {gene: list(details) for gene, details in zip(data[:, 0], data[:, 2:])}
K = 5


def k_mean(K, data):
    centroids = [genes[random.randrange(len(data) * (i) // K, len(data) * (i + 1) // K) + 1] for i in range(K)]
    cluster, centroids_new = add_to_cluster(centroids)
    while 1:

        if centroids == centroids_new:

            return cluster
        else:
            centroids = centroids_new
            cluster, centroids_new = add_to_cluster(centroids)


def compute_new_centroids(o_cluster):
    centroids = []
    for i in o_cluster:
        temp = list(map(lambda x: x / len(o_cluster[i]), np.sum(o_cluster[i], axis=0)))
        centroids.append(temp)
    return centroids


def add_to_cluster(centroids):
    cluster = {i: [] for i in range(K)}
    for gene in genes:
        minimum = (0, 1000000)
        for i in range(len(centroids)):
            dist = euclidean(genes[gene], centroids[i])
            if dist <= minimum[1]:
                minimum = (i, dist)
        cluster[minimum[0]].append(genes[gene])

    centroids_new = compute_new_centroids(cluster)
    return cluster, centroids_new


def euclidean(point1, point2):
    dist = 0
    for p1, p2 in zip(point1, point2):
        dist += ((p1 - p2) ** 2)
        # print(dist)
    dist = sqrt(dist)

    return dist


k_mean(K, genes)
