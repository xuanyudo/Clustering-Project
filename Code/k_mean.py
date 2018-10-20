import numpy as np
from math import sqrt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

data = np.genfromtxt("cho.txt", delimiter='\t')
expect = data[:, 1]
genes = {gene: list(details) for gene, details in zip(data[:, 0], data[:, 2:])}
K = 5
color = ['bo', 'ro', 'go', 'co', 'mo', 'yo', 'ko']
colorMap = {}


def k_mean(K, genes):
    cluster = {i: data[len(data) * i // K:len(data) * (i + 1) // K, 2:] for i in range(K)}
    centroids = compute_new_centroids(cluster)

    labels, cluster, centroids_new = add_to_cluster(centroids)

    while 1:
        if centroids == centroids_new:
            print(np.asarray(labels, dtype=float))
            print(expect)
            return cluster
        else:
            centroids = centroids_new
            labels, cluster, centroids_new = add_to_cluster(centroids)


def hier_agg_cluster(K):
    genes_cluster = {key: [key] for key in range(1, len(genes) + 1)}
    genes_target = {key: key for key in range(1, len(genes) + 1)}
    distance = []
    for gene in genes:
        for gene1 in genes:
            if gene != gene1:
                dist = euclidean(genes[gene], genes[gene1])
                distance.append((gene, (gene1, dist)))

    distance.sort(key=lambda kv: kv[1][1])

    for gene in distance:

        if genes_target[gene[0]] != genes_target[gene[1][0]]:
            genes_cluster[genes_target[gene[0]]].extend(genes_cluster[genes_target[gene[1][0]]])
            deleted = genes_cluster.pop(genes_target[gene[1][0]], None)

            for key in deleted:
                genes_target[key] = genes_target[gene[0]]
        # print(genes_cluster[genes_target[gene[0]]])
        if len(genes_cluster) == K:
            # print(genes_cluster)
            print(genes_cluster)
            return genes_cluster, genes_target


def compute_new_centroids(o_cluster):
    centroids = []

    for i in o_cluster:
        mss = np.sum(o_cluster[i], axis=0)
        temp = list(map(lambda x: x / len(o_cluster[i]), mss))
        centroids.append(temp)
    return centroids


def add_to_cluster(centroids):
    cluster = {i + 1: [] for i in range(K)}
    labels = []
    for gene in genes:
        minimum = (0, 1000000)
        for i in range(len(centroids)):
            dist = euclidean(genes[gene], centroids[i])
            if dist <= minimum[1]:
                minimum = (i + 1, dist)
        cluster[minimum[0]].append(genes[gene])
        labels.append(minimum[0])

    centroids_new = compute_new_centroids(cluster)
    return labels, cluster, centroids_new


def euclidean(point1, point2):
    dist = 0

    for p1, p2 in zip(point1, point2):
        dist += ((p1 - p2) ** 2)

        # print(dist)
    dist = sqrt(dist)

    return dist


c, target = hier_agg_cluster(K)
testData = data[:, 2:]
pca = PCA(n_components=len(testData[0]))
d = pca.fit_transform(testData)

colorMap = {}
for i in target:
    if target[i] in colorMap.keys():
        plt.plot(d[i-1,0], d[i-1,1], colorMap[target[i]])
    else:
        colorMap[target[i]] = color[len(colorMap)]
        plt.plot(d[i-1, 0], d[i-1, 1], colorMap[target[i]])
plt.savefig("hier.jpg")
plt.show()
# k_mean(K, genes)
