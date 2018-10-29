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
min_supp = 3
ep = 1


def k_mean(K, num_iter=0, center=[]):
    if center == []:
        suffle_data = data.copy()
        np.random.shuffle(suffle_data)

        cluster = {i: suffle_data[len(data) * i // K:len(data) * (i + 1) // K, 2:] for i in range(K)}
        centroids = compute_new_centroids(cluster)
    else:
        centroids = [genes[c] for c in center]

    labels, cluster, centroids_new = add_to_cluster(centroids)

    while 1:
        if centroids == centroids_new:
            print(np.asarray(labels, dtype=float))
            print(expect)
            return cluster, labels
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


def density_cluster(min_supp, ep):
    distance_map = {key: [] for key in genes}
    genes_label = {key: "OUT" for key in genes}
    checked = {key: 0 for key in genes}
    cluster_dict = {}
    outlier = []
    for gene in genes:
        for gene1 in genes:
            if gene != gene1:
                dist = euclidean(genes[gene], genes[gene1])
                if dist <= ep:
                    distance_map[gene].append((gene1, dist))

        if len(distance_map[gene]) >= min_supp:
            genes_label[gene] = "CORE"
            for sub_gene in distance_map[gene]:
                if genes_label[sub_gene[0]] == "OUT":
                    genes_label[sub_gene[0]] = "BORDER"
    # print(genes_label)

    for gene in genes_label:
        if genes_label[gene] == "CORE" and checked[gene] != 1:
            queue = []
            cluster = [int(gene) - 1]
            checked[gene] = 1
            for item in distance_map[gene]:
                queue.append(item[0])
            while len(queue) != 0:
                sub_gene = queue.pop()
                if checked[sub_gene] != 1:
                    cluster.append(int(sub_gene) - 1)
                    checked[sub_gene] = 1
                    if genes_label[sub_gene] == "CORE":
                        for item in distance_map[sub_gene]:
                            if checked[item[0]] != 1:
                                queue.append(item[0])
            cluster_dict[len(cluster_dict)] = cluster
        elif genes_label[gene] == "OUT":
            outlier.append(int(gene) - 1)
    # print(cluster_dict)
    return cluster_dict, genes_label, outlier


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


cluster, density_labels, out = density_cluster(min_supp, ep)
c, target = hier_agg_cluster(K)
k_mean_cluster, labels = k_mean(K, 0, [])
testData = data[:, 2:]
pca = PCA(n_components=len(testData[0]))
d = pca.fit_transform(testData)
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
colorMap = {}
ax[0].set_title("Hierarchical Agglomerative clustering")
ax[1].set_title("K-mean clustering")
ax[2].set_title("density clustering")
for i in target:
    if target[i] in colorMap.keys():
        ax[0].plot(d[i - 1, 0], d[i - 1, 1], colorMap[target[i]], markersize=3)
    else:
        colorMap[target[i]] = color[len(colorMap)]
        ax[0].plot(d[i - 1, 0], d[i - 1, 1], colorMap[target[i]], markersize=3)
for i in range(len(labels)):
    ax[1].plot(d[i, 0], d[i, 1], color[labels[i] - 1], markersize=3)
c = 0
legends_label = []
for i in cluster:
    ax[2].plot(d[cluster[i], 0], d[cluster[i], 1], color[i], markersize=3)
    c = i
    legends_label.append("cluster {}".format(i + 1))

ax[2].plot(d[out, 0], d[out, 1], color[c + 1], markersize=3)
legends_label.append("outlier")
ax[2].legend(legends_label,
             loc='upper right')
plt.savefig("hier.jpg")
plt.show()
