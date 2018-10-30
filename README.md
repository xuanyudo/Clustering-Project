# Clustering-Project
This repo includes the implementations of three clustering algorithms to find clusters of genes that exhibit similar expression profiles: K-means, Hierarchical Agglomerative clustering with Single Link (Min), and one from (density-based, mixture model, spectral).

# infomation for tester:
<h2>Parmeters to adjust (in the top of the code)</h2>
<ul>
  <li> <b>file </b>: filename, default = "cho" </li>
  <li><b>K</b>: numbers of cluster for k-mean and hierarchical clustering, default = 5 </li>
  <li><b>min_supp</b>: minmum numbers of node need to be explored by core, default = 4
 </li>
  <li><b>ep</b>: the exploration range for each node, default = 1
 </li>
  <li><b>num_iter</b>: number of iteration for finding best k mean clustering result, default = 10
 </li>
  <li><b>max_iter</b>: number of iteration for compute centroids, if the cluster not stable until max_iter, the program will just take current clustering result as our k-mean result, default = 1000
 </li>
  <li><b>init_center</b>: set initial clustering center, defalut = []</li></ul>
  
<h2> adjust above parameters and run k_mean.py with any python IDE </h2>

# Information for running Hadoop Kmeans:

We summarized all necessary bash commands into the KM_try.sh file.

To reproduce the results in our report, please get to the KM_Hadoop_code directory, and run:

(1) ./KM_try.sh cho.txt centroids_cho.txt KM_try18.jar

(2) ./KM_try.sh iyer.txt centroids_iyer.txt KM_try18.jar

To reproduce the results we show during the demo, please run:

./KM_try.sh new_dataset_1.txt centroids_new2.txt KM_try18.jar

# Testing Output

![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/all_threecho.jpg)
![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/all_threeiyer.jpg)

<h2> <b>Details:</b> </h2>

![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/k_meancho.jpg)
![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/k_meaniyer.jpg)
![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/hiercho.jpg)
![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/hieriyer.jpg)
![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/densitycho.jpg)
![alt text](https://github.com/xuanyudo/Clustering-Project/blob/master/Report/densityiyer.jpg)
