#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import os
import numpy as np
from AnalysisTools.clustering_utils import *


# # Load Data

# In[2]:


file_path = 'D:\DETR\parquets\processed'
df_clustering = pd.read_parquet(file_path + '\\twice_a_day_5minMovingAverage_Clustering.parquet')


# In[3]:


df_clustering = sampling(df_clustering, 20*20) #sample every 20 secs


# In[4]:


#PCA
if 'twice_a_day_PCA.parquet' in os.listdir(file_path):
    df_PCA = pd.read_parquet(file_path + '\\twice_a_day_PCA.parquet')
else:
    df_PCA = df_clustering.copy()
    df_PCA = PCA_transform_with_statistics(df_PCA)
    df_PCA.to_parquet(file_path + '\\twice_a_day_PCA.parquet')


# In[5]:


#t-SNE
if 'twice_a_day_tsne.parquet' in os.listdir(file_path):
    df_tsne = pd.read_parquet(file_path + '\\twice_a_day_tsne.parquet')
else:
    df_tsne = df_clustering.copy()
    df_tsne = tsne_transform(df_tsne, 2)
    df_tsne.to_parquet(file_path + '\\twice_a_day_tsne.parquet')


# # K-means

# ## Without Dimensionality Reduction

# In[158]:


df_kmeans = df_clustering.copy()


# In[35]:


k_means_elbow_plot(df_kmeans)


# In[164]:


get_kmeans_statistics(df_kmeans, 3)


# ## PCA

# In[226]:


df_kmeans_PCA = df_PCA[["PC1", "PC2", "PC3", "PC4"]]


# In[227]:


k_means_elbow_plot(df_kmeans_PCA)


# In[228]:


get_kmeans_statistics(df_kmeans_PCA, 4)


# ## t-SNE

# In[167]:


df_kmeans_tsne = df_tsne.copy()


# In[41]:


k_means_elbow_plot(df_kmeans_tsne)


# In[179]:


get_kmeans_statistics(df_kmeans_tsne, 4)


# # DBScan

# ## Without Dimensionality Reduction

# In[64]:


df_dbscan = df_clustering.copy()


# In[71]:


k_list = [15, 30, 45]
plot_dbscan_k_distance(df_dbscan, k_list, 140)


# In[182]:


get_dbscan_statistics(df_dbscan, 80, 30)


# ## PCA

# In[109]:


df_dbscan_PCA = df_PCA[["PC1", "PC2", "PC3", "PC4"]]


# In[136]:


k_list = [15, 30, 45]
plot_dbscan_k_distance(df_dbscan_PCA, k_list)


# In[184]:


get_dbscan_statistics(df_dbscan_PCA, 0.28, 30)


# ## t-SNE

# In[6]:


df_dbscan_tsne = df_tsne.copy()


# In[110]:


k_list = [15, 30, 45]
plot_dbscan_k_distance(df_dbscan_tsne, k_list)


# In[15]:


get_dbscan_statistics(df_dbscan_tsne, 0.55, 30)


# # Gaussian Mixture Model

# ## Without Dimensionality Reduction

# In[23]:


df_gmm = df_clustering.copy()


# In[27]:


AIC_BIC_plot(df_gmm)


# In[32]:


get_gmm_statistics(df_gmm, n_components = 20)


# ## PCA

# In[16]:


df_gmm_PCA = df_PCA.copy()


# In[29]:


AIC_BIC_plot(df_gmm_PCA)


# In[8]:


result = get_gmm_statistics(df_gmm_PCA, n_components = 20)


# In[11]:


cluster_list = [9,14]
result[result["Cluster"].apply(lambda x: x in cluster_list)]


# ## t-SNE

# In[30]:


df_gmm_tsne = df_tsne.copy()


# In[31]:


AIC_BIC_plot(df_gmm_tsne)


# In[35]:


get_gmm_statistics(df_gmm_tsne, n_components = 18)


# # BIRCH

# ## Without Dimensionality Reduction

# In[27]:


df_birch = df_clustering.copy()


# In[213]:


birch_silhouette_score_plot(df_birch, np.linspace(0.1, 2.0, 20))


# In[214]:


#apparently in our case birch failed to detect any cluster and assigns each data point as its own cluster.
#So we try set 5 clusters
get_birch_statistics(df_birch, n_clusters = 5)


# In[28]:


#apparently in our case birch failed to detect any cluster and assigns each data point as its own cluster.
#So we try set 10 clusters
get_birch_statistics(df_birch, n_clusters = 10)


# ## PCA

# In[221]:


np.linspace(0.1, 1, 20)


# In[29]:


df_birch_PCA = df_PCA[["PC1", "PC2", "PC3", "PC4"]]


# In[18]:


birch_silhouette_score_plot(df_birch_PCA, np.linspace(0.1, 1, 20))


# In[24]:


#we choose the threshold value where the silhouette score peaks. In this case it is around 0.29
get_birch_statistics(df_birch_PCA, threshold = 0.29)


# In[25]:


#too many clusters. Try to set 5 clusters as well
get_birch_statistics(df_birch_PCA, n_clusters = 5)


# In[30]:


#too many clusters. Try to set 10 clusters as well
get_birch_statistics(df_birch_PCA, n_clusters = 10)


# ## t-SNE

# In[17]:


np.linspace(0.1, 1, 20)


# In[31]:


df_birch_tsne = df_tsne.copy()


# In[223]:


birch_silhouette_score_plot(df_birch_tsne, np.linspace(0.1, 1, 20))


# In[23]:


#we choose the threshold value where the silhouette score peaks. In this case it is around 0.48
get_birch_statistics(df_birch_tsne, threshold = 0.48)


# In[26]:


#too many clusters. Try to set 5 clusters as well
get_birch_statistics(df_birch_tsne, n_clusters = 5)


# In[32]:


#too many clusters. Try to set 10 clusters as well
get_birch_statistics(df_birch_tsne, n_clusters = 10)


# # OPTICS

# ## Without Dimensionality Reduction

# In[26]:


df_optics = df_clustering.copy()
get_optics_statistics(df_optics, 90, xi=0.01, min_cluster_size=0.003)


# ## PCA

# In[25]:


df_optics_PCA = df_PCA[["PC1", "PC2", "PC3", "PC4"]]
get_optics_statistics(df_optics_PCA, 60, xi=0.01, min_cluster_size=0.002)


# ## t-SNE

# In[37]:


df_optics_tsne = df_tsne.copy()
get_optics_statistics(df_optics_tsne, 120, xi=0.05, min_cluster_size=0.004)

