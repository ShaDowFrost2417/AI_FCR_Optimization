#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D #for 3D plotting
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AffinityPropagation


# # Functions

# ## Data Processing

# In[2]:


def sampling(df, interval_frames):
    # Define the size of the DataFrame and the desired sample size
    N = len(df)
    k = interval_frames
    
    # Generate a random starting point between 0 and k-1
    start = np.random.randint(0, k)

    # Create an array of indices starting from 'start' to 'N' with a step of 'k'
    indices = np.arange(start, N, k)

    # Use the indices to sample the DataFrame
    df_sampled = df.iloc[indices]


    return df_sampled


# In[3]:


def PCA_transform_with_statistics(df_PCA):
    scaled_data_PCA = preprocessing.scale(df_PCA)
    pca = PCA()
    pca.fit(scaled_data_PCA)
    pca_data = pca.transform(scaled_data_PCA)
    
    
    percentage_variation = np.round(pca.explained_variance_ratio_* 100, decimals = 1)
    labels = ['PC' + str(x) for x in range(1, len(percentage_variation)+1)]
    
    plt.bar(x=range(1,len(percentage_variation)+1), height = percentage_variation, tick_label = labels)
    plt.ylabel('Percentage of Expained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    # Using plt.plot to create a line plot
    plt.plot(range(1, len(percentage_variation) + 1), percentage_variation, marker='o', linestyle='-', color='b')

    # Adding labels to each point for clarity
    for i, txt in enumerate(percentage_variation):
        plt.annotate(txt, (i + 1, percentage_variation[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xticks(range(1, len(percentage_variation) + 1), labels)  # Ensuring we have all principal component labels
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot - Line Plot Version')
    plt.grid(True)  # Adding grid for better readability
    plt.show()
    
    pca_result_df = pd.DataFrame(pca_data, index = df_PCA.index, columns = labels)
    
    return pca_result_df


# In[4]:


def plot_3D_PCA(pca_result_df):
    # Create a 3D subplot
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')

    # Extract PC1, PC2, and PC3 values
    x_values = pca_result_df['PC1']
    y_values = pca_result_df['PC2']
    z_values = pca_result_df['PC3']

    # Scatter plot for PC1, PC2, PC3
    ax.scatter(x_values, y_values, z_values)

    # Set titles and labels
    ax.set_title('3D PCA Graph')
    ax.set_xlabel(f"PC1 - {percentage_variation[0]}%")
    ax.set_ylabel(f"PC2 - {percentage_variation[1]}%")
    ax.set_zlabel(f"PC3 - {percentage_variation[2]}%")  # Assuming percentage_variation[2] exists for PC3

    # Annotate each point with its index (e.g., sample identifier)
    # This part is slightly more complex in 3D and might clutter the plot. Consider carefully if needed.
    # for sample in pca_result_df.index:
    #     ax.text(pca_result_df.loc[sample, 'PC1'], pca_result_df.loc[sample, 'PC2'], pca_result_df.loc[sample, 'PC3'],
    #             sample, textcoords="offset points", xytext=(0,10), zorder=1)

    plt.show()


# In[5]:


def tsne_transform(df_tsne, resulting_dimension):
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_tsne)

    # Initialize t-SNE and reduce dimensionality
    tsne = TSNE(n_components=resulting_dimension, random_state=42) #random state ensures repeatability of the result. Value is arbitrary
    data_tsne = tsne.fit_transform(data_scaled)

    # Convert to DataFrame for easier plotting and manipulation
    tsne_result_df = pd.DataFrame(data_tsne, columns=['TSNE1', 'TSNE2'], index = df_tsne.index)
    
    return tsne_result_df


# ## K-Means

# In[6]:


def k_means_elbow_plot(df):
    inertias = []
    range_of_clusters = range(1, 11)  # Trying 1 to 10 clusters

    for k in range_of_clusters:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        inertias.append(kmeans.inertia_)

    plt.plot(range_of_clusters, inertias, '-o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


# In[7]:


def get_kmeans_statistics(df_kmeans, cluster_count):
    # KMeans clustering, density based
    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(df_kmeans)
    clusters = kmeans.predict(df_kmeans)

    # Adding the cluster IDs to the PCA result DataFrame
    df_kmeans_temp = df_kmeans.copy()
    df_kmeans_temp['Cluster'] = clusters
    df_kmeans_temp['TimeNumeric'] = df_kmeans_temp.index.astype(np.float64)/60  # Convert to hours

    # Mean, median, std of numeric time for each cluster
    cluster_stats = df_kmeans_temp.groupby('Cluster')['TimeNumeric'].agg(['mean', 'median', 'std'])
    cluster_stats_sorted = cluster_stats.sort_values(by='median', ascending=True)

    print(cluster_stats_sorted)
    
    if 'PC1' in df_kmeans_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_kmeans_temp['PC1'], df_kmeans_temp['PC2'], df_kmeans_temp['PC3'], c=df_kmeans_temp['Cluster'], cmap='viridis')
        ax.set_title('PCA K-Means Clustering Result')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster ID")
        ax.add_artist(legend1)
        plt.show()
    elif 'TSNE1' in df_kmeans_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        scatter = plt.scatter(df_kmeans_temp['TSNE1'], df_kmeans_temp['TSNE2'], c=df_kmeans_temp['Cluster'], cmap='viridis')
        plt.title('TSNE K-Means Clustering Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="Cluster ID")
        plt.show()


# ## DBScan

# In[8]:


def plot_dbscan_k_distance(df_dbscan, k_list, y_lim = 99999999):
    for k in k_list:
        # Step 1: Calculate the distance to the nearest k points
        nearest_neighbors = NearestNeighbors(n_neighbors=k)
        nearest_neighbors.fit(df_dbscan)
        distances, indices = nearest_neighbors.kneighbors(df_dbscan)

        # Step 2: Sort the distances
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]  # Take the distance to the k-th nearest neighbor (excluding the point itself)

        # Step 3: Plot the k-distance graph
        plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
        plt.plot(distances)
        plt.xlabel('Points sorted by distance to the k-th nearest neighbor')
        plt.ylabel(f'Distance to {k}-th nearest neighbor')
        plt.title(f'k-Distance Graph (k={k})')

        # Setting y-axis ticks interval
        y_max = np.max(distances)  # Get the max distance to dynamically adjust the y-ticks
        plt.yticks(np.arange(0, y_max, y_max/20))  # Adjust this to change the interval
        
        if y_lim != 99999999:
            plt.ylim(top = y_lim)
        else:
            plt.ylim(top = np.max(distances))

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enhance grid visibility
        plt.show()


# In[9]:


def get_dbscan_statistics(df_dbscan, eps, min_samples):
    # DBSCAN, density based
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_dbscan)

    df_dbscan_temp = df_dbscan.copy()
    df_dbscan_temp['Cluster'] = clusters
    df_dbscan_temp['TimeNumeric'] = df_dbscan_temp.index.astype(np.float64)/60  # Convert to hours

    # Mean, median, std of numeric time for each cluster
    cluster_stats = df_dbscan_temp.groupby('Cluster')['TimeNumeric'].agg(['mean', 'median', 'std'])
    cluster_stats_sorted = cluster_stats.sort_values(by='median', ascending=True)

    print(df_dbscan_temp['Cluster'].value_counts())
    print(cluster_stats_sorted) #in hrs
    
    if 'PC1' in df_dbscan_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_dbscan_temp['PC1'], df_dbscan_temp['PC2'], df_dbscan_temp['PC3'], c=df_dbscan_temp['Cluster'], cmap='viridis')
        ax.set_title('PCA DBScan Clustering Result')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster ID")
        ax.add_artist(legend1)
        plt.show()
    elif 'TSNE1' in df_dbscan_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        scatter = plt.scatter(df_dbscan_temp['TSNE1'], df_dbscan_temp['TSNE2'], c=df_dbscan_temp['Cluster'], cmap='viridis')
        plt.title('TSNE DBScan Clustering Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="Cluster ID")
        plt.show()


# ## Birch

# In[10]:


def birch_silhouette_score_plot(df_birch, thresholds):
    # Initialize a list to store silhouette scores
    silhouette_scores = []

    # Loop over the range of threshold values
    for t in thresholds:
        # Initialize the BIRCH model with the current threshold
        birch = Birch(threshold=t, n_clusters=None)
        # Fit the model to the data
        birch.fit(df_birch)
        labels = birch.predict(df_birch)

        # Calculate the silhouette score and append it to the list
        # Note: silhouette_score requires at least 2 labels to compute, hence we check the number of unique labels
        if len(np.unique(labels)) > 1:
            score = silhouette_score(df_birch, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1) # Assigning a low score if there's only one cluster

    # Plotting the elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, silhouette_scores, marker='o')
    plt.title('Elbow Plot for Selecting BIRCH Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()


# In[11]:


def get_birch_statistics(df_birch, threshold = 0.5, branching_factor=50, n_clusters=None):
    #BIRCH, Hierarchical
    birch = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=n_clusters)
    clusters = birch.fit_predict(df_birch)

    df_birch_temp = df_birch.copy()
    df_birch_temp['Cluster'] = clusters
    df_birch_temp['TimeNumeric'] = df_birch_temp.index.astype(np.float64) / 60  # Assuming index is in minutes, convert to hours

    # Mean, median, std of numeric time for each cluster
    cluster_stats = df_birch_temp.groupby('Cluster')['TimeNumeric'].agg(['mean', 'median', 'std'])
    cluster_stats_sorted = cluster_stats.sort_values(by='median', ascending=True)

    print(df_birch_temp['Cluster'].value_counts())
    print(cluster_stats_sorted)  # in hrs
    
    if 'PC1' in df_birch_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_birch_temp['PC1'], df_birch_temp['PC2'], df_birch_temp['PC3'], c=df_birch_temp['Cluster'], cmap='viridis')
        ax.set_title('PCA Birch Clustering Result')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster ID")
        ax.add_artist(legend1)
        plt.show()
    elif 'TSNE1' in df_birch_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        scatter = plt.scatter(df_birch_temp['TSNE1'], df_birch_temp['TSNE2'], c=df_birch_temp['Cluster'], cmap='viridis')
        plt.title('TSNE Birch Clustering Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="Cluster ID")
        plt.show()


# ## OPTICS

# In[12]:


def get_optics_statistics(df, min_samples, xi=0.05, min_cluster_size=0.05):
    # OPTICS, density based
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    clusters = optics.fit_predict(df)

    df_temp = df.copy()
    df_temp['Cluster'] = clusters
    df_temp['TimeNumeric'] = df_temp.index.astype(np.float64)/60  # Convert to hours

    # Mean, median, std of numeric time for each cluster
    cluster_stats = df_temp.groupby('Cluster')['TimeNumeric'].agg(['mean', 'median', 'std'])
    cluster_stats_sorted = cluster_stats.sort_values(by='median', ascending=True)

    print(df_temp['Cluster'].value_counts())
    print(cluster_stats_sorted)  # in hrs

    if 'PC1' in df_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_temp['PC1'], df_temp['PC2'], df_temp['PC3'], c=df_temp['Cluster'], cmap='viridis')
        ax.set_title('PCA OPTICS Clustering Result')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster ID")
        ax.add_artist(legend1)
        plt.show()
    elif 'TSNE1' in df_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        scatter = plt.scatter(df_temp['TSNE1'], df_temp['TSNE2'], c=df_temp['Cluster'], cmap='viridis')
        plt.title('TSNE OPTICS Clustering Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title="Cluster ID")
        plt.show()


# ## GMM

# In[13]:


def AIC_BIC_plot(df):
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df) for n in n_components]

    aic = [m.aic(df) for m in models]
    bic = [m.bic(df) for m in models]

    plt.plot(n_components, aic, label='AIC')
    plt.plot(n_components, bic, label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.ylabel('Information Criterion')
    plt.show()
    
    #Pick lowest AIC if accuracy is valued. Pick lowest BIC if complexity is valued and needed to avoid overfitting


# In[14]:


def get_gmm_statistics(df, n_components=2, covariance_type='full'):
    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.fit(df)
    clusters = gmm.predict(df)

    df_temp = df.copy()
    df_temp['Cluster'] = clusters
    df_temp['TimeNumeric'] = df_temp.index.astype(np.float64) / 60  # Assuming index is in minutes, convert to hours

    # Mean, median, std of numeric time for each cluster
    cluster_stats = df_temp.groupby('Cluster')['TimeNumeric'].agg(['mean', 'median', 'std'])
    cluster_stats_sorted = cluster_stats.sort_values(by='median', ascending=True)

    print(df_temp['Cluster'].value_counts())
    print(cluster_stats_sorted)  # in hrs

    if 'PC1' in df_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df_temp['PC1'], df_temp['PC2'], df_temp['PC3'], c=df_temp['Cluster'], cmap='viridis')
        ax.set_title('PCA GMM Clustering Result')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        legend1 = ax.legend(*scatter.legend_elements(), title="Cluster ID")
        ax.add_artist(legend1)
        plt.show()
    elif 'TSNE1' in df_temp.columns:
        fig = plt.figure(figsize=(15, 15))
        scatter = plt.scatter(df_temp['TSNE1'], df_temp['TSNE2'], c=df_temp['Cluster'], cmap='viridis')
        plt.title('TSNE GMM Clustering Result')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.legend(*scatter.legend_elements(), title="Cluster ID")
        plt.show()

