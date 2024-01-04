from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pprint
import re
import os

# project directory path on a local machine
working_directory = '/Users/macbookair/Documents/UNIC FALL 2023/Machine Learning and Data Mining II/Evaluation-of-Clustering-Algorithms'

# path to a folder with datasets documents
datasets = working_directory + '/datasets'

# complete paths to 3 csv-files containing datasets
countries = datasets + '/Countries/Country-data.csv'
songs = datasets + '/Songs/spotify_millsongdata.csv'
customers = datasets + '/Customers/segmentation data.csv'


'''
Function <code>apply_clustering_algorithms</code> performs clustering on a given 
dataset using 3 clustering algorithms: K-MEANS, HIERARCHICAL CLUSTERING and DBSCAN.
<BR>
@param csv_file (string): path to csv file with a dataset
@param num_clusters (int): number of clusters (k) for K-MEANS algorithm
@param dataset_type (string): countries, customers or songs
'''
def apply_clustering_algorithms(csv_file, num_clusters, dataset_type):

    # Read the CSV file
    data = pd.read_csv(csv_file)

    # for countries and customers datasets
    if dataset_type == "countries" or dataset_type == "customers":
        '''
        ===============
        === K-MEANS ===
        ===============
        '''
        # Exclude first column since it serves as an identification feature - unique for each instance
        # 'country' column for Countries dataset
        # 'ID' column for Customers dataset
        data_no_id = data.iloc[:, 1:]

        # Standardize the data
        scaler = StandardScaler()
        data_no_id_scaled = scaler.fit_transform(data_no_id)

        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        data['cluster'] = kmeans.fit_predict(data_no_id_scaled)

        # display the results
        for cluster_num in range(num_clusters):
            cluster_data = data[data['cluster'] == cluster_num]
            print(f'\nCluster {cluster_num}:')
            if dataset_type == "countries":
                print(cluster_data[['country']])

            elif dataset_type == "customers":
                print(cluster_data[['Sex', 'Marital status', 'Age', 'Education', 'Income',
                                    'Occupation', 'Settlement size']])
        '''
        ==========================
        === K-MEANS EVALUATION ===
        ==========================
        '''
        # within cluster sum of squared errors (WSS)
        wss = kmeans.inertia_
        print(f"WSS: {wss}")

        # between cluster sum of squared errors (BSS)
        # cluster centroids
        cluster_centers = kmeans.cluster_centers_
        # overall centroid
        overall_mean = data_no_id_scaled.mean(axis=0)
        # count the size of each cluster
        cluster_sizes = np.bincount(kmeans.labels_)
        # calculating between-cluster sum of squares (BSS)
        bss = np.sum(cluster_sizes * np.linalg.norm(cluster_centers - overall_mean, axis=1) ** 2)
        print(f"BSS: {bss}")

        # Elbow Method: determining the appropriate value for k-parameter - number of clusters
        wss = {}
        bss = {}
        # varying number of clusters in range from 1 to 20
        for k in range(1, 20):
            # performing K-Means clustering with current number of clusters K
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(data_no_id_scaled)

            # appending the value of within cluster sum of squared errors (WSS)
            # for a current K to a list
            wss[k] = kmeans.inertia_

            # between cluster sum of squared errors (BSS)
            # cluster centroids
            cluster_centers = kmeans.cluster_centers_
            # overall centroid
            overall_mean = data_no_id_scaled.mean(axis=0)
            # count the size of each cluster
            cluster_sizes = np.bincount(kmeans.labels_)
            # calculating between-cluster sum of squares (BSS) for a current K and appending it to a list
            bss[k] = np.sum(cluster_sizes * np.linalg.norm(cluster_centers - overall_mean, axis=1) ** 2)

        # plotting WSS and BSS on the same graph
        plt.figure(figsize=(10, 7))
        # plotting WSS
        sns.pointplot(x=list(wss.keys()), y=list(wss.values()), color='blue', label='WSS')
        # plotting BSS
        sns.pointplot(x=list(bss.keys()), y=list(bss.values()), color='red', label='BSS')
        plt.grid(True)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squares')
        plt.title('Elbow Method for optimal k')
        plt.legend()
        plt.show()

        '''
        ===============================
        === HIERARCHICAL CLUSTERING ===
        ===============================
        '''
        # Apply hierarchical clustering
        hierarchical = AgglomerativeClustering(linkage='ward')
        data['Cluster'] = hierarchical.fit_predict(data_no_id_scaled)

        # Plot the dendrogram
        plt.figure(figsize=(12, 8))

        ward_linkage_matrix = linkage(data_no_id_scaled, method='ward', metric='euclidean')

        if dataset_type == "countries":
            dendrogram(ward_linkage_matrix, orientation='top', labels=data['country'].tolist())
            plt.xlabel('Country')
        else:
            dendrogram(ward_linkage_matrix, orientation='top', labels=data['ID'].tolist())
            plt.xlabel('Customers')

        plt.title('Hierarchical Clustering Dendrogram')
        plt.ylabel('Distance')
        plt.show()

        '''
        ==========================================
        === HIERARCHICAL CLUSTERING EVALUATION ===
        ==========================================
        '''
        # todo: vary linkage and metric
        single_linkage_matrix = linkage(data_no_id_scaled, method='single', metric='euclidean')
        complete_linkage_matrix = linkage(data_no_id_scaled, method='complete', metric='euclidean')
        average_linkage_matrix = linkage(data_no_id_scaled, method='average', metric='euclidean')

        # todo: check for WSS and BSS with optimal numer of clusters
        #  (determined automatically by the algorithm, by truncating a tree when distance gets big)

        '''
        ==============
        === DBSCAN ===
        ==============
        '''
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        data['Cluster'] = dbscan.fit_predict(data_no_id_scaled)

        # todo: present results

        '''
        =========================
        === DBSCAN EVALUATION ===
        =========================
        '''
        # todo: vary eps and min_samples
        # todo: plot a graph for WSS and BSS on a 3D plane
    # for songs dataset
    elif dataset_type == "songs":
        '''
        ===============
        === K-MEANS ===
        ===============
        '''
        # extract a column with lyrics
        # to perform clustering only considering songs' lyrics
        lyrics_data = data["text"]

        # vectorize the text using TF-IDF representation
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        lyrics_matrix = tfidf_vectorizer.fit_transform(lyrics_data)

        # standardize the sparse text data
        scaler = MaxAbsScaler()
        lyrics_matrix_scaled = scaler.fit_transform(lyrics_matrix)

        # apply k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        data['cluster'] = kmeans.fit_predict(lyrics_matrix_scaled)

        '''
        ==========================
        === K-MEANS EVALUATION ===
        ==========================
        '''
        # within cluster sum of squared errors (WSS)
        sse = kmeans.inertia_
        print(f"SSE: {sse}")

        # the Elbow method
        wss = {}
        bss = {}
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, n_init=10)
            kmeans.fit(lyrics_matrix_scaled)
            wss[k] = kmeans.inertia_

            # Get cluster centers
            cluster_centers = kmeans.cluster_centers_
            # Calculate overall mean
            overall_mean = lyrics_matrix_scaled.mean(axis=0)
            cluster_sizes = np.bincount(kmeans.labels_)
            # Calculate between-cluster sum of squares (BSS)
            bss[k] = np.sum(cluster_sizes * np.linalg.norm(cluster_centers - overall_mean, axis=1) ** 2)

        # Plot both WSS and BSS on the same graph
        plt.figure(figsize=(10, 7))

        # Plot WSS
        sns.pointplot(x=list(wss.keys()), y=list(wss.values()), color='blue', label='WSS')

        # Plot BSS
        sns.pointplot(x=list(bss.keys()), y=list(bss.values()), color='red', label='BSS')

        plt.grid(True)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squares')
        plt.title('Elbow Method for optimal k')
        plt.legend()
        plt.show()

        for cluster_num in range(num_clusters):
            cluster_data = data[data['cluster'] == cluster_num]
            print(f'\nCluster {cluster_num}:')
            print(cluster_data[['artist', 'song', 'cluster']])


num_clusters = 4  # Number of clusters (k)

apply_clustering_algorithms(countries, num_clusters, "countries")
apply_clustering_algorithms(customers, num_clusters, "customers")
# apply_clustering_algorithms(songs, num_clusters, "songs")


#     # Preprocessing: removing everything except letters and digits
#     for i in range(len(documents)):
#         documents[i] = re.sub(r'[^a-zA-Z0-9 ]', '', documents[i])
#         documents[i] = documents[i].lower()
#
#     words_vector = TfidfVectorizer(max_features=1000, stop_words='english')
#
#     bag_of_words_matrix = words_vector.fit_transform(documents)
#
#     # creating csv files for text vector representations
#     cv_dataframe = pd.DataFrame(bag_of_words_matrix.toarray(), columns=words_vector.get_feature_names_out())
#     if not is_toy:
#         cv_dataframe.index = filenames
#
#     output_file_path = f"/Users/macbookair/Documents/UNIC FALL 2023/Machine Learning and Data Mining II/HW1-Clustering/representations/representation_{representation}{'_toy' if is_toy else ''}.csv"
#     cv_dataframe.to_csv(output_file_path, sep=',', index=True, header=True)