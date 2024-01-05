from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
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


def calculate_wss(data, clusters):
    wss = 0.0
    unique_clusters = set(clusters)

    for cluster in unique_clusters:
        cluster_indices = (clusters == cluster)
        cluster_data = data[cluster_indices]
        cluster_center = cluster_data.mean(axis=0)
        cluster_distance = euclidean_distances(cluster_data, [cluster_center])
        wss += (cluster_distance ** 2).sum()

    return wss


# Calculate the between-cluster sum of squared errors
def calculate_bss(data, clusters):
    unique_clusters = set(clusters)
    overall_mean = data.mean(axis=0)
    bss = 0.0

    for cluster in unique_clusters:
        cluster_indices = (clusters == cluster)
        cluster_data = data[cluster_indices]
        cluster_center = cluster_data.mean(axis=0)
        bss += len(cluster_data) * euclidean_distances([cluster_center], [overall_mean]) ** 2

    return float(bss)


'''
Function <code>apply_clustering_algorithms</code> performs clustering on a given 
dataset using 3 clustering algorithms: K-MEANS, HIERARCHICAL CLUSTERING and DBSCAN.
<BR>
@param csv_file (string): path to csv file with a dataset
@param num_clusters (int): number of clusters (k) for K-MEANS algorithm
@param dataset_type (string): countries, customers or songs
'''
def apply_clustering_algorithms(csv_file, dataset_type):

    # Read the CSV file
    data = pd.read_csv(csv_file)

    # for countries and customers datasets
    if dataset_type == "countries" or dataset_type == "customers":
        '''
        ===============
        === K-MEANS ===
        ===============
        '''
        num_clusters = 4  # Number of clusters (k)

        # Exclude first column since it serves as an identification feature - unique for each instance
        # 'country' column for Countries dataset
        # 'ID' column for Customers dataset
        data_no_id = data.iloc[:, 1:]

        # Standardize the data
        scaler = StandardScaler()
        data_no_id_scaled = scaler.fit_transform(data_no_id)

        # # Perform K-Means clustering
        # kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        # cluster_assignments = kmeans.fit_predict(data_no_id_scaled)
        #
        # # display the results
        # for cluster_num in range(num_clusters):
        #     cluster_indices = (cluster_assignments == cluster_num)
        #     cluster_data = data[cluster_indices]
        #
        #     print(f'\nCluster {cluster_num}:')
        #     if dataset_type == "countries":
        #         print(cluster_data[['country']])
        #
        #     elif dataset_type == "customers":
        #         print(cluster_data[['Sex', 'Marital status', 'Age', 'Education', 'Income',
        #                             'Occupation', 'Settlement size']])
        # '''
        # ==========================
        # === K-MEANS EVALUATION ===
        # ==========================
        # '''
        # # within cluster sum of squared errors (WSS)
        # wss = kmeans.inertia_
        # print(f"WSS: {wss}")
        #
        # # Get cluster assignments
        # clusters = kmeans.labels_
        #
        # bss = calculate_bss(data_no_id_scaled, clusters)
        # print(f"BSS: {bss}")
        #
        # # Elbow Method: determining the appropriate value for k-parameter - number of clusters
        # wss = {}
        # bss = {}
        # # varying number of clusters in range from 1 to 20
        # for k in range(1, 20):
        #     # performing K-Means clustering with current number of clusters K
        #     kmeans = KMeans(n_clusters=k, n_init=10)
        #     kmeans.fit(data_no_id_scaled)
        #
        #     # appending the value of within-cluster sum of squared errors (WSS) for a current k to a list
        #     wss[k] = kmeans.inertia_
        #
        #     clusters = kmeans.labels_
        #
        #     # calculating between-cluster sum of squares (BSS) for a current K and appending it to a list
        #     bss[k] = calculate_bss(data_no_id_scaled, clusters)
        #
        # # plotting WSS and BSS on the same graph
        # plt.figure(figsize=(10, 7))
        # # plotting WSS
        # sns.pointplot(x=list(wss.keys()), y=list(wss.values()), color='red', label='WSS')
        # # plotting BSS
        # sns.pointplot(x=list(bss.keys()), y=list(bss.values()), color='blue', label='BSS')
        # plt.grid(True)
        # plt.xlabel('Number of Clusters (k)')
        # plt.ylabel('Sum of Squares')
        # plt.title('Elbow Method for optimal k')
        # plt.legend()
        # plt.show()
        #
        # '''
        # ===============================
        # === HIERARCHICAL CLUSTERING ===
        # ===============================
        # '''
        # # Apply hierarchical clustering
        # ward_linkage_matrix = linkage(data_no_id_scaled, method='ward', metric='euclidean')
        #
        # if dataset_type == "countries":
        #     height = 15
        #     labels = data['country'].tolist()
        # else:
        #     height = 35
        #     labels = data['ID'].tolist()
        #
        # # Cut the dendrogram and get cluster assignments
        # clusters = fcluster(ward_linkage_matrix, height, criterion='distance')
        #
        # # Plot the dendrogram
        # plt.figure(figsize=(12, 8))
        #
        # dendrogram(ward_linkage_matrix, orientation='top', labels=labels, color_threshold=height)
        # plt.xlabel('Country' if dataset_type == "countries" else 'Customers')
        #
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.ylabel('Distance')
        # plt.show()
        #
        # # Display the results for each identified cluster
        # num_clusters = len(set(clusters))
        #
        # for cluster_num in range(1, num_clusters + 1):
        #     cluster_indices = (clusters == cluster_num)
        #     cluster_data = data[cluster_indices]
        #
        #     print(f'\nCluster {cluster_num}:')
        #     if dataset_type == "countries":
        #         print(cluster_data[['country']])
        #
        #     elif dataset_type == "customers":
        #         print(cluster_data[['Sex', 'Marital status', 'Age', 'Education', 'Income',
        #                             'Occupation', 'Settlement size']])
        #
        # '''
        # ==========================================
        # === HIERARCHICAL CLUSTERING EVALUATION ===
        # ==========================================
        # '''
        # if dataset_type == "countries":
        #     heights = [6, 5, 4.5, 4, 2, 8, 15]
        # else:
        #     heights = [4, 3.5, 4.3, 2.75, 1.75, 6, 35]
        #
        # methods = ['weighted', 'median', 'average', 'centroid', 'single', 'complete', 'ward']
        # wss = []
        # bss = []
        # for i, method in enumerate(methods):
        #     linkage_matrix = linkage(data_no_id_scaled, method=method, metric='euclidean')
        #     clusters = fcluster(linkage_matrix, heights[i], criterion='distance')
        #
        #     wss.append(calculate_wss(data_no_id_scaled, clusters))
        #
        #     bss.append(calculate_bss(data_no_id_scaled, clusters))
        #
        # # plotting WSS and BSS on the same graph
        # plt.figure(figsize=(10, 7))
        # # plotting WSS
        # sns.pointplot(x=methods, y=wss, color='red', label='WSS')
        # # plotting BSS
        # sns.pointplot(x=methods, y=bss, color='blue', label='BSS')
        # plt.grid(True)
        # plt.xlabel('Linkage method')
        # plt.ylabel('Sum of Squares')
        # plt.title('WSS and BSS for Different Linkage Methods')
        # plt.legend()
        # plt.show()

        '''
        ==============
        === DBSCAN ===
        ==============
        '''
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=2, min_samples=3)
        clusters = dbscan.fit_predict(data_no_id_scaled)

        # Display the results for each identified cluster
        num_clusters = len(set(clusters))

        for cluster_num in range(1, num_clusters + 1):
            cluster_indices = (clusters == cluster_num)
            cluster_data = data[cluster_indices]

            print(f'\nCluster {cluster_num}:')
            if dataset_type == "countries":
                print(cluster_data[['country']])

            elif dataset_type == "customers":
                print(cluster_data[['Sex', 'Marital status', 'Age', 'Education', 'Income',
                                    'Occupation', 'Settlement size']])

        '''
        =========================
        === DBSCAN EVALUATION ===
        =========================
        '''
        # Varying parameters
        eps_values = np.linspace(0.1, 2.5, 10)
        min_samples_values = np.arange(3, 10)

        # Create a meshgrid for eps and min_samples
        eps_grid, min_samples_grid = np.meshgrid(eps_values, min_samples_values)

        # Initialize an array to store SSE values
        sse_values = np.zeros_like(eps_grid)

        # Calculate SSE for each combination of eps and min_samples
        for i in range(eps_grid.shape[0]):
            for j in range(eps_grid.shape[1]):
                eps = eps_grid[i, j]
                min_samples = min_samples_grid[i, j]

                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(data_no_id_scaled)

                # Calculate SSE (sum of distances to core points)
                core_distances = np.zeros_like(clusters, dtype=float)

                for label in set(clusters):
                    if label != -1:  # Exclude noise points
                        cluster_points = data_no_id_scaled[clusters == label]
                        core_point = np.mean(cluster_points, axis=0)
                        core_distances[clusters == label] = np.linalg.norm(cluster_points - core_point, axis=1)**2

                sse = np.sum(core_distances)
                sse_values[i, j] = sse

        # Create a 3D surface plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(eps_grid, min_samples_grid, sse_values, cmap='viridis')

        # Customize the plot
        ax.set_xlabel('eps')
        ax.set_ylabel('min_samples')
        ax.set_zlabel('SSE')
        ax.set_title('DBSCAN Clustering SSE with Varying Parameters')

        plt.show()
    # for songs dataset
    elif dataset_type == "songs":
        '''
        ===============
        === K-MEANS ===
        ===============
        '''
        num_clusters = 4  # Number of clusters (k)

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


apply_clustering_algorithms(countries, "countries")
apply_clustering_algorithms(customers, "customers")
# apply_clustering_algorithms(songs, "songs")


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