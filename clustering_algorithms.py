from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import pandas as pd
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


def apply_kmeans(csv_file, num_clusters, dataset_type):

    # Read the CSV file
    data = pd.read_csv(csv_file)

    if dataset_type == "countries" or dataset_type == "customers":
        # Exclude first column since it serves as an identification feature - unique for each instance
        # 'country' column for Countries dataset
        # 'ID' column for Customers dataset
        data_no_id = data.iloc[:, 1:]

        # Standardize the data
        scaler = StandardScaler()
        data_no_id_scaled = scaler.fit_transform(data_no_id)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        data['cluster'] = kmeans.fit_predict(data_no_id_scaled)

        for cluster_num in range(num_clusters):
            cluster_data = data[data['cluster'] == cluster_num]
            print(f'\nCluster {cluster_num}:')
            if dataset_type == "countries":
                print(cluster_data[['country']])
            elif dataset_type == "customers":
                print(cluster_data[['Sex', 'Marital status', 'Age', 'Education', 'Income',
                                    'Occupation', 'Settlement size']])

    elif dataset_type == "songs":
        print("songs")


num_clusters = 3  # Number of clusters (k)

apply_kmeans(countries, num_clusters, "countries")
apply_kmeans(customers, num_clusters, "customers")
apply_kmeans(songs, num_clusters, "songs")

# Visualize the clusters
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['cluster'], cmap='viridis')
# plt.title('K-means Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()


def apply_hierarchical():
    pass


def apply_dbscan():
    pass


apply_hierarchical()
apply_dbscan()

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
