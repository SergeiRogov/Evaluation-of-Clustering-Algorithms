from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import pandas as pd
import pprint
import re
import os

working_directory = '/Users/macbookair/Documents/UNIC FALL 2023/Machine Learning and Data Mining II/Project'
datasets = working_directory + '/datasets'
countries = datasets + '/Countries/Country-data.csv'


def apply_kmeans(csv_file, num_clusters, doc_type):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    if doc_type == 1:
        # Select the relevant columns for clustering
        data_no_countries = data.iloc[:, 1:]  # Exclude the 'country' column for clustering

        # Standardize the data
        scaler = StandardScaler()
        data_no_countries_scaled = scaler.fit_transform(data_no_countries)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        data['cluster'] = kmeans.fit_predict(data_no_countries_scaled)

        for cluster_num in range(num_clusters):
            cluster_data = data[data['cluster'] == cluster_num]
            print(f'\nCluster {cluster_num}:')
            print(cluster_data[['country', 'cluster']])


num_clusters = 3  # Number of clusters (k)
apply_kmeans(countries, num_clusters, 1)

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
