from sklearn.cluster import DBSCAN
import networkx as nx
from networkx.algorithms import bipartite
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix

number = 1

# Read the JSONL file
input_file = '/home/ihyun/sni/hs_test_2000.json'
output_file_domain = "/home/ihyun/sni_sd_trials/dbscan_custom_domain_{:.2f}.json".format(number)
output_file_clients = "/home/ihyun/sni_sd_trials/dbscan_custom_client_{:.2f}.json".format(number)

# one copy for each appearance
domains = []
domains_flat = []
clients = {
    'server_name':[],
    'fp':[]
}

with open(input_file, 'r') as file:
    for line in file:
        fp = json.loads(line)
        # client_ip = record['client_ip']
        temp = []
        if 'client_hello' in fp['handshake'] and 'server_name' in fp['handshake']['client_hello']:
            domain = fp['handshake']['client_hello']['server_name']
            domains_flat.append(domain)
            temp.append(domain)
            domains.append(temp)

            # accounting for missing data
            if 'tcp_header_length' in fp['tcp_fingerprint']:
                tcp_header_length = fp['tcp_fingerprint']['tcp_header_length']
            else:
                tcp_header_length = ""

            if 'ip_ttl' in fp['tcp_fingerprint']:
                ip_ttl = fp['tcp_fingerprint']['ip_ttl']
            else:
                ip_ttl = ""

            if 'tcp_window_size' in fp['tcp_fingerprint']:
                tcp_window_size = fp['tcp_fingerprint']['tcp_window_size']
            else:
                tcp_window_size = ""

            if 'tcp_flags' in fp['tcp_fingerprint']:
                tcp_flags = fp['tcp_fingerprint']['tcp_flags']
            else:
                tcp_flags = ""

            if 'tcp_mss' in fp['tcp_fingerprint']:
                tcp_mss = fp['tcp_fingerprint']['tcp_mss']
            else:
                tcp_mss = ""

            if 'tcp_options' in fp['tcp_fingerprint']:
                tcp_options = fp['tcp_fingerprint']['tcp_options']
            else:
                tcp_options = ""

            if 'tcp_window_scaling' in fp['tcp_fingerprint']:
                tcp_window_scaling = fp['tcp_fingerprint']['tcp_window_scaling']
            else:
                tcp_window_scaling = ""

            client = tcp_header_length + ip_ttl + tcp_window_size + tcp_flags + tcp_mss + tcp_options + tcp_window_scaling

            clients['fp'].append(client)
            server_name = fp['handshake']['client_hello']['server_name']
            clients['server_name'].append(server_name)

################################## intelligently compute distance between domain names, used for Bayesian optimization AND actual clustering ##################################

# compute distance between two SNIs
def custom_distance(x, y):
    if x[:4] == 'www.':
        x=x[4:]
    if y[:4] == 'www.':
        y=y[4:]
    x_indices = x.split('.')
    y_indices = y.split('.')

    distance = 0
    x_len = len(x_indices)
    y_len = len(y_indices)
    shorter = min(x_len, y_len)

    for index in range(shorter):
        # for TLD (e.g. netflix.com) needs exact matching
        if index <2:
            if x_indices[x_len-index-1] != y_indices[y_len-index-1]:
                if index == 0:
                    distance += 1/3 # arbitrary low weight for TLD (top level domain)
                else:
                    distance += 1/index  # assign more weight to the end of the string
        # for paths less than TLD, check for inclusion e.g. profiles.stanford.edu and profiles-02.stanford.edu in same cluster
        else:
            similarity_score = fuzz.ratio(x_indices[x_len-index-1], y_indices[y_len-index-1])
            distance += (1-0.01*similarity_score) * 1/index
    return distance

################################## compute weight of edges, used for generating bipartite graph ##################################

# normalize by taking avg of everything
# given an domain cluster and a client clustser, compute the weight of edge between them
def compute_weight(domain_cluster, client_cluster):
    # avg. number of times each client was connected to a domain in domain_cluster (sum/number of client)
    total_num_clients = len(client_cluster)
    total_num_domain = len(domain_cluster)

    frequency = 0
    for index, row in client_cluster.iterrows():
        for index, row1 in domain_cluster.iterrows():
            if row['server_name'] == row1['domain']:
                frequency += 1
                break

    frequency = frequency/total_num_clients

    non_exclusivity = total_num_domain
    for index, row in client_cluster.iterrows():
        for index, row1 in domain_cluster.iterrows():
            if row['server_name'] == row1['domain']:
                non_exclusivity -= 1
                break

    non_exclusivity = (non_exclusivity+0.5)/total_num_domain

    # TODO: come up with name
    weight = frequency/non_exclusivity
    return np.float64(weight)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(domains_flat)
Y = vectorizer.fit_transform(clients['fp'])

count_good_clusters = 0

################################## compute distance matrix ##################################

dimension = len(domains_flat)
distance_data = []

for x in range(dimension):
    for y in range(dimension):
        distance = custom_distance(domains_flat[x], domains_flat[y])
        distance_data.append((x, y, distance))

# Build the sparse distance matrix using coo_matrix
rows, cols, distances = zip(*distance_data)
distance_matrix = csr_matrix((distances, (rows, cols)), shape=(dimension, dimension))

# Divide by maximum entry in distance_matrix to normalize distances
max_distance = distance_matrix.max()
if max_distance != 0:
    distance_matrix /= max_distance

################################## evaluate clusters, used for Bayesian optimization ##################################

all_good_clusters = []
# NOTE: eps for domain and clients can be different
def evaluate_clusters(eps_domain, eps_client):
    dbscan_domain = DBSCAN(eps=eps_domain, min_samples=1).fit(X)
    domain_labels = dbscan_domain.labels_

    dbscan_client = DBSCAN(eps=eps_client, min_samples=1).fit(Y)
    client_labels = dbscan_client.labels_

    df_domain = pd.DataFrame({'domain': domains_flat, 'cluster_label': domain_labels})
    df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

    edge_list = []
    # add edges
    for client in np.unique(client_labels):
        for domain in np.unique(domain_labels):
            weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
            edge_list.append((domain, client, weight))

    count_good_clusters = 0

    # loop through all client clusters (identified by cluster label)
    for item in np.unique(client_labels):
        sd_list = []
        for edge in edge_list:
            # only consider non-zero weights
            if edge[1] == item and edge[2] != 0:
                sd_list.append(edge[2])
        
        # skip if all weights are zero
        if len(sd_list) == 0:
            continue

        if len(sd_list) == 1:
            count_good_clusters += 1
            continue

        # not good if all non-zero weights equal
        if len(set(sd_list)) == 1:
            continue

        mean = sum(sd_list)/len(sd_list)
        sd = np.std(sd_list)
        if max(sd_list) >= mean + number*sd:
            count_good_clusters += 1

        # add unique_len_c/d to prevent singletons or large domain clusters
    return count_good_clusters + (len(np.unique(domain_labels)) + len(domains_flat)/len(np.unique(domain_labels)))/2

################################## Bayesian optimization to find optimal n_clusters ##################################

param_bounds = {'eps_domain': (1e-6, 1),
                'eps_client': (1e-6, 1)}  # TODO: What to set as min, max, increment size?

# Create the BayesianOptimization object
optimizer = BayesianOptimization(f=evaluate_clusters, pbounds=param_bounds)
optimizer.maximize(n_iter=10)

# Retrieve the input that resulted in the highest score
best_eps_domain = optimizer.max['params']['eps_domain']
best_eps_client = optimizer.max['params']['eps_client']
best_score = optimizer.max['target']

################################## perform DBSCAN clustering with optimal n_clusters ##################################

# compute custom distances on raw strings -> relate string domains to vectorized -> .fit(X vectorized)dbsc
dbscan_domain = DBSCAN(eps=best_eps_domain, min_samples=1, metric='precomputed').fit(distance_matrix)
domain_labels = dbscan_domain.labels_

dbscan_client = DBSCAN(eps=best_eps_client, min_samples=1).fit(Y)
client_labels = dbscan_client.labels_

df_domain = pd.DataFrame({'domain': domains_flat, 'cluster_label': domain_labels})
df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

# save clustering results in json
df_sorted_domain = df_domain.sort_values('cluster_label')
with open(output_file_domain, 'w') as output:
    df_sorted_domain.to_json(output_file_domain, orient='records', lines=True)

df_sorted_clients = df_clients.sort_values('cluster_label')
with open(output_file_clients, 'w') as output:
    df_sorted_clients.to_json(output_file_clients, orient='records', lines=True)

################################## create bipartite graph ##################################

edge_list = []

for domain in np.unique(domain_labels):
    index = 0
    for client in np.unique(client_labels):
        weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
        index = index+1
        edge_list.append((domain, client, weight))

max_weight = float('-inf')

# Iterate through edge_list and find the maximum weight
for u, v, weight in edge_list:
    if weight > max_weight:
        max_weight = weight

count_good_clusters = 0
all_good_clusters = []
for item in np.unique(client_labels):
    sd_list = []
    for edge in edge_list:
        # only consider non-zero weights
        if edge[1] == item and edge[2] != 0:
            sd_list.append(edge[2])
    
    mean = sum(sd_list)/len(sd_list)
    
    # skip if all weights are zero
    if len(sd_list) == 0:
        continue

    if len(sd_list) == 1:
        count_good_clusters += 1
        all_good_clusters.append(item)
        continue

    # not good if all non-zero weights equal
    if len(set(sd_list)) == 1:
        continue

    sd = np.std(sd_list)
    if max(sd_list) >= mean + number*sd:
        count_good_clusters += 1
        all_good_clusters.append(item)

# visualize as table
edges_data = pd.DataFrame({'Domain': [edge[0] for edge in edge_list],
                           'Client': [edge[1] for edge in edge_list],
                           'Weight': [edge[2] for edge in edge_list],
                           'Weight significance': [edge[2]/max_weight for edge in edge_list]})

data_output = "/home/ihyun/sni_sd_trials/dbscan_custom_result_{:.2f}.json".format(number)
with open(data_output, 'w') as output:
    edges_data.to_json(data_output, orient='records', lines=True)

with open("/home/ihyun/sni_sd_trials/dbscan_custom_info_{:.2f}.json".format(number), 'w') as output:
    output.write(json.dumps("num good clusters" + str(count_good_clusters)) + '\n')
    output.write(json.dumps("all good clusters" + str(all_good_clusters)) + '\n')
