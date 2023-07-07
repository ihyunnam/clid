import networkx as nx
from networkx.algorithms import bipartite
import json
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
import numpy as np
from scipy.stats import norm
from numpy import vstack
from numpy import argmax
# from numpy.random import random
from numpy import asarray
from sklearn.gaussian_process import GaussianProcessRegressor

# Read the JSONL file
input_file = '/home/ihyun/sni/hs_test2.json'
output_file_domain = '/home/ihyun/sni/domain_optimal2.json'
output_file_clients = '/home/ihyun/sni/client_optimal2.json'

# one copy for each appearance
domains = []
clients = {
    'server_name':[],
    'fp':[]
}

with open(input_file, 'r') as file:
    for line in file:
        fp = json.loads(line)
        # client_ip = record['client_ip']
        if 'client_hello' in fp['handshake'] and 'server_name' in fp['handshake']['client_hello']:
            domain = fp['handshake']['client_hello']['server_name']
            # if domain not in domains:
            domains.append(domain)

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
            
            # if client not in clients:
            # clients and server_name index must be the same
            clients['fp'].append(client)
            server_name = fp['handshake']['client_hello']['server_name']
            clients['server_name'].append(server_name)

# normalize by taking avg of everything
# given an domain cluster and a client clustser, compute the weight of edge between them
def compute_weight(domain_cluster, client_cluster):
    # df_domain, df_clients = df_domain.align(df_clients, axis=1, copy=False)
    # domain_cluster, client_cluster = domain_cluster.align(client_cluster, axis=1, copy=False)
    # avg. number of times each client was connected to a domain in domain_cluster (sum/number of client)
    frequency = 0
    for index, row in client_cluster.iterrows():
        # total_num_clients = 0
        client_sni = row['server_name']
        # total_num_clients = total_num_clients + len(df_clients[df_clients['cluster_label'] == row['cluster_label']])
        for index, row1 in domain_cluster.iterrows():
            if client_sni == row1['client_ip']:
                frequency = frequency+ 1

    total_num_clients = len(client_cluster)
    total_num_domain = len(domain_cluster)

    frequency = frequency/total_num_clients
    print("frequency " + str(frequency))

    # number of client clusters that this appears in = avg. number of 
    # TODO: counterintuitive, change na me
    non_exclusivity = 0
    # total_num_domain = 0
    # for index, row in domain_cluster.iterrows():
    #     total_num_domain = total_num_domain + len(df_domain[df_domain['cluster_label'] == row['cluster_label']])

    for index, row in domain_cluster.iterrows():
        print("domain " + str(domain))
        in_others_count = domains.count(row['client_ip'])
        in_C_count = 0
        for index, row in client_cluster.iterrows():
            if row['server_name'] == row['client_ip']:
                in_C_count += 1
        non_exclusivity += (in_others_count - in_C_count)

    non_exclusivity = non_exclusivity/total_num_domain+1

    # TODO: come up with name
    weight = frequency/non_exclusivity
    return np.float64(weight)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(domains)
Y = vectorizer.fit_transform(clients['fp'])
count_good_clusters = 0

# NOTE: n_clusters for domain and clients can be different
def evaluate_clusters(n_clusters_domain, n_clusters_client):
    kmeans_domain = KMeans(n_clusters=n_clusters_domain)
    kmeans_domain.fit(X)

    kmeans_client = KMeans(n_clusters=n_clusters_client)
    kmeans_client.fit(Y)

    domain_labels = kmeans_domain.labels_
    client_labels = kmeans_client.labels_

    df_domain = pd.DataFrame({'client_ip': domains, 'cluster_label': domain_labels})
    df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

    # create bipartite graph
    G = nx.Graph()
    bipartite.is_bipartite(G)

    # group 0 is domains
    # group 1 is client IPs
    G.add_nodes_from(domain_labels, bipartite=0)
    G.add_nodes_from(client_labels, bipartite=1)

    # how to calculate weight - reversely proportional to non_exclusivity (/including this one, in how many IP clusters does this appear)
    # proportional to how many times it appears in one IP(multiply)

    edge_list = []
    # add edges
    for domain in domain_labels:
        index = 0
        for client in client_labels:
            print(df_domain[df_domain['cluster_label'] == domain])
            weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
            if weight == 0:
                continue
            print("weight " + str(weight))
            index = index+1
            edge_list.append((domain, client, weight))
            G.add_weighted_edges_from([(u, v, {'weight': w}) for u, v, w in edge_list])

    # Optimization - optimize for number of 'good' D clusters that were identified by some S cluster with 
    # For each client (fp) compute the SD of all its domain weights

    count_good_clusters = 0

    # loop through all client clusters (identified by cluster label)
    for item in client_labels:
        sd_list = ()
        for edge in edge_list:
            if edge[1] == item:
                np.append(sd_list, list[2])

        sd = np.std(sd_list, dtype=np.float64)

        max_weight = 0
        second_max_weight = 0
        for item1 in edge_list:
            if item1[1] == item and item1[2] > second_max_weight:
                second_max_weight = item1[2]
                if item1[2] > max_weight:
                    max_weight = item1[2]
        
        # if there is at least one domain cluster that is associated with the client cluster (weight) by more than 1 std. dev. from all others -> GOOD
        if max_weight >= second_max_weight + sd:
            count_good_clusters += 1

    return count_good_clusters

# actual Bayesian optimization begins here
def approx(model, X):
    return model.predict(X)

def next_sample(X, Xsamples, model):
    yhat, _ = approx(model, X)
    best = max(yhat)
    mu, std = approx(model, Xsamples)
    mu = mu[:, 0]
    probs = norm.cdf((mu-best) / (std+1E-9))
    return probs

def opt_next_sample(X, y, model):
    Xsamples = []
    for i in range(10):
        Xsamples.append(random.randint(2,len(domains)))
    Xsamples = np.array(Xsamples).reshape(len(Xsamples),1)
    scores = next_sample(X.toarray(), Xsamples, model)
    ix = argmax(scores)
    return Xsamples[ix, 0]

# X is 'n_clusters' randomly chosen and then smart-ly predicted
rand_n_domain = []
rand_n_client = []

for i in range(10):
    rand_n_domain.append(random.randint(2,len(domains)))
    rand_n_client.append(random.randint(2,len(clients['fp'])))

rand_n_domain_y = []
for i in range(len(rand_n_domain)):
    # a, b = a.align(b, axis=1, copy=False)
    rand_n_domain_y.append(evaluate_clusters(rand_n_domain[i], rand_n_client[i]))

rand_n_domain = np.array(rand_n_domain).reshape(len(rand_n_domain), 1)
rand_n_domain_y = np.array(rand_n_domain_y).reshape(len(rand_n_domain_y), 1)
model = GaussianProcessRegressor()
model.fit(rand_n_domain,rand_n_domain_y)

# rand_n_client_y = asarray([evaluate_clusters(x) for x in rand_n_client])
# rand_n_client = rand_n_client.reshape(len(rand_n_client), 1)
# rand_n_client_y = rand_n_client_y.reshape(len(rand_n_client_y), 1)
# model_domain = GaussianProcessRegressor()
# model_domain.fit(rand_n_domain,rand_n_domain_y)

# model_client = GaussianProcessRegressor()
# model_client.fit(rand_n_client,rand_n_client_y)

# change range() to len(domains) if we want to 'sample' ALL possible n_clusters
for i in range(10):
    x = opt_next_sample(X,rand_n_domain_y, model)
    x_1 = opt_next_sample(Y, rand_n_client_y, model)
    actual = evaluate_clusters(x, x_1)
    est, _ = approx(model, [[x]])

    rand_n_domain = vstack((rand_n_domain, [[x]]))
    rand_n_domain_y = vstack((rand_n_domain_y, [[actual]]))
    rand_n_client = vstack((rand_n_client, [[x_1]]))
    rand_n_client_y = vstack((rand_n_client_y, [[actual]]))

    model.fit(rand_n_domain,rand_n_domain_y)
    model.fit(rand_n_client, rand_n_client_y)    

best_n_clusters_domain = argmax(rand_n_domain_y)
best_n_clusters_client = argmax(rand_n_client_y)

# search_space = {'n_clusters': (2, len(set(domains)))}

# # Perform Bayesian optimization
# optimizer = BayesSearchCV(
#     evaluate_clusters,
#     search_spaces=search_space,
#     n_iter=50,
#     # random_state=42
# )

# bested on optimization results, ACTUALLY producing data and graph

# optimizer.fit(evaluate_clusters, X)
# Get the best number of clusters found by Bayesian optimization
# optimizer.fit(X)
# best_n_clusters_domain = optimizer.best_score_

# optimizer.fit(Y, count_good_clusters)
# best_n_clusters_client = optimizer.best_score_

# K-means
kmeans_domain = KMeans(best_n_clusters_domain)

# num_clusters_clients = find_optimal_clusters(Y, len(clients['fp']))
kmeans_clients = KMeans(best_n_clusters_client)

# Get the cluster labels
domain_labels = kmeans_domain.labels_
client_labels = kmeans_clients.labels_

# Create a DataFrame with client_ip and its associated cluster label
# cluster_label = domains[0].split(".")[-2]
# cluster_label1 = ".".join(cluster_label)
df_domain = pd.DataFrame({'client_ip': domains, 'cluster_label': domain_labels})
df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

# save clustering results in json
df_sorted_domain = df_domain.sort_values('cluster_label')
with open(output_file_domain, 'w') as output:
    df_sorted_domain.to_json(output_file_domain, orient='records', lines=True)

df_sorted_clients = df_clients.sort_values('cluster_label')
with open(output_file_clients, 'w') as output:
    df_sorted_clients.to_json(output_file_clients, orient='records', lines=True)

# create bipartite graph
G = nx.Graph()
bipartite.is_bipartite(G)

# group 0 is domains
# group 1 is client IPs
G.add_nodes_from(domain_labels, bipartite=0)
G.add_nodes_from(client_labels, bipartite=1)

# how to calculate weight - reversely proportional to non_exclusivity (/including this one, in how many IP clusters does this appear)
# proportional to how many times it appears in one IP(multiply)

edge_list = []
# add edges
for domain in domain_labels:
    index = 0
    for client in client_labels:
        weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
        if weight == 0:
            continue
        print("weight " + str(weight))
        index = index+1
        edge_list.append((domain, client, weight))
        G.add_weighted_edges_from([(u, v, {'weight': w}) for u, v, w in edge_list])

weights = [G[u][v]['weight'] for u, v in G.edges()]

# pos = dict()
# pos.update((node, (0, index)) for index, node in enumerate(domain_labels))
# pos.update((node, (1, index)) for index, node in enumerate(client_labels))
pos = nx.drawing.layout.bipartite_layout(G, nodes=domain_labels)

labels = nx.get_edge_attributes(G,'weight')

# nx.draw_networkx(G, dpos = nx.drawing.layout.bipartite_layout(G, all_nodes), width = 2)

# nx.draw_networkx_edge_labels(G, pos=pos, edge_labels = labels )
plt.figure(figsize=(50,50))
plt.axis('scaled')  # Maintain the aspect ratio
plt.axis([0, 50, 0, 50])  # Example limits, adjust as needed
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
plt.show()
plt.savefig("graph.png")

# visualize as table
edges_data = pd.DataFrame({'Source': [edge[0] for edge in edge_list],
                           'Target': [edge[1] for edge in edge_list],
                           'Weight': [edge[2] for edge in edge_list]})

data_output = '/home/ihyun/sni/data_output.json'
with open(data_output, 'w') as output:
    edges_data.to_json(data_output, orient='records', lines=True)