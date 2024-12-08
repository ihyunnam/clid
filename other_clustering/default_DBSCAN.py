from sklearn.cluster import DBSCAN
import networkx as nx
from networkx.algorithms import bipartite
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import numpy as np

input_file = '/Users/ihyunnam/Desktop/hs_test_100.json'
output_file_clients = '/Users/ihyunnam/Desktop/DBSCANlocal/dbscan_default_dist_client1.json'
output_file_domain = '/Users/ihyunnam/Desktop/DBSCANlocal/dbscan_default_dist_domain1.json'

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
        temp = []
        if 'client_hello' in fp['handshake'] and 'server_name' in fp['handshake']['client_hello']:
            domain = fp['handshake']['client_hello']['server_name']
            domains_flat.append(domain)
            temp.append(domain)
            domains.append(temp)

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

################################## compute edge weights ##################################

# given an domain cluster and a client clustser, compute the weight of edge between them
def compute_weight(domain_cluster, client_cluster):
    frequency = 0
    for index, row in client_cluster.iterrows():
        client_sni = row['server_name']
        for index, row1 in domain_cluster.iterrows():
            if client_sni == row1['client_ip']:
                frequency = frequency+ 1

    total_num_clients = len(client_cluster)
    total_num_domain = len(domain_cluster)

    frequency = frequency/total_num_clients

    non_exclusivity = 0
    for index, row in domain_cluster.iterrows():
        in_others_count = domains.count(row['client_ip'])
        in_C_count = 0
        for index, row in client_cluster.iterrows():
            if row['server_name'] == row['client_ip']:
                in_C_count += 1
        non_exclusivity += (in_others_count - in_C_count)

    non_exclusivity = non_exclusivity/total_num_domain+1

    weight = frequency/non_exclusivity
    return np.float64(weight)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(domains_flat)
Y = vectorizer.fit_transform(clients['fp'])
count_good_clusters = 0

################################## objective function for Bayesian optimization ##################################

# NOTE: eps for domain and clients can be different
def evaluate_clusters(eps_domain, eps_client):
    dbscan_domain = DBSCAN(eps=eps_domain, min_samples=1).fit(X)
    domain_labels = dbscan_domain.labels_

    dbscan_client = DBSCAN(eps=eps_client, min_samples=1).fit(Y)
    client_labels = dbscan_client.labels_

    df_domain = pd.DataFrame({'client_ip': domains_flat, 'cluster_label': domain_labels})
    df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

    G = nx.Graph()
    bipartite.is_bipartite(G)

    G.add_nodes_from(domain_labels, bipartite=0)
    G.add_nodes_from(client_labels, bipartite=1)

    # add edges
    edge_list = []
    for client in np.unique(client_labels):
        for domain in np.unique(domain_labels):
            unique_len_d = len(np.unique(df_domain[df_domain['cluster_label'] == str(domain)]))
            weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
            edge_list.append((domain, client, weight))
    
    G.add_weighted_edges_from([(u, v, {'weight': w}) for u, v, w in edge_list])

    count_good_clusters = 0
      
    # identify and count good client clusters
    for item in np.unique(client_labels):
        good = False
        sd_list = []
        for edge in edge_list:
            if edge[1] == item:
                sd_list.append(edge[2])
        sd_list = sorted(sd_list, reverse = True)
        
        # up to three is okay
        for count in range(3):
            sd_list_rest = sd_list[count+1:]
            sd = np.std(sd_list_rest)
            if count > 0:
                if sd_list[count] - sd_list[count-1] < sd:
                    if np.min(sd_list[:count+1]) > sd + np.max(sd_list_rest):
                        good = True
                        break

        if good == True:
            count_good_clusters += 1

    return count_good_clusters 

################################## Bayesian optimization to find optimal eps ##################################
param_bounds = {'eps_domain': (1e-6, 1),
                'eps_client': (1e-6, 1)}

optimizer = BayesianOptimization(f=evaluate_clusters, pbounds=param_bounds)
optimizer.maximize(n_iter=10)

# Retrieve the input that resulted in the highest score
best_eps_domain = optimizer.max['params']['eps_domain']
best_eps_client = optimizer.max['params']['eps_client']
best_score = optimizer.max['target']


################################## perform DBSCAN clustering with optimal eps ##################################

dbscan_domain = DBSCAN(eps=best_eps_domain, min_samples=1).fit(X)
domain_labels = dbscan_domain.labels_

dbscan_client = DBSCAN(eps=best_eps_client, min_samples=1).fit(Y)
client_labels = dbscan_client.labels_

df_domain = pd.DataFrame({'client_ip': domains_flat, 'cluster_label': domain_labels})
df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

# save clustering results in json files
df_sorted_domain = df_domain.sort_values('cluster_label')
with open(output_file_domain, 'w') as output:
    df_sorted_domain.to_json(output_file_domain, orient='records', lines=True)

df_sorted_clients = df_clients.sort_values('cluster_label')
with open(output_file_clients, 'w') as output:
    df_sorted_clients.to_json(output_file_clients, orient='records', lines=True)

################################## create bipartite graph ##################################

G = nx.Graph()
bipartite.is_bipartite(G)

G.add_nodes_from(domain_labels, bipartite=0)
G.add_nodes_from(client_labels, bipartite=1)

# add edges
edge_list = []
for domain in np.unique(domain_labels):
    index = 0
    for client in np.unique(client_labels):
        weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
        index = index+1
        edge_list.append((domain, client, weight))
        G.add_weighted_edges_from([(u, v, {'weight': w}) for u, v, w in edge_list])

weights = [G[u][v]['weight'] for u, v in G.edges()]

pos = nx.drawing.layout.bipartite_layout(G, nodes=np.unique(domain_labels))
labels = nx.get_edge_attributes(G,'weight')

plt.figure(figsize=(50,50))
plt.axis('scaled')  # Maintain the aspect ratio
plt.axis([0, 50, 0, 50])  # Example limits, adjust as needed
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels)
plt.show()
plt.savefig("graph.png")

# visualize as table
edges_data = pd.DataFrame({'Domain': [edge[0] for edge in edge_list],
                           'Client': [edge[1] for edge in edge_list],
                           'Weight': [edge[2] for edge in edge_list]})

# save clustering results in json files
with open('/Users/ihyunnam/Desktop/DBSCANlocal/dbscan_default_dist_result1.json', 'w') as output:
    edges_data.to_json(data_output, orient='records', lines=True)

with open('/Users/ihyunnam/Desktop/DBSCANlocal/dbscan_default_dist_info1.json', 'w') as output:
    output.write(json.dumps("all sd " + str(all_sd)) + '\n')
    output.write(json.dumps("num good clusters" + str(best_score)) + '\n')
    for cluster in np.unique(client_labels):
        for edge in edge_list:
            if edge[1] == cluster:
                output.write(json.dumps(str(cluster)+ ":"+str(edge[0])+ "," +str(edge[2])) + '\n')
