import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from bayes_opt import BayesianOptimization
import numpy as np
import ujson

num_highest_allowed = 1
number = 1
################################## Set up ##################################

# Read the JSONL file
input_file = '/Users/ihyunnam/Desktop/hs_test_200.json'
output_file_domain = '/Users/ihyunnam/Desktop/kmeans_domain.json'
output_file_clients = '/Users/ihyunnam/Desktop/kmeans_client.json'

# one copy for each appearance
domains = []
domains_flat = []
clients = {
    'server_name':[],
    'fp':[]
}

with open(input_file, 'r') as file:
    for line in file:
        fp = ujson.loads(line)
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


# normalize by taking avg of everything
# given an domain cluster and a client clustser, compute the weight of edge between them
def compute_weight(domain_cluster, client_cluster):
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


################################## evaluate clusters, used for Bayesian optimization ##################################
all_good_clusters = []
all_sd = []

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(domains_flat)
Y = vectorizer.fit_transform(clients['fp'])

# NOTE: n_clusters for domain and clients can be different
def evaluate_clusters(n_clusters_domain, n_clusters_client):
    kmeans_domain = KMeans(n_clusters=int(n_clusters_domain))
    kmeans_domain.fit(X)

    kmeans_client = KMeans(n_clusters=int(n_clusters_client))
    kmeans_client.fit(Y)

    domain_labels = kmeans_domain.labels_
    client_labels = kmeans_client.labels_

    df_domain = pd.DataFrame({'domain': domains_flat, 'cluster_label': domain_labels})
    df_clients = pd.DataFrame({'client_ip': clients['fp'], 'server_name': clients['server_name'], 'cluster_label': client_labels})

    edge_list = []
    # add edges
    for client in np.unique(client_labels):
        for domain in np.unique(domain_labels):
            weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
            if weight != 0:
                edge_list.append((domain, client, weight))

    df = pd.DataFrame(edge_list, columns=['Domain', 'Client', 'Weight'])

    # Group by 'client' column
    client_groups = df.groupby('Client')['Weight']

    # Function to calculate standard deviation and other metrics for each client group
    count_good_clusters = 0

    for group, weights in client_groups:
        if len(weights)<= num_highest_allowed:
            count_good_clusters += 1
            continue

        std_dev = np.std(weights)
        highest_weight = np.max(weights)
        mean = np.mean(weights)
        
        # if std dev is 0, then all weights are equal, therefore not good
        if std_dev == 0:
            continue
        else:
            z_score_highest = (highest_weight - mean) / std_dev
            if z_score_highest >= number:
                count_good_clusters += 1 

        # add unique_len_c/d to prevent singletons or large domain clusters
    return count_good_clusters +(len(np.unique(domain_labels))+len(domains_flat)/len(np.unique(domain_labels)))/2



################################## Bayesian optimization to find optimal n_clusters ##################################

pbounds = {'n_clusters_domain': (1, len(domains)),
           'n_clusters_client': (1, len(clients))}

optimizer = BayesianOptimization(f=evaluate_clusters, pbounds=pbounds)

# Perform Bayesian optimization iterations
optimizer.maximize(n_iter=10)

# Get the optimal hyperparameters and maximum objective value
best_n_clusters_domain = optimizer.max['params']['n_clusters_domain']
best_n_clusters_client = optimizer.max['params']['n_clusters_client']
best_score = optimizer.max['target']

################################## perform kmeans clustering with optimal n_clusters ##################################

kmeans_domain = KMeans(int(best_n_clusters_domain))
kmeans_domain.fit(X)

kmeans_clients = KMeans(int(best_n_clusters_client))
kmeans_clients.fit(Y)

# Get the cluster labels
domain_labels = kmeans_domain.labels_
client_labels = kmeans_clients.labels_

# Create a DataFrame with client_ip and its associated cluster label
df_domain = pd.DataFrame({'domain': domains, 'cluster_label': domain_labels})
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
    for client in np.unique(client_labels):
        weight = compute_weight(df_domain[df_domain['cluster_label'] == domain], df_clients[df_clients['cluster_label'] == client])
        if weight != 0:
            edge_list.append((domain, client, weight))

max_weight = float('-inf')

# Iterate through edge_list and find the maximum weight
for u, v, weight in edge_list:
    if weight > max_weight:
        max_weight = weight

count_good_clusters = 0
all_good_clusters = []

df = pd.DataFrame(edge_list, columns=['Domain', 'Client', 'Weight'])

# Group by 'client' column
client_groups = df.groupby('Client')['Weight']

# Function to calculate standard deviation and other metrics for each client group

for group, weights in client_groups:
    std_dev = np.std(weights)
    highest_weight = np.max(weights)
    mean = np.mean(weights)
    mean1 = sum(weights)/len(weights)
    if len(weights)<= num_highest_allowed:
        count_good_clusters += 1
        all_good_clusters.append(group)
        continue
    
    # if std dev is 0, then all weights are equal, therefore not good
    if std_dev == 0:
        continue
    else:
        z_score_highest = (highest_weight - mean) / std_dev
        if z_score_highest >= number:
            count_good_clusters += 1
            all_good_clusters.append(group)

# visualize as table
edges_data = pd.DataFrame({'Domain': [edge[0] for edge in edge_list],
                           'Client': [edge[1] for edge in edge_list],
                           'Weight': [edge[2] for edge in edge_list],
                           'Weight significance': [edge[2]/max_weight for edge in edge_list]})

data_output = "/Users/ihyunnam/Desktop/kmeans_result_{:.2f}.json".format(number)
with open(data_output, 'w') as output:
    edges_data.to_json(data_output, orient='records', lines=True)

with open("/Users/ihyunnam/Desktop/kmeans_info_{:.2f}.json".format(number), 'w') as output:
    output.write(ujson.dumps("num good clusters" + str(count_good_clusters)) + '\n')
    output.write(ujson.dumps("all good clusters" + str(all_good_clusters)) + '\n')
