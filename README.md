# What is Clid?
This repository contains the source code for Clid, a Transport Layer Security (TLS) client identification tool based on unsupervised learning on domain names in the server name indication (SNI) field. Clid aims to provide some information on a wide range of clients, even though it may not be able to identify a definitive characteristic about each one of the clients. This is a different approach from that of many existing rule-based client identification tools that rely on hardcoded databases to identify granular characteristics of a few clients.

# Clid Preprint
https://arxiv.org/pdf/2410.02040

# How does Clid work?
For this research, we utilize some 345 million anonymized TLS handshakes collected from a large university campus network. From each handshake, we create a TCP fingerprint that identifies each unique client that corresponds to a physical device on the network. Clid uses Bayesian optimization to find the 'optimal' DBSCAN clustering of clients and domain names for a set of TLS connections. Clid maps each client cluster to one or more domain clusters that are most strongly associated with it based on the frequency and exclusivity of their TLS connections. While learning highly associated domain names of a client may not immediately tell us specific characteristics of the client like its the operating system, manufacturer, or TLS configuration, it may serve as a strong first step to doing so.

# Preformance
We evaluate Clid's performance on various subsets of our captured TLS handshakes and on different parameter settings that affect the granularity of identification results. Our experiments show that Clid is able to identify 'strongly associated' domain names for at least 60% of all clients in all our experiments.
