import os

import networkx as nx
from tqdm import tqdm

from lib.network_generators import kb_attachment

ns = [100, 1000, 5000, 10000]
num_of_network = 100

ER_dir = "./data/net/ER/"

for n in ns:
    print(f"Number of node: {n}")
    n_dir = ER_dir + f"n{n}/"
    os.makedirs(n_dir, exist_ok=True)

    p = 3.9 / n

    G = nx.erdos_renyi_graph(n, p)
    print(G)
    print(nx.is_connected(G))

    i = 0
    j = 1
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            with open(n_dir + f"g_{p}_{i}.net", "w") as f:
                f.write(f"*Vertices {n}\n")
                f.write(f"*Edges {len(G.edges())}\n")
                for edge in G.edges():
                    f.write(f"{edge[0]+1} {edge[1]+1}\n")
            print(f"{i}/{j}")
            i = i + 1
            j = j + 1
        if i == num_of_network:
            break
    # for i in tqdm(range(num_of_network)):
    #     G = nx.erdos_renyi_graph(n, p)
    #     with open(n_dir + f"g_{p}_{i}.net", "w") as f:
    #         f.write(f"*Vertices {n}\n")
    #         f.write(f"*Edges {len(G.edges())}\n")
    #         for edge in G.edges():
    #             f.write(f"{edge[0]+1} {edge[1]+1}\n")
