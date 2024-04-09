import multiprocessing
from itertools import product
from multiprocessing import Pool

import graph_tool as gt
import networkx as nx

from lib.utilities import graph_from_netfile_gt

num_of_network = 100
for i in range(num_of_network):
    print(i)
    network_path = f"./data/network/ER/n1000/g_0.003974_{i}.net"
    result_path = f"./data/result/ER/bc_{i}.csv"
    G = graph_from_netfile_gt(network_path)
    bc = gt.centrality.betweenness(G)
    with open(result_path, "w") as f:
        for bc_ in bc[1]:  # Edge betweenness
            f.write(f"{bc_}\n")

