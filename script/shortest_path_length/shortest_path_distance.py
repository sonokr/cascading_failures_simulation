import multiprocessing
import os
from itertools import product
from multiprocessing import Pool, Process

import graph_tool as gt
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from lib.utilities import graph_from_netfile, graph_from_netfile_gt


def parallel(n, m, b, is_config, num_of_network):
    if is_config:
        network_dir = f"./data/network/net_n{n}_m{m}_b{b}/"
        result_file_path = f"./data/result/sp_n{n}_m{m}_b{b}_{num_of_network}mean.csv"
    else:
        network_dir = f"./data/network/net_n{n}_m{m}_b{b}_no_config/"
        result_file_path = f"./data/result/sp_n{n}_m{m}_b{b}_{num_of_network}mean_no_config.csv"

    print(f"n={n}\tm={m}\tb={b}\tconfig={is_config}")
    sp_dict = {}
    for i in range(num_of_network):
        net_file_path = network_dir + f"link{i}.net"
        G = graph_from_netfile_gt(net_file_path)
        ave_shortest_path_length = 0
        for s in G.get_vertices():
            ave_shortest_path_length += sum(gt.topology.shortest_distance(G, source=G.vertex(s)).a) / (n * (n - 1))
        sp_dict[f"net{i+1}"] = ave_shortest_path_length

    t = [length for length in sp_dict.values()]
    sp_dict["all"] = [sum(t) / len(t)]
    sp_df = pd.DataFrame.from_dict(sp_dict)
    sp_df.to_csv(result_file_path)

def wrapper(arg):
    parallel(*arg)
    pass

if __name__ == '__main__':
    # pl = Pool(multiprocessing.cpu_count())
    pl = Pool(3)

    num_of_network = [10]
    ns = [100, 1000, 5000, 10000]
    ms = [2, 4]
    bs = [1, 0, -1, -5, -10, -20, -50, -100]
    is_config = [True, False]

    args = product(ns, ms, bs, is_config, num_of_network)

    pl.map(wrapper, args)


