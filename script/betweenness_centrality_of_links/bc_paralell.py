import multiprocessing
from itertools import product
from multiprocessing import Pool

import graph_tool as gt
import networkx as nx

from lib.utilities import graph_from_netfile_gt


def parallel(n, m, b, config, i):
    if config:
        network_file_path = f"./data/network/net_n{n}_m{m}_b{b}/link{i}.net"
        result_file_path = f"./data/result/bc_n{n}_m{m}_b{b}_{i}.csv"
    else:
        network_file_path = f"./data/network/net_n{n}_m{m}_b{b}_no_config/link{i}.net"
        result_file_path = f"./data/result/bc_n{n}_m{m}_b{b}_no_config_{i}.csv"

    print(f"n={n}\tm={m}\tb={b}\tconfig={config}")

    G = graph_from_netfile_gt(network_file_path)
    bc = gt.centrality.betweenness(G)
    with open(result_file_path, "w") as f:
        for bc_ in bc[1]:  # Edge betweenness
            f.write(f"{bc_}\n")


def wrapper(arg):
    parallel(*arg)


if __name__ == "__main__":
    pl = Pool(3)

    num_of_network = 100
    ns = [1000]
    ms = [2, 4]
    bs = [1, 0, -1, -5, -10, -20, -50, -100]
    config = [True, False]

    args = product(ns, ms, bs, config, range(num_of_network))

    pl.map(wrapper, args)
