import multiprocessing
from itertools import product
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

from lib.cascade_failure import cascade_failure
from lib.evaluation import efficiency, robustness
from lib.node_removal import (k_max_failure, k_max_localized_failure,
                              l_max_failure, l_max_localized_failure,
                              random_failure, random_localized_failure)
from lib.utilities import graph_from_netfile, graph_from_netfile_gt


def parallel(n, m, b, f, p, config, i):
    print(f"n={n} m={m} b={b} f={f} p={p} config={config} i={i}")

    nr_name = f[0]
    nr_func = f[1]

    if config:
        network_file_path = f"./data/network/net_n{n}_m{m}_b{b}/link{i}.net"
        result_file_path = f"./data/result/cf_n{n}_m{m}_b{b}_{nr_name}_AN{int(p*n)}_{i}.csv"
    else:
        network_file_path = f"./data/network/net_n{n}_m{m}_b{b}_no_config/link{i}.net"
        result_file_path = f"./data/result/cf_n{n}_m{m}_b{b}_{nr_name}_AN{int(p*n)}_no_config_{i}.csv"

    aa = np.arange(0, 1.10, 0.10)

    df = pd.DataFrame()

    for a in aa:
        G = graph_from_netfile(network_file_path)
        initial_node_size = nx.number_of_nodes(G)

        G = cascade_failure(G, a, nr_func, p)
        R = robustness(G, initial_node_size)
        E = efficiency(G)

        df.loc[f"{a:.2f}", f"R"] = R
        df.loc[f"{a:.2f}", f"E"] = E

    df.to_csv(result_file_path)


def wrapper(arg):
    parallel(*arg)


if __name__ == "__main__":
    # pl = Pool(multiprocessing.cpu_count())
    pl = Pool(1)

    num_of_network = 100
    ns = [1000]
    ms = [2, 4]
    bs = [1, 0, -1, -5, -10, -20, -50, -100]
    ps = [p / ns[0] for p in [1, 50, 100]]
    fs = [
        ["random", random_failure],
        ["k_max", k_max_failure],
        ["l_max", l_max_failure],
        ["random_localized", random_localized_failure],
        ["k_max_localized", k_max_localized_failure],
        ["l_max_localized", l_max_localized_failure]
    ]
    config = [True, False]

    args = product(ns, ms, bs, fs, ps, config, range(num_of_network))

    pl.map(wrapper, args)
