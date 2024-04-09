import multiprocessing
import os
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
from lib.utilities import graph_from_netfile


def parallel(i, n, f, p):
    print(f"i={i} Nodes {n} {f[0]} rate p={p}")

    nr_name = f[0]
    nr_func = f[1]

    network_file_path = f"./data/ER/net/n1000/g_0.003974_{i}.net"
    result_dir = f"./data/ER/res/n1000/{nr_name}/AN{int(n*p)}/"
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = result_dir + f"g_0.003974_{i}.csv"

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
    pl = Pool(multiprocessing.cpu_count())
    # pl = Pool(1)

    nn = range(100)
    ns = [1000]
    ps = [p / ns[0] for p in [1, 50, 100]]
    fs = [
        ["random", random_failure],
        ["k_max", k_max_failure],
        ["l_max", l_max_failure],
        ["random_localized", random_localized_failure],
        ["k_max_localized", k_max_localized_failure],
        ["l_max_localized", l_max_localized_failure]
    ]

    args = product(nn, ns, fs, ps)

    pl.map(wrapper, args)
