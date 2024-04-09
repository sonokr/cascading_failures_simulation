import sys
from operator import is_

import networkx as nx
import numpy as np

from lib.dbl_edge_mcmc import *
from lib.utilities import graph_from_netfile


def cm(G):
    if nx.number_of_edges(G) % 2 != 0:
        print("Number of edges is not even.")

    L = [i for i, d in nx.degree(G) for _ in range(d)]

    new_G = nx.Graph()

    while L:
        i, j = np.random.choice(L, 2)

        if i == j:  # 自己ループ
            continue
        if new_G.has_edge(i, j):  # 多重ループ
            continue

        new_G.add_edge(i, j)

        L.remove(i)
        L.remove(j)

        if len(L) == 2:
            # 残った２つが自己ループか多重ループ
            if L[0] == L[1] or new_G.has_edge(L[0], L[1]):
                return None
        elif len(L) == 1:
            return None

    dd1 = sorted(nx.degree(G))
    dd2 = sorted(nx.degree(new_G))

    assert dd1 == dd2

    return new_G


def configuration_model(G):
    while True:
        new_G = cm(G)
        if new_G:
            break

    dd1 = sorted(nx.degree(G))
    dd2 = sorted(nx.degree(new_G))

    assert dd1 == dd2

    return new_G


def edge_swapping(G, n):
    MC = MCMC_class(G, allow_loops=False, allow_multi=False, is_v_labeled=False)
    for _ in range(1):
        G_new = MC.step_and_get_graph()
    mapping = dict(zip(G_new, map(str, range(1, nx.number_of_nodes(G_new) + 1))))
    G_new = nx.relabel_nodes(G_new, mapping)
    return G_new


def main():
    netfile_path = "./data/kb1.net"

    G = configuration_model(graph_from_netfile(netfile_path))

    bc_e = list(nx.edge_betweenness_centrality(G, normalized=True).values())
    with open("./data/bc_e_nx.txt", "w") as f:
        for bc_e_ in bc_e:
            f.write(f"{bc_e_}\n")


if __name__ == "__main__":
    main()
