from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def random_failure(G, p):
    for _ in range(int(nx.number_of_nodes(G) * p)):
        remove_node_index = _get_random_node(G)
        G.remove_node(remove_node_index)
    return G


def _get_random_node(G):
    return np.random.choice([i for i, _ in nx.degree(G)])


def k_max_failure(G, p):
    for _ in range(int(nx.number_of_nodes(G) * p)):
        remove_node_index = _get_k_max_node(G)
        # dd = nx.degree(G)
        # k_max = np.max([int(k) for _, k in dd])
        # k_max_indexes = [i for i, k in dd if k == k_max]
        # remove_node_index = np.random.choice(k_max_indexes)
        G.remove_node(f"{remove_node_index}")
    return G


def _get_k_max_node(G):
    dd = nx.degree(G)
    k_max = np.max([int(k) for _, k in dd])
    k_max_indexes = [i for i, k in dd if k == k_max]
    remove_node_index = np.random.choice(k_max_indexes)
    return remove_node_index


def l_max_failure(G, p):
    for _ in range(int(nx.number_of_nodes(G) * p)):
        remove_node_index = _get_l_max_node(G)
        # bc = nx.betweenness_centrality(G=G, normalized=True)
        # bc_max = np.max([bc_ for _, bc_ in bc.items()])
        # bc_max_indexes = [i for i, bc_ in bc.items() if bc_ == bc_max]
        # remove_node_index = np.random.choice(bc_max_indexes)
        G.remove_node(f"{remove_node_index}")
    return G


def _get_l_max_node(G):
    bc = nx.betweenness_centrality(G=G, normalized=True)
    bc_max = np.max([bc_ for _, bc_ in bc.items()])
    bc_max_indexes = [i for i, bc_ in bc.items() if bc_ == bc_max]
    remove_node_index = np.random.choice(bc_max_indexes)
    return remove_node_index


def bp_attack(G):
    remove_node_indexes = _get_bp_nodes(G)
    for n in remove_node_indexes:
        G.remove_node(f"{n}")
    return G


def _get_bp_nodes(G):
    remove_node_indexes = []
    return remove_node_indexes


def random_localized_failure(G, p):
    number_of_removal_nodes = int(nx.number_of_nodes(G) * p)

    initial_removal_node = _get_random_node(G)

    spl = dict(nx.all_pairs_dijkstra_path_length(G))[initial_removal_node]

    length_from_root_node = sorted(spl.items(), key=lambda i: i[1])
    removal_nodes = [
        length_from_root_node[i][0] for i in range(1, number_of_removal_nodes)
    ]
    removal_nodes.append(initial_removal_node)
    for n in removal_nodes:
        G.remove_node(n)

    return G


def k_max_localized_failure(G, p):
    number_of_removal_nodes = int(nx.number_of_nodes(G) * p)

    initial_removal_node = _get_k_max_node(G)

    spl = dict(nx.all_pairs_dijkstra_path_length(G))[initial_removal_node]

    length_from_root_node = sorted(spl.items(), key=lambda i: i[1])
    removal_nodes = [
        length_from_root_node[i][0] for i in range(1, number_of_removal_nodes)
    ]
    removal_nodes.append(initial_removal_node)
    for n in removal_nodes:
        G.remove_node(n)

    return G


def l_max_localized_failure(G, p):
    number_of_removal_nodes = int(nx.number_of_nodes(G) * p)

    initial_removal_node = _get_l_max_node(G)

    spl = dict(nx.all_pairs_dijkstra_path_length(G))[initial_removal_node]

    length_from_root_node = sorted(spl.items(), key=lambda i: i[1])
    removal_nodes = [
        length_from_root_node[i][0] for i in range(1, number_of_removal_nodes)
    ]
    removal_nodes.append(initial_removal_node)
    for n in removal_nodes:
        G.remove_node(n)

    return G


if __name__ == "__main__":
    G = nx.barabasi_albert_graph(20, 2)

    dd = nx.degree(G)
    k_max = np.max([int(k) for _, k in dd])
    k_max_node = np.max([int(k) for _, k in dd])
    bc = nx.betweenness_centrality(G, normalized=False)
    bc_max = np.max([bc_ for _, bc_ in bc.items()])
    l_max_node = np.max([bc_ for _, bc_ in bc.items()])

    print(f"k_max {k_max_node}")
    print(f"l_max {l_max_node:.2f}")

    G_la = random_localized_failure(deepcopy(G), 0.6)
    print(f"number of remain nodes: {nx.number_of_nodes(G_la)}")

    pos = nx.spring_layout(G, k=0.8)
    nx.draw_networkx(G, pos)
    plt.show()
