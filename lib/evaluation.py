import networkx as nx


def robustness(G, initial_node_size):
    try:
        lc = len(max(nx.connected_components(G), key=len, default=0))
    except TypeError:
        lc = max(nx.connected_components(G), key=len, default=0)

    return lc / initial_node_size


def efficiency(G):
    shortest_path_length_dict = nx.shortest_path_length(G)

    sum_ = 0
    for spl in shortest_path_length_dict:
        origin, targets = spl
        for i, l in targets.items():
            if l == 0:
                continue
            sum_ += 1 / l

    N = nx.number_of_nodes(G)
    if N == 0:
        return 0

    return (1 / (N * (N - 1))) * (sum_)


if __name__ == "__main__":
    G = nx.barabasi_albert_graph(1000, 4)
    print(efficiency(G))
