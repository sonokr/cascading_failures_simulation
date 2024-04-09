import time

import networkx as nx
import numpy as np

from lib.configuration_model import configuration_model


def kb_attachment(n, m, b=1):
    if m < 1 or m >= n:
        raise nx.NetworkXError(
            f"Barabási-Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
        )

    G = nx.star_graph(m)

    nodes = [n for n, _ in G.degree()]
    weights = [d ** b for _, d in G.degree()]

    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(nodes, m, weights)
        # 元のグラフにm本のエッジを追加する.
        G.add_edges_from(zip([source] * m, targets))

        nodes = [n for n, _ in G.degree()]
        weights = [d ** b for _, d in G.degree()]

        source += 1

    # return G
    return configuration_model(G)


def _random_subset(nodes, m, weights):
    targets = set()

    sum_weights = sum(weights)
    weights_normalized = [w / sum_weights for w in weights]

    xs = np.random.choice(nodes, m, replace=False, p=weights_normalized)
    for x in xs:
        targets.add(x)
    return targets


def main():
    start = time.time()

    node_size = int(input("node size >> "))
    m = int(input("m >> "))
    b = float(input("b >> "))
    number_of_network = int(input("number of network >> "))

    for i in range(number_of_network):
        G = kb_attachment(node_size, m, b)
        print(nx.info(G))

        with open(f"./kb_n{node_size}_m{m}_b{b}.net", "w") as f:
            f.write(f"*Vertices {node_size}\n")
            f.write(f"*Edges {len(G.edges())}\n")
            for edge in G.edges():
                f.write(f"{edge[0]+1} {edge[1]+1} 1\n")

    print(f"Elapsed time: {time.time() - start}[s]")


if __name__ == "__main__":
    main()
