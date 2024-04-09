import networkx as nx
import numpy as np
import pandas as pd


def graph_from_netfile(path):
    with open(path, "r") as f:
        lines = [l.replace("\n", "").split() for l in f.readlines()]
        G = nx.Graph()
        for i, edge in enumerate(lines):
            if i == 0 or i == 1:
                continue
            G.add_edge(edge[0], edge[1])
    return G


def frequency_distribution(data, class_width=None):
    data = np.asarray(data)
    if class_width is None:
        class_size = int(np.log2(data.size).round()) + 1
        class_width = (data.max() - data.min()) / class_size

    bins = np.arange(0, data.max() + class_width + 1, class_width)
    hist = np.histogram(data, bins)[0]
    cumsum = hist.cumsum()

    return pd.DataFrame(
        {
            "class_value": (bins[1:] + bins[:-1]) / 2,
            "frequency": hist,
            "cumulative_frequency": cumsum,
            "relative_frequency": hist / cumsum[-1],
            "cumulative_relative_frequency": cumsum / cumsum[-1],
        },
        index=pd.Index(
            [f"{bins[i]}以上{bins[i+1]}未満" for i in range(hist.size)], name="階級"
        ),
    )
