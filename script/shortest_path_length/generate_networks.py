import multiprocessing
import os
from itertools import product
from multiprocessing import Pool, Process

import networkx as nx
from tqdm import tqdm

from lib.network_generators import kb_attachment


def test():
    ns = [100, 1000, 5000, 10000]
    ms = [2, 4]
    bs = [1, 0, -1, -5, -10, -20, -50, -100]
    is_config = False

    num_of_network = 100

    for n in ns:
        for m in ms:
            for b in bs:
                savedir = f"./data/network/net_n{n}_m{m}_b{b}/" if is_config else f"./data/network/net_n{n}_m{m}_b{b}_no_config/"
                os.makedirs(savedir, exist_ok=True)
                filepath = savedir + f"link{i}.net"
                print(filepath)
                for i in tqdm(range(100)):
                    if b == 1:
                        G = nx.barabasi_albert_graph(n, m)
                    elif b == 0:
                        G = kb_attachment(n, m, b, False)
                    else:
                        G = kb_attachment(n, m, b, is_config)
                    with open(filepath, "w") as f:
                        f.write(f"*Vertices {n}\n")
                        f.write(f"*Edges {len(G.edges())}\n")
                        for edge in G.edges():
                            f.write(f"{edge[0]+1} {edge[1]+1}\n")

def parallel(index, total, n, m, b, is_config, i):
    savedir = f"./data/network/net_n{n}_m{m}_b{b}/" if is_config else f"./data/network/net_n{n}_m{m}_b{b}_no_config/"
    os.makedirs(savedir, exist_ok=True)
    filepath = savedir + f"link{i}.net"
    print(index/total, filepath)

    if b == 1:
        G = nx.barabasi_albert_graph(n, m)
    elif b == 0:
        G = kb_attachment(n, m, b, False)
    else:
        G = kb_attachment(n, m, b, is_config)
    with open(savedir + f"link{i}.net", "w") as f:
        f.write(f"*Vertices {n}\n")
        f.write(f"*Edges {len(G.edges())}\n")
        for edge in G.edges():
            f.write(f"{edge[0]+1} {edge[1]+1}\n")


def wrapper(args):
    parallel(*args)


def main():
    pl = Pool(multiprocessing.cpu_count())

    ns = [100, 1000, 5000, 10000]
    ms = [2, 4]
    bs = [1, 0, -1, -5, -10, -20, -50, -100]
    is_config = [False]
    num_of_network = 100
    args_tmp = list(product(ns, ms, bs, is_config, range(num_of_network)))

    args = []
    for i, arg in enumerate(args_tmp):
        args.append([i, len(args_tmp), *arg])

    pl.map(wrapper, args)


if __name__ == "__main__":
    main()
