from locale import normalize
from pprint import pprint

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def edge_betweenness_centrality(G, normalized=False):
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    nodes = G
    for s in tqdm(nodes):
        # single source shortest pathsi
        S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        # accumulation
        betweenness = _accumulate_edges(betweenness, S, P, sigma, s)
    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = _rescale_e(
        betweenness, len(G), normalized=normalized, directed=G.is_directed()
    )
    return betweenness


def _single_source_shortest_path_basic(G, s):
    # 探索対象のリスト
    S = []  # 空のスタック.

    # それぞれのノードの先人ノードを記録する
    P = {}  # 空のリスト.
    for v in G:
        P[v] = []  # 全てのノードについて空リスト.

    # 多分先生が言ってた1を流すやつ
    sigma = dict.fromkeys(G, 0.0)  # 全てのノードについて，sigma[v]=0.
    sigma[s] = 1.0  # 現在のノードsでは1.0

    # ホップ数で測ったときの距離のことかな
    D = {}  # ノードsについては0, それ以外のノードtは-1.
    D[s] = 0

    # 探索するノード
    Q = [s]  # キュー

    while Q:  # 最短パスに幅優先探索を用いる
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:  # ノードvの近接ノードwに関するループ
            # まだ調べていないノード
            # Dはグローバル
            if w not in D:
                Q.append(w)  # 未探索ノードなのでキューに追加
                D[w] = Dv + 1  # 隣接ノードなのでホップ差は1
            # ホップ数の差が1なら最小パスといえるので，カウントする
            # ここで最小パスが確定する?
            if D[w] == Dv + 1:
                sigma[w] += sigmav  # シグマを流す
                P[w].append(v)  # 一つ前の, 元となるノードを記録する
    return S, P, sigma, D


def _accumulate_edges(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            if (v, w) not in betweenness:
                betweenness[(w, v)] += c
            else:
                betweenness[(v, w)] += c
            delta[v] += c
        if w != s:
            betweenness[w] += delta[w]
    return betweenness


def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale

    return betweenness


def rescale_e(bc, n):
    scale = 2 / (n * (n - 1))
    return [bc_ * scale for bc_ in bc]


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


def brandes_betweenness_centrality(G, normalized=False):
    betweenness = dict.fromkeys(G, 0.0)
    nodes = G
    for s in tqdm(nodes):
        # 何に関しての最短経路を見つけるの？
        S, P, sigma, _ = _single_source_shortest_path_basic(G, s)

        # 現在のノードsに関しての媒介中心性を計算する？
        betweenness, delta = _accumulate_basic(betweenness, S, P, sigma, s)

    # スケールを調整する
    betweenness = _rescale(
        betweenness=betweenness,
        n=len(G),
        normalized=normalized,
    )

    return betweenness


def _single_source_shortest_path_basic(G, s):
    # 探索対象のリスト
    S = []  # 空のスタック.

    # それぞれのノードの先人ノードを記録する
    P = {}  # 空のリスト.
    for v in G:
        P[v] = []  # 全てのノードについて空リスト.

    # 多分先生が言ってた1を流すやつ
    sigma = dict.fromkeys(G, 0.0)  # 全てのノードについて，sigma[v]=0.
    sigma[s] = 1.0  # 現在のノードsでは1.0

    # ホップ数で測ったときの距離のことかな
    D = {}  # ノードsについては0, それ以外のノードtは-1.
    D[s] = 0

    # 探索するノード
    Q = [s]  # キュー

    while Q:  # 最短パスに幅優先探索を用いる
        v = Q.pop(0)
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:  # ノードvの近接ノードwに関するループ
            # まだ調べていないノード
            # Dはグローバル
            if w not in D:
                Q.append(w)  # 未探索ノードなのでキューに追加
                D[w] = Dv + 1  # 隣接ノードなのでホップ差は1
            # ホップ数の差が1なら最小パスといえるので，カウントする
            # ここで最小パスが確定する?
            if D[w] == Dv + 1:
                sigma[w] += sigmav  # シグマを流す
                P[w].append(v)  # 一つ前の, 元となるノードを記録する
    return S, P, sigma, D


def _accumulate_basic(betweenness, S, P, sigma, s):
    delta = dict.fromkeys(S, 0)
    while S:
        w = S.pop()
        coeff = (1 + delta[w]) / sigma[w]
        for v in P[w]:
            delta[v] += sigma[v] * coeff
        if w != s:
            betweenness[w] += delta[w]
    return betweenness, delta


def _rescale(betweenness, n, normalized):
    if normalized:
        if n <= 2:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / ((n - 1) * (n - 2))
    else:  # 同じノードの組み合わせを2回数えるので補正する
        scale = 0.5

    if scale is not None:
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def rescale(bc):
    n = len(bc)
    scale = 2 / ((n - 1) * (n - 2))
    for i in range(len(bc)):
        bc[i] *= scale
    return bc


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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--number_of_network")
    # args = parser.parse_args()
    # number_of_network = int(args.number_of_network)

    number_of_network = int(input("number of network: "))

    files = [f"./kb{i+1}.net" for i in range(number_of_network)]
    for i, file in enumerate(files):
        with open(file, "r") as f:
            lines = [l.replace("\n", "").split() for l in f.readlines()]
            G = nx.Graph()
            for j, edge in enumerate(lines):
                if j == 0 or j == 1:
                    continue
                G.add_edge(edge[0], edge[1])

        bc = list(
            dict(
                sorted(
                    brandes_betweenness_centrality(G).items(), key=lambda x: int(x[0])
                )
            ).values()
        )
        with open(f"./data/bc{i+1}.txt", "w") as f:
            for bc_ in bc:
                f.write(f"{bc_}\n")

        bc_normd = rescale(bc)
        with open(f"./data/normd_bc{i+1}.txt", "w") as f:
            for bc_normd_ in bc_normd:
                f.write(f"{bc_normd_}\n")

        fd_bc = frequency_distribution(bc)
        x = list(fd_bc["class_value"])
        y = list(fd_bc["relative_frequency"])
        df_fd_dc = pd.DataFrame(
            [xy for xy in zip(x, y)], columns=["class_value", "relative_frequency"]
        )
        df_fd_dc.to_csv(
            f"./data/fd_bc{i}.txt",
            index=False,
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
