import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from evaluation import efficiency, robustness
from lib import graph_from_netfile
from node_removal import k_max_failure


def cascade_failure(G, a, initial_failure, p):
    """_summary_

    Args:
        G (networkx.classes.graph.Graph): 初期グラフ
        a (float): 耐久性パラメータ
        initial_failure (function): 初期故障ノードを除去する関数
        p (float): 初期故障のノード除去率

    Returns:
        networkx.classes.graph.Graph: カスケード故障収束後のグラフ
    """
    load_i = nx.betweenness_centrality(G, normalized=False)
    capa_i = {k: (1 + a) * float(v) for k, v in load_i.items()}

    G = initial_failure(G, p)

    prev_node_size = nx.number_of_nodes(G)
    while True:
        load_i = nx.betweenness_centrality(G, normalized=False)
        nodes = list(load_i.keys())

        removed_node_count = 0
        for n in nodes:
            if load_i[n] > capa_i[n]:
                G.remove_node(n)
                removed_node_count += 1

        # カスケード故障が収束
        if prev_node_size == nx.number_of_nodes(G):
            break
        # ノード数が0になる
        if nx.number_of_nodes(G) <= 0:
            break

        prev_node_size = nx.number_of_nodes(G)

    return G


def main():
    print("start program")

    netdir = "./data/network/"
    outdir = "./data/test/"

    aa = np.arange(0, 1.1, 0.1)
    aa = [0.5]

    number_of_network = 10
    filename_list = [f"ba{i+1}.net" for i in range(number_of_network)]
    netfile_path_list = [netdir + fn for fn in filename_list]

    name = "k_max"
    func = k_max_failure

    df = pd.DataFrame(
        {
            "R_0": [0 for _ in range(len(aa))],
            "R_1": [0 for _ in range(len(aa))],
            "R_2": [0 for _ in range(len(aa))],
            "R_3": [0 for _ in range(len(aa))],
            "R_4": [0 for _ in range(len(aa))],
            "R_5": [0 for _ in range(len(aa))],
            "R_6": [0 for _ in range(len(aa))],
            "R_7": [0 for _ in range(len(aa))],
            "R_8": [0 for _ in range(len(aa))],
            "R_9": [0 for _ in range(len(aa))],
            "R_mean": [0 for _ in range(len(aa))],
            "E_0": [0 for _ in range(len(aa))],
            "E_1": [0 for _ in range(len(aa))],
            "E_2": [0 for _ in range(len(aa))],
            "E_3": [0 for _ in range(len(aa))],
            "E_4": [0 for _ in range(len(aa))],
            "E_5": [0 for _ in range(len(aa))],
            "E_6": [0 for _ in range(len(aa))],
            "E_7": [0 for _ in range(len(aa))],
            "E_8": [0 for _ in range(len(aa))],
            "E_9": [0 for _ in range(len(aa))],
            "E_mean": [0 for _ in range(len(aa))],
        },
        index=[f"{a:.2f}" for a in aa],
    )

    # 最大連結成分比とネットワークの効率を計算する
    # ネットワークファイルごと
    for i, f in enumerate(netfile_path_list):
        # αの値
        for a in aa:
            print(f"{name} i={i} a={a:.2f} {f}")

            G = graph_from_netfile(f)
            print(f"\tnetwork loaded: \t\t{nx.info(G)}")

            initial_node_size = nx.number_of_nodes(G)

            G = cascade_failure(G, a, func)

            R = robustness(G, initial_node_size)
            E = efficiency(G)

            df.loc[f"{a:.2f}", f"R_{i}"] = R
            df.loc[f"{a:.2f}", f"E_{i}"] = E

            print(f"\tR={R:.6f} E={E:.6f}")

    # 10回計算した平均を計算する
    for a in aa:
        R_sum = 0
        E_sum = 0
        for i in range(10):
            R_sum += df.loc[f"{a:.2f}", f"R_{i}"]
            E_sum += df.loc[f"{a:.2f}", f"E_{i}"]
        R_mean = R_sum / 10
        E_mean = E_sum / 10
        df.loc[f"{a:.2f}", "R_mean"] = R_mean
        df.loc[f"{a:.2f}", "E_mean"] = E_mean

    print(df)
    df.to_csv(outdir + f"result_{name}_ba.csv")

    # グラフの設定
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.grid"] = True
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.handlelength"] = 0.7
    plt.rcParams["legend.labelspacing"] = 0
    plt.rcParams["legend.handletextpad"] = 0.8  # 凡例の線と文字の距離の長さ
    plt.rcParams["legend.markerscale"] = 1  # 点がある場合のmarker scale
    plt.rcParams["legend.fontsize"] = 15

    kusunoki_k_max = pd.read_csv("./data/test/result_k_max_ba.csv")

    # 最大連結成分比
    fig, ax = plt.subplots()

    fig.subplots_adjust(bottom=0.2, left=0.18, top=0.99, right=0.96)

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("G")

    ax.plot(
        aa,
        kusunoki_k_max["R_mean"],
        marker="^",
        label=r"$\rm{kusunoki}$",
    )

    plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
