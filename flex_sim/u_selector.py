from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform

def kmeans_selector(features, num_select, seed):
    kmeans = KMeans(n_clusters=num_select, n_init='auto', random_state=seed).fit(features)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    selected = []
    for i in range(num_select):
        cluster_idx = np.where(labels==i)[0]
        center = centroids[i]
        closest = min(cluster_idx, key=lambda idx: np.linalg.norm(features[idx] - center))
        selected.append(closest)
    
    return selected

def selector(frames, number, sort, species=['H', 'C', 'N', 'O'], seed=1314):
    desc = SOAP(species=species, r_cut=7.0, n_max=6, l_max=6, sigma=0.2, periodic=True, compression={"mode":"off"}, sparse=False)
    all_feats = [desc.create(f) for f in frames]
    flt_feats = np.array([np.mean(x, axis=0) for x in all_feats])
    selected = kmeans_selector(flt_feats, number, seed)
    
    selected_frames = [frames[i] for i in selected]
    
    if not sort:
        return selected_frames, selected
    else:
        G, _ = build_graph(flt_feats[selected])
        order = mst_dfs_tsp(flt_feats[selected])
        ordered_frames = [selected_frames[i] for i in order]
        ordered_selected = [selected[i] for i in order]
        return ordered_frames, ordered_selected


def selector_tail(frames, number, sort, species=['H', 'C', 'N', 'O'], seed=1314, n_tail=3):
    desc = SOAP(
        species=species,
        r_cut=7.0,
        n_max=6,
        l_max=6,
        sigma=0.2,
        periodic=True,
        compression={"mode": "off"},
        sparse=False
    )

    all_feats = [desc.create(f) for f in frames]

    flt_feats = []
    for x in all_feats:
        if x.shape[0] <= n_tail:
            raise ValueError(f"每个结构至少应包含 {n_tail + 1} 个原子")
        mean_tail = np.mean(x[-n_tail:], axis=0)   # 倒数 n_tail 个原子
        mean_rest = np.mean(x[:-n_tail], axis=0)   # 其余原子
        combined = mean_tail + mean_rest           # 可改为加权或平均
        flt_feats.append(combined)
    flt_feats = np.array(flt_feats)

    selected = kmeans_selector(flt_feats, number, seed)
    selected_frames = [frames[i] for i in selected]

    if not sort:
        return selected_frames, selected
    else:
        G, _ = build_graph(flt_feats[selected])
        order = mst_dfs_tsp(flt_feats[selected])
        ordered_frames = [selected_frames[i] for i in order]
        ordered_selected = [selected[i] for i in order]
        return ordered_frames, ordered_selected


def build_graph(points):
    dist_matrix = squareform(pdist(points, metric='euclidean'))
    G = nx.Graph()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    return G, dist_matrix

def mst_dfs_tsp(points, start_node=0):
    G, _ = build_graph(points)
    # 构造最小生成树
    T = nx.minimum_spanning_tree(G, weight='weight')
    # DFS遍历生成树
    visited = list(nx.dfs_preorder_nodes(T, source=start_node))
    return visited
