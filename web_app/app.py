import io
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix

from sklearn.datasets import (
    make_moons,
    make_circles,
    make_blobs,
    load_iris,
    load_wine,
    load_breast_cancer,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix

warnings.filterwarnings("ignore")

# Global default seed matching the research notebook
SEED = 42

# Consistent discrete color palette for clusters (tab10-inspired hex codes)
CLUSTER_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
NOISE_COLOR = "#222222"

METHOD_ORDER = [
    "MST (Top k-1)",
    "MST (Inconsistency)",
    "Single-Linkage",
    "DBSCAN",
    "Spectral Clustering",
    "k-Means",
    "k-Medoids (PAM)",
    "GMM",
]

DATASET_DEFAULT_K = {
    "Two Moons": 2,
    "Concentric Circles": 2,
    "Two Spirals": 2,
    "Anisotropic Blobs": 3,
    "Iris (4D → PCA 2D)": 3,
    "Wine (13D → PCA 2D)": 3,
    "Breast Cancer (30D → PCA 2D)": 2,
    "Custom CSV Upload": 3,
}

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="MST-Based Clustering — Interactive Viva & Demo",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .viva-card {
        background-color: rgba(128, 128, 128, 0.06);
        border-left: 4px solid #1f77b4;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .viva-card-warn {
        background-color: rgba(214, 39, 40, 0.06);
        border-left: 4px solid #d62728;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .viva-card-success {
        background-color: rgba(44, 160, 44, 0.06);
        border-left: 4px solid #2ca02c;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 1. Core Algorithms & Upgraded MST Capabilities
# ==============================================================================
def make_spirals(n_samples: int = 300, noise: float = 0.04, random_state: int = SEED):
    """
    Generates two intertwined Archimedean spirals in 2D space.
    Points are sampled along angle theta so the inner tips remain separated
    while the arms wrap around each other.
    """
    rng = np.random.RandomState(random_state)
    n_per_arm = n_samples // 2

    theta = np.sqrt(rng.uniform(0.15, 1.0, n_per_arm)) * 2.5 * np.pi
    r = theta

    x1 = r * np.cos(theta) + rng.normal(0, noise * 8.0, n_per_arm)
    y1 = r * np.sin(theta) + rng.normal(0, noise * 8.0, n_per_arm)

    x2 = -r * np.cos(theta) + rng.normal(0, noise * 8.0, n_per_arm)
    y2 = -r * np.sin(theta) + rng.normal(0, noise * 8.0, n_per_arm)

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_per_arm, dtype=int), np.ones(n_per_arm, dtype=int)])
    return X, y


@st.cache_data(show_spinner=False)
def generate_all_datasets(n_samples: int = 300, noise: float = 0.05, random_state: int = SEED):
    """
    Generates and standardizes the 4 synthetic benchmark datasets along with ground-truth k.
    """
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_circles, y_circles = make_circles(
        n_samples=n_samples, factor=0.45, noise=noise, random_state=random_state
    )
    X_spirals, y_spirals = make_spirals(n_samples=n_samples, noise=noise, random_state=random_state)

    X_raw_blobs, y_blobs = make_blobs(
        n_samples=n_samples,
        centers=[[-3, -2], [0, 2], [3, -1]],
        cluster_std=0.6 + 6.0 * noise,
        random_state=random_state,
    )
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X_aniso = np.dot(X_raw_blobs, transformation)

    datasets = {
        "Two Moons": (StandardScaler().fit_transform(X_moons), y_moons, 2),
        "Concentric Circles": (StandardScaler().fit_transform(X_circles), y_circles, 2),
        "Two Spirals": (StandardScaler().fit_transform(X_spirals), y_spirals, 2),
        "Anisotropic Blobs": (StandardScaler().fit_transform(X_aniso), y_blobs, 3),
    }
    return datasets


@st.cache_data(show_spinner=False)
def load_real_world_dataset(name: str):
    """
    Loads real-world multi-dimensional datasets from scikit-learn, standardizes them in R^d,
    and projects to 2D via PCA for visual inspection while preserving full R^d for MST construction.
    """
    if name.startswith("Iris"):
        bunch = load_iris()
        k_true = 3
    elif name.startswith("Wine"):
        bunch = load_wine()
        k_true = 3
    else:
        bunch = load_breast_cancer()
        k_true = 2

    X_high = StandardScaler().fit_transform(bunch.data)
    y_true = bunch.target.astype(int)
    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X_high)
    var_ratio = pca.explained_variance_ratio_
    axis_labels = (
        f"PC1 ({var_ratio[0]*100:.1f}% var)",
        f"PC2 ({var_ratio[1]*100:.1f}% var)",
    )
    return X_high, X_2d, y_true, k_true, axis_labels


@st.cache_data(show_spinner=False)
def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Computes the pairwise Euclidean distance matrix D where D[i, j] = ||x_i - x_j||_2."""
    return cdist(X, X, metric="euclidean")


@st.cache_data(show_spinner=False)
def compute_mutual_reachability_matrix(X: np.ndarray, m_neighbors: int = 5) -> np.ndarray:
    """
    Computes HDBSCAN's Mutual Reachability Distance matrix:
      d_mreach(x_i, x_j) = max(core_m(x_i), core_m(x_j), ||x_i - x_j||_2)
    where core_m(x_i) is the Euclidean distance from x_i to its m-th nearest neighbor.
    Pushes sparse bridge/noise points away from dense manifolds to prevent chaining.
    """
    D = compute_distance_matrix(X)
    n = D.shape[0]
    m_eff = min(max(1, m_neighbors), n - 1)
    # Sort each row to find distance to m_eff-th nearest neighbor (index m_eff since index 0 is self=0)
    sorted_d = np.sort(D, axis=1)
    core_distances = sorted_d[:, m_eff]
    core_max = np.maximum(core_distances[:, None], core_distances[None, :])
    D_mreach = np.maximum(D, core_max)
    np.fill_diagonal(D_mreach, 0.0)
    return D_mreach


@st.cache_data(show_spinner=False)
def prims_mst_from_scratch(D: np.ndarray):
    """
    Dense O(n^2) implementation of Prim's Algorithm for complete graphs.
    Returns:
      edges: numpy array of shape (n-1, 3) where each row is [u, v, weight] in greedy addition order
      total_weight: sum of all n-1 MST edge weights
    """
    n = D.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    key[0] = 0.0
    edges = []

    for _ in range(n):
        masked_keys = np.where(in_mst, np.inf, key)
        u = int(np.argmin(masked_keys))
        in_mst[u] = True

        if parent[u] != -1:
            edges.append((parent[u], u, key[u]))

        unvisited_mask = ~in_mst
        better_edge_mask = unvisited_mask & (D[u] < key)
        parent[better_edge_mask] = u
        key[better_edge_mask] = D[u, better_edge_mask]

    edges = np.array(edges, dtype=float)
    total_weight = float(np.sum(edges[:, 2]))
    return edges, total_weight


@st.cache_data(show_spinner=False)
def sparse_knn_mst(X: np.ndarray, n_neighbors: int = 15):
    """
    Scalable O(N log N) Sparse k-NN Graph MST construction.
    Builds a symmetric k-NN distance graph; if disconnected, bridges components via nearest pairs
    before extracting the exact Minimum Spanning Tree.
    """
    n = len(X)
    k_eff = min(max(2, n_neighbors), n - 1)
    knn_csr = kneighbors_graph(X, n_neighbors=k_eff, mode="distance", include_self=False)
    knn_sym = knn_csr.maximum(knn_csr.T)

    n_comp, comp_labels = connected_components(knn_sym, directed=False)
    if n_comp > 1:
        # Connect disconnected k-NN components with shortest inter-component bridges
        D_full = compute_distance_matrix(X)
        knn_lil = knn_sym.tolil()
        for c1 in range(n_comp):
            idx1 = np.where(comp_labels == c1)[0]
            idx2 = np.where(comp_labels != c1)[0]
            sub_d = D_full[np.ix_(idx1, idx2)]
            flat_min = int(np.argmin(sub_d))
            u = int(idx1[flat_min // len(idx2)])
            v = int(idx2[flat_min % len(idx2)])
            w = float(D_full[u, v])
            knn_lil[u, v] = w
            knn_lil[v, u] = w
        knn_sym = knn_lil.tocsr()

    mst_sparse = minimum_spanning_tree(knn_sym).tocoo()
    edges = np.column_stack([mst_sparse.row, mst_sparse.col, mst_sparse.data]).astype(float)
    total_weight = float(np.sum(edges[:, 2]))
    return edges, total_weight


@st.cache_data(show_spinner=False)
def verify_mst_against_scipy(D: np.ndarray, custom_weight: float):
    """Cross-checks custom Prim's MST total weight against SciPy's minimum_spanning_tree."""
    mst_scipy = minimum_spanning_tree(D)
    w_scipy = float(mst_scipy.sum())
    diff = abs(custom_weight - w_scipy)
    is_match = bool(np.isclose(custom_weight, w_scipy, atol=1e-6))
    return w_scipy, diff, is_match


def edges_to_labels(n_nodes: int, kept_edges: np.ndarray, min_size_noise: int = 0) -> np.ndarray:
    """
    Converts a set of retained MST edges into cluster labels via Connected Components.
    If min_size_noise > 0, components with fewer than `min_size_noise` nodes are marked as -1 (Noise).
    """
    if len(kept_edges) == 0:
        labels = np.arange(n_nodes)
    else:
        u = kept_edges[:, 0].astype(int)
        v = kept_edges[:, 1].astype(int)
        w = np.ones(len(u), dtype=float)

        adj = csr_matrix(
            (np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(n_nodes, n_nodes),
        )
        _, labels = connected_components(csgraph=adj, directed=False)

    if min_size_noise > 1:
        unique_lbls, counts = np.unique(labels, return_counts=True)
        valid_lbls = unique_lbls[counts >= min_size_noise]
        remap = {old_l: new_l for new_l, old_l in enumerate(valid_lbls)}
        labels = np.array([remap.get(l, -1) for l in labels], dtype=int)

    return labels


@st.cache_data(show_spinner=False)
def mst_cluster_top_k(n_nodes: int, mst_edges: np.ndarray, k: int, mark_noise_below: int = 0):
    """Clusters an MST into k clusters by cutting the k-1 heaviest edges."""
    if k <= 1:
        return np.zeros(n_nodes, dtype=int), mst_edges, np.empty((0, 3))

    sorted_idx = np.argsort(mst_edges[:, 2])
    sorted_edges = mst_edges[sorted_idx]

    kept_edges = sorted_edges[: -(k - 1)]
    cut_edges = sorted_edges[-(k - 1) :]

    labels = edges_to_labels(n_nodes, kept_edges, min_size_noise=mark_noise_below)
    return labels, kept_edges, cut_edges


def get_nearby_edge_weights(adj_list, u: int, v: int, depth: int = 2, exclude_edge_idx: int = -1):
    """
    BFS traversal up to `depth` hops from endpoint nodes `u` and `v` in the MST
    to gather local neighborhood edge weights (excluding the candidate edge itself).
    """
    collected_edge_indices = set()
    for start_node in (u, v):
        visited_nodes = {u, v}
        queue = [(start_node, 0)]
        while queue:
            curr_node, curr_d = queue.pop(0)
            if curr_d >= depth:
                continue
            for neighbor, edge_idx in adj_list[curr_node]:
                if edge_idx != exclude_edge_idx:
                    collected_edge_indices.add(edge_idx)
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, curr_d + 1))
    return list(collected_edge_indices)


@st.cache_data(show_spinner=False)
def mst_cluster_inconsistency(
    n_nodes: int,
    mst_edges: np.ndarray,
    t: float = 2.2,
    depth: int = 3,
    min_cluster_size: int = 5,
    mark_outliers_as_noise: bool = False,
):
    """
    Zahn's Inconsistency-Based MST Clustering (for unknown k).
    Supports two outlier handling modes:
      - mark_outliers_as_noise=False: Greedily rejects cuts that isolate fragments < min_cluster_size.
      - mark_outliers_as_noise=True: Applies all inconsistent cuts and labels fragments < min_cluster_size as Noise (-1).
    """
    num_edges = len(mst_edges)
    adj_list = [[] for _ in range(n_nodes)]
    for idx, (u, v, _) in enumerate(mst_edges):
        adj_list[int(u)].append((int(v), idx))
        adj_list[int(v)].append((int(u), idx))

    cut_mask = np.zeros(num_edges, dtype=bool)
    edge_stats = np.full((num_edges, 3), np.nan, dtype=float)  # [mu, sigma, threshold]

    for idx, (u, v, w) in enumerate(mst_edges):
        neighbor_indices = get_nearby_edge_weights(
            adj_list, int(u), int(v), depth=depth, exclude_edge_idx=idx
        )
        if len(neighbor_indices) < 2:
            continue
        local_weights = mst_edges[neighbor_indices, 2]
        mu = float(np.mean(local_weights))
        sigma = float(np.std(local_weights) + 1e-8)
        threshold = mu + t * sigma
        edge_stats[idx] = [mu, sigma, threshold]

        if w > threshold:
            cut_mask[idx] = True

    if mark_outliers_as_noise:
        final_cut_mask = cut_mask
        kept_edges = mst_edges[~final_cut_mask]
        cut_edges = mst_edges[final_cut_mask]
        labels = edges_to_labels(n_nodes, kept_edges, min_size_noise=min_cluster_size)
    else:
        candidate_cuts = np.where(cut_mask)[0]
        candidate_cuts = candidate_cuts[np.argsort(-mst_edges[candidate_cuts, 2])]

        final_cut_mask = np.zeros(num_edges, dtype=bool)
        for c_idx in candidate_cuts:
            trial_mask = final_cut_mask.copy()
            trial_mask[c_idx] = True
            trial_labels = edges_to_labels(n_nodes, mst_edges[~trial_mask])
            _, counts = np.unique(trial_labels, return_counts=True)
            if np.all(counts >= min_cluster_size):
                final_cut_mask = trial_mask

        kept_edges = mst_edges[~final_cut_mask]
        cut_edges = mst_edges[final_cut_mask]
        labels = edges_to_labels(n_nodes, kept_edges)

    kept_stats = edge_stats[~final_cut_mask]
    cut_stats = edge_stats[final_cut_mask]
    return labels, kept_edges, cut_edges, kept_stats, cut_stats


def k_medoids_pam(D: np.ndarray, k: int, max_iter: int = 100, random_state: int = SEED):
    """Pure NumPy implementation of Partitioning Around Medoids (PAM / k-Medoids)."""
    rng = np.random.RandomState(random_state)
    n = D.shape[0]

    medoids = [rng.randint(0, n)]
    for _ in range(1, k):
        dist_sq = np.min(D[:, medoids], axis=1) ** 2
        total_dist_sq = np.sum(dist_sq)
        if total_dist_sq == 0:
            next_medoid = rng.randint(0, n)
        else:
            probs = dist_sq / total_dist_sq
            next_medoid = rng.choice(n, p=probs)
        medoids.append(int(next_medoid))
    medoids = np.array(medoids, dtype=int)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new_medoids = medoids.copy()

        for cluster_id in range(k):
            member_indices = np.where(labels == cluster_id)[0]
            if len(member_indices) == 0:
                continue
            intra_distances = D[np.ix_(member_indices, member_indices)]
            best_local_idx = np.argmin(np.sum(intra_distances, axis=1))
            new_medoids[cluster_id] = member_indices[best_local_idx]

        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids

    return np.argmin(D[:, medoids], axis=1)


@st.cache_data(show_spinner=False)
def run_all_clustering_methods(
    X: np.ndarray,
    k: int,
    inconsistency_t: float = 2.3,
    depth: int = 3,
    min_cluster_size: int = 6,
    use_mreach: bool = False,
    m_neighbors: int = 5,
    mark_noise: bool = False,
    random_state: int = SEED,
):
    """Executes both MST approaches and 6 baselines (including DBSCAN & Spectral) on dataset X."""
    D_euc = compute_distance_matrix(X)
    D_mst = compute_mutual_reachability_matrix(X, m_neighbors=m_neighbors) if use_mreach else D_euc
    mst_edges, _ = prims_mst_from_scratch(D_mst)

    labels_mst_k, _, _ = mst_cluster_top_k(
        len(X), mst_edges, k=k, mark_noise_below=(min_cluster_size if mark_noise else 0)
    )
    labels_mst_inc, _, _, _, _ = mst_cluster_inconsistency(
        len(X),
        mst_edges,
        t=inconsistency_t,
        depth=depth,
        min_cluster_size=min_cluster_size,
        mark_outliers_as_noise=mark_noise,
    )
    labels_single = AgglomerativeClustering(n_clusters=k, linkage="single").fit_predict(X)

    # Estimate eps for DBSCAN from 5th-nearest-neighbor distance percentile
    nn_dists = np.sort(D_euc, axis=1)[:, min(5, len(X) - 1)]
    auto_eps = float(np.percentile(nn_dists, 72))
    labels_dbscan = DBSCAN(eps=max(auto_eps, 0.12), min_samples=min(5, len(X) - 1)).fit_predict(X)

    labels_spectral = SpectralClustering(
        n_clusters=k, affinity="nearest_neighbors", n_neighbors=min(10, len(X) - 1), random_state=random_state
    ).fit_predict(X)

    labels_kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
    labels_kmedoids = k_medoids_pam(D_euc, k=k, random_state=random_state)
    labels_gmm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=random_state
    ).fit_predict(X)

    return {
        "MST (Top k-1)": labels_mst_k,
        "MST (Inconsistency)": labels_mst_inc,
        "Single-Linkage": labels_single,
        "DBSCAN": labels_dbscan,
        "Spectral Clustering": labels_spectral,
        "k-Means": labels_kmeans,
        "k-Medoids (PAM)": labels_kmedoids,
        "GMM": labels_gmm,
    }


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes cluster purity: sum of majority class counts divided by total samples."""
    c_matrix = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix))


def evaluate_predictions(X: np.ndarray, y_true: np.ndarray | None, y_pred: np.ndarray):
    """Computes ARI, NMI, Purity, and Silhouette Score (handling unsupervised/single-cluster cases)."""
    if y_true is not None:
        ari = float(adjusted_rand_score(y_true, y_pred))
        nmi = float(normalized_mutual_info_score(y_true, y_pred))
        purity = compute_purity(y_true, y_pred)
    else:
        ari, nmi, purity = np.nan, np.nan, np.nan

    # Evaluate Silhouette on non-noise points if noise (-1) is present
    valid_mask = y_pred != -1
    if np.sum(valid_mask) > 2 and 1 < len(np.unique(y_pred[valid_mask])) < np.sum(valid_mask):
        sil = float(silhouette_score(X[valid_mask], y_pred[valid_mask], metric="euclidean"))
    else:
        sil = np.nan
    return {"ARI": ari, "NMI": nmi, "Purity": purity, "Silhouette": sil}


@st.cache_data(show_spinner=False)
def run_noise_sweep(
    dataset_name: str,
    n_samples: int,
    k_target: int,
    t_val: float,
    depth: int,
    min_cluster_size: int,
    m_neighbors: int,
    base_seed: int,
):
    """
    Sweeps Gaussian noise sigma from 0.02 to 0.22 across 5 trials per level,
    explicitly comparing Euclidean MST vs. Mutual Reachability MST alongside baselines.
    """
    noise_levels = np.linspace(0.02, 0.22, 9)
    sweep_ds = dataset_name if dataset_name in ["Two Moons", "Concentric Circles", "Two Spirals", "Anisotropic Blobs"] else "Two Moons"
    records = []
    for nl in noise_levels:
        for trial_seed in range(base_seed, base_seed + 5):
            ds_dict = generate_all_datasets(
                n_samples=n_samples, noise=float(nl), random_state=int(trial_seed)
            )
            X_s, y_s, k_true = ds_dict[sweep_ds]
            k_use = k_true if sweep_ds != "Anisotropic Blobs" else k_target

            # Standard Euclidean methods
            D_euc = compute_distance_matrix(X_s)
            mst_euc, _ = prims_mst_from_scratch(D_euc)
            lbl_euc_k, _, _ = mst_cluster_top_k(len(X_s), mst_euc, k=k_use)
            lbl_euc_inc, _, _, _, _ = mst_cluster_inconsistency(
                len(X_s), mst_euc, t=t_val, depth=depth, min_cluster_size=min_cluster_size
            )

            # Mutual Reachability MST (Chaining-resistant upgrade)
            D_mr = compute_mutual_reachability_matrix(X_s, m_neighbors=m_neighbors)
            mst_mr, _ = prims_mst_from_scratch(D_mr)
            lbl_mr_k, _, _ = mst_cluster_top_k(len(X_s), mst_mr, k=k_use)

            lbl_kmeans = KMeans(n_clusters=k_use, n_init=10, random_state=int(trial_seed)).fit_predict(X_s)
            lbl_gmm = GaussianMixture(n_components=k_use, random_state=int(trial_seed)).fit_predict(X_s)

            for m_name, y_p in [
                ("MST Top k-1 (Euclidean)", lbl_euc_k),
                ("MST Top k-1 (Mutual Reachability)", lbl_mr_k),
                ("MST Inconsistency (Euclidean)", lbl_euc_inc),
                ("k-Means", lbl_kmeans),
                ("GMM", lbl_gmm),
            ]:
                records.append(
                    {
                        "Noise": round(float(nl), 4),
                        "Method": m_name,
                        "ARI": float(adjusted_rand_score(y_s, y_p)),
                    }
                )
    return pd.DataFrame(records), sweep_ds


@st.cache_data(show_spinner=False)
def run_threshold_sweep(
    n_samples: int,
    noise: float,
    random_state: int,
    depth: int,
    min_cluster_size: int,
    use_mreach: bool,
    m_neighbors: int,
):
    """Sweeps Zahn's inconsistency threshold t from 1.3 to 3.3 across all 4 benchmark datasets."""
    t_values = np.linspace(1.3, 3.3, 11)
    datasets = generate_all_datasets(n_samples=n_samples, noise=noise, random_state=random_state)
    records = []
    for ds_name, (X, y_true, _) in datasets.items():
        D = (
            compute_mutual_reachability_matrix(X, m_neighbors=m_neighbors)
            if use_mreach
            else compute_distance_matrix(X)
        )
        mst_edges, _ = prims_mst_from_scratch(D)
        for t_val in t_values:
            y_inc, _, _, _, _ = mst_cluster_inconsistency(
                len(X),
                mst_edges,
                t=float(t_val),
                depth=depth,
                min_cluster_size=min_cluster_size,
            )
            records.append(
                {
                    "Dataset": ds_name,
                    "Threshold_t": round(float(t_val), 2),
                    "ARI": float(adjusted_rand_score(y_true, y_inc)),
                    "Discovered_k": int(len(np.unique(y_inc))),
                }
            )
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def run_scalability_benchmark():
    """Benchmarks Dense Prim's O(N^2) vs. Sparse k-NN MST O(N log N) vs. k-Means across N."""
    sizes = [150, 300, 600, 1000, 1500]
    records = []
    for N in sizes:
        X_s, _ = make_moons(n_samples=N, noise=0.05, random_state=SEED)
        X_s = StandardScaler().fit_transform(X_s)

        t0 = time.perf_counter()
        D_s = compute_distance_matrix(X_s)
        e_dense, _ = prims_mst_from_scratch(D_s)
        mst_cluster_top_k(N, e_dense, k=2)
        records.append({"N": N, "Method": "Dense Prim's MST O(N²)", "Time_ms": (time.perf_counter() - t0) * 1000})

        t0 = time.perf_counter()
        e_sparse, _ = sparse_knn_mst(X_s, n_neighbors=15)
        mst_cluster_top_k(N, e_sparse, k=2)
        records.append({"N": N, "Method": "Sparse k-NN MST O(N log N)", "Time_ms": (time.perf_counter() - t0) * 1000})

        t0 = time.perf_counter()
        KMeans(n_clusters=2, n_init=5, random_state=SEED).fit_predict(X_s)
        records.append({"N": N, "Method": "k-Means", "Time_ms": (time.perf_counter() - t0) * 1000})

    return pd.DataFrame(records)


# ==============================================================================
# 2. Visualization Helpers
# ==============================================================================
def _color_for_label(label: int) -> str:
    if int(label) == -1:
        return NOISE_COLOR
    return CLUSTER_PALETTE[int(label) % len(CLUSTER_PALETTE)]


def build_plotly_mst_explorer(
    X_plot: np.ndarray,
    y_true: np.ndarray | None,
    labels_mst: np.ndarray,
    kept_edges: np.ndarray,
    cut_edges: np.ndarray,
    cut_stats: np.ndarray | None,
    dataset_name: str,
    strategy_label: str,
    show_edges: bool,
    show_midpoints: bool,
    is_inconsistency: bool,
    axis_labels: tuple[str, str] = ("x₁ (Standardized)", "x₂ (Standardized)"),
):
    """Builds the interactive side-by-side Plotly figure: Ground Truth vs. Constructed MST."""
    gt_title = (
        f"{dataset_name} — Ground Truth (k = {len(np.unique(y_true))})"
        if y_true is not None
        else f"{dataset_name} — Input Data (Unlabeled)"
    )
    n_found = len(set(labels_mst) - {-1})
    noise_cnt = int(np.sum(labels_mst == -1))
    mst_title_suffix = f"k = {n_found}" + (f", noise = {noise_cnt}" if noise_cnt > 0 else "")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            gt_title,
            f"Constructed MST — {strategy_label} ({mst_title_suffix})",
        ),
        horizontal_spacing=0.07,
    )

    # Left panel: Ground Truth scatter
    gt_colors = [_color_for_label(lbl) for lbl in y_true] if y_true is not None else ["#1f77b4"] * len(X_plot)
    gt_texts = (
        [f"Node {i}<br>True Cluster: {lbl}" for i, lbl in enumerate(y_true)]
        if y_true is not None
        else [f"Node {i}" for i in range(len(X_plot))]
    )
    fig.add_trace(
        go.Scatter(
            x=X_plot[:, 0],
            y=X_plot[:, 1],
            mode="markers",
            marker=dict(
                color=gt_colors,
                size=7,
                line=dict(width=0.6, color="black"),
            ),
            text=gt_texts,
            hoverinfo="text",
            name="Ground Truth",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Right panel: Retained MST edges
    if show_edges and len(kept_edges) > 0:
        edge_x, edge_y = [], []
        for u, v, _ in kept_edges:
            u, v = int(u), int(v)
            edge_x.extend([X_plot[u, 0], X_plot[v, 0], None])
            edge_y.extend([X_plot[u, 1], X_plot[v, 1], None])

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="slategray", width=1.1),
                opacity=0.55,
                hoverinfo="skip",
                name="Retained MST Edges",
            ),
            row=1,
            col=2,
        )

    # Right panel: Cut edges (bold dashed crimson)
    if show_edges and len(cut_edges) > 0:
        cut_x, cut_y = [], []
        for u, v, _ in cut_edges:
            u, v = int(u), int(v)
            cut_x.extend([X_plot[u, 0], X_plot[v, 0], None])
            cut_y.extend([X_plot[u, 1], X_plot[v, 1], None])

        fig.add_trace(
            go.Scatter(
                x=cut_x,
                y=cut_y,
                mode="lines",
                line=dict(color="crimson", width=3.0, dash="dash"),
                hoverinfo="skip",
                name="Cut Edges",
            ),
            row=1,
            col=2,
        )

    # Cut-edge midpoint markers & rich hover tooltips (weight + mu + t*sigma)
    if len(cut_edges) > 0:
        mid_x, mid_y, hover_texts = [], [], []
        for idx, (u, v, w) in enumerate(cut_edges):
            u, v = int(u), int(v)
            mid_x.append((X_plot[u, 0] + X_plot[v, 0]) / 2.0)
            mid_y.append((X_plot[u, 1] + X_plot[v, 1]) / 2.0)

            if is_inconsistency and cut_stats is not None and not np.isnan(cut_stats[idx, 2]):
                mu, sigma, thresh = cut_stats[idx]
                hover_texts.append(
                    f"<b>Cut Edge ({u}, {v})</b><br>"
                    f"Edge Weight (w): <b>{w:.4f}</b><br>"
                    f"Local Mean (μ): {mu:.4f}<br>"
                    f"Local Std (σ): {sigma:.4f}<br>"
                    f"Threshold (μ + tσ): <b>{thresh:.4f}</b>"
                )
            else:
                hover_texts.append(
                    f"<b>Cut Edge ({u}, {v})</b><br>"
                    f"Edge Weight (w): <b>{w:.4f}</b><br>"
                    f"Criterion: Top (k-1) Heaviest Edge"
                )

        fig.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="markers",
                marker=dict(
                    symbol="x" if show_midpoints else "circle",
                    size=11 if show_midpoints else 8,
                    color="crimson" if show_midpoints else "rgba(220, 20, 60, 0.01)",
                    line=dict(width=1.5 if show_midpoints else 0, color="crimson"),
                ),
                text=hover_texts,
                hoverinfo="text",
                name="Cut Midpoint" if show_midpoints else "Cut Edge Info",
                showlegend=show_midpoints,
            ),
            row=1,
            col=2,
        )

    # Right panel: Nodes colored by discovered cluster
    mst_colors = [_color_for_label(lbl) for lbl in labels_mst]
    mst_symbols = ["diamond-open" if lbl == -1 else "circle" for lbl in labels_mst]
    fig.add_trace(
        go.Scatter(
            x=X_plot[:, 0],
            y=X_plot[:, 1],
            mode="markers",
            marker=dict(
                color=mst_colors,
                symbol=mst_symbols,
                size=7,
                line=dict(width=0.6, color="black"),
            ),
            text=[
                f"Node {i}<br>Discovered: {'Noise (-1)' if lbl == -1 else f'Cluster {lbl}'}"
                + (f"<br>True Cluster: {y_true[i]}" if y_true is not None else "")
                for i, lbl in enumerate(labels_mst)
            ],
            hoverinfo="text",
            name="MST Nodes",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text=axis_labels[0], row=1, col=1)
    fig.update_yaxes(title_text=axis_labels[1], row=1, col=1)
    fig.update_xaxes(title_text=axis_labels[0], row=1, col=2)
    fig.update_yaxes(title_text=axis_labels[1], row=1, col=2)

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
        hovermode="closest",
    )
    return fig


def build_matplotlib_mst_explorer(
    X_plot: np.ndarray,
    y_true: np.ndarray | None,
    labels_mst: np.ndarray,
    kept_edges: np.ndarray,
    cut_edges: np.ndarray,
    dataset_name: str,
    strategy_label: str,
    show_edges: bool,
    show_midpoints: bool,
    axis_labels: tuple[str, str] = ("$x_1$ (Standardized)", "$x_2$ (Standardized)"),
):
    """Builds the crisp Matplotlib side-by-side figure for Ground Truth vs. Constructed MST."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=120)

    c_gt = y_true if y_true is not None else np.zeros(len(X_plot))
    axes[0].scatter(X_plot[:, 0], X_plot[:, 1], c=c_gt, cmap="tab10", s=26, edgecolor="k", linewidth=0.3)
    axes[0].set_title(
        f"{dataset_name} — Ground Truth",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0].set_xlabel(axis_labels[0])
    axes[0].set_ylabel(axis_labels[1])
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    if show_edges and len(kept_edges) > 0:
        for u, v, _ in kept_edges:
            u, v = int(u), int(v)
            ax.plot(
                [X_plot[u, 0], X_plot[v, 0]],
                [X_plot[u, 1], X_plot[v, 1]],
                color="slategray",
                alpha=0.55,
                lw=1.0,
                zorder=1,
            )

    if show_edges and len(cut_edges) > 0:
        for idx_c, (u, v, _) in enumerate(cut_edges):
            u, v = int(u), int(v)
            lbl = "Cut Edge" if idx_c == 0 else None
            ax.plot(
                [X_plot[u, 0], X_plot[v, 0]],
                [X_plot[u, 1], X_plot[v, 1]],
                color="crimson",
                ls="--",
                lw=2.6,
                zorder=3,
                label=lbl,
            )
            if show_midpoints:
                mid_x, mid_y = (X_plot[u, 0] + X_plot[v, 0]) / 2.0, (X_plot[u, 1] + X_plot[v, 1]) / 2.0
                ax.scatter([mid_x], [mid_y], color="crimson", marker="x", s=75, lw=2, zorder=4)

    ax.scatter(
        X_plot[:, 0], X_plot[:, 1], c=labels_mst, cmap="tab10", s=26, edgecolor="k", linewidth=0.3, zorder=2
    )
    ax.set_title(
        f"Constructed MST — {strategy_label} (k = {len(set(labels_mst) - {-1})})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.grid(alpha=0.3)
    if show_edges and len(cut_edges) > 0:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def build_edge_weight_histogram(mst_edges: np.ndarray, cut_edges: np.ndarray):
    """Builds an interactive histogram of MST edge weights highlighting the cut threshold(s)."""
    df_w = pd.DataFrame(
        {
            "Weight": mst_edges[:, 2],
            "Status": [
                "Cut Bridge Edge" if w in set(cut_edges[:, 2]) else "Retained Intra-Cluster Edge"
                for w in mst_edges[:, 2]
            ],
        }
    )
    fig = px.histogram(
        df_w,
        x="Weight",
        color="Status",
        nbins=35,
        color_discrete_map={
            "Retained Intra-Cluster Edge": "slategray",
            "Cut Bridge Edge": "crimson",
        },
        title="MST Edge Weight Distribution (Short Intra-Cluster Skeleton vs. Heavy Inter-Cluster Bridges)",
    )
    if len(cut_edges) > 0:
        min_cut_w = float(np.min(cut_edges[:, 2]))
        fig.add_vline(
            x=min_cut_w,
            line_dash="dash",
            line_color="crimson",
            annotation_text=f"Min Cut Weight = {min_cut_w:.3f}",
        )
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def build_plotly_method_grid(X_plot: np.ndarray, y_true: np.ndarray | None, preds_dict: dict):
    """Builds a 2x4 interactive Plotly grid comparing all 8 clustering methods."""
    titles = []
    for m_name in METHOD_ORDER:
        if y_true is not None:
            ari_val = adjusted_rand_score(y_true, preds_dict[m_name])
            titles.append(f"<b>{m_name}</b><br><span style='font-size:12px'>ARI = {ari_val:.3f}</span>")
        else:
            k_found = len(set(preds_dict[m_name]) - {-1})
            titles.append(f"<b>{m_name}</b><br><span style='font-size:12px'>Found k = {k_found}</span>")

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.16,
    )

    for idx, m_name in enumerate(METHOD_ORDER):
        row = idx // 4 + 1
        col = idx % 4 + 1
        y_pred = preds_dict[m_name]
        colors = [_color_for_label(lbl) for lbl in y_pred]
        fig.add_trace(
            go.Scatter(
                x=X_plot[:, 0],
                y=X_plot[:, 1],
                mode="markers",
                marker=dict(color=colors, size=6, line=dict(width=0.4, color="black")),
                text=[f"{m_name}<br>Cluster: {lbl}" for lbl in y_pred],
                hoverinfo="text",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
    )
    return fig


def build_matplotlib_method_grid(
    X_plot: np.ndarray, y_true: np.ndarray | None, preds_dict: dict, dataset_name: str
):
    """Builds a 2x4 Matplotlib grid comparing all 8 clustering methods."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2), sharex=True, sharey=True, dpi=115)
    for idx, m_name in enumerate(METHOD_ORDER):
        ax = axes[idx // 4, idx % 4]
        y_pred = preds_dict[m_name]
        ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_pred, cmap="tab10", s=20, edgecolor="k", linewidth=0.25)
        ax.set_title(m_name, fontsize=11, fontweight="bold")
        if y_true is not None:
            ari_val = adjusted_rand_score(y_true, y_pred)
            badge = f"ARI = {ari_val:.3f}"
        else:
            badge = f"k = {len(set(y_pred) - {-1})}"
        ax.text(
            0.04,
            0.88,
            badge,
            transform=ax.transAxes,
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
        )
        ax.grid(alpha=0.3)
        if idx // 4 == 1:
            ax.set_xlabel("$x_1$")
        if idx % 4 == 0:
            ax.set_ylabel("$x_2$")

    fig.suptitle(f"Cluster Structure Comparison on {dataset_name} (8 Methods)", fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    return fig


# ==============================================================================
# 3. Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.title("🌳 MST Clustering Lab")
    st.caption("Interactive Academic Viva & Demo Dashboard")

    # Section 1: Dataset Configuration
    st.header("1. Dataset Configuration")
    selected_dataset = st.selectbox(
        "Dataset / Topology",
        options=[
            "Two Moons",
            "Concentric Circles",
            "Two Spirals",
            "Anisotropic Blobs",
            "Iris (4D → PCA 2D)",
            "Wine (13D → PCA 2D)",
            "Breast Cancer (30D → PCA 2D)",
            "Custom CSV Upload",
        ],
        index=0,
    )

    is_synthetic = selected_dataset in [
        "Two Moons",
        "Concentric Circles",
        "Two Spirals",
        "Anisotropic Blobs",
    ]

    n_samples = st.slider(
        "Number of Samples (N)",
        min_value=100,
        max_value=600,
        value=300,
        step=20,
        disabled=not is_synthetic,
    )
    noise_level = st.slider(
        "Gaussian Noise (σ)",
        min_value=0.01,
        max_value=0.25,
        value=0.05,
        step=0.01,
        disabled=not is_synthetic,
    )
    seed_val = st.slider("Random Seed", min_value=0, max_value=200, value=SEED, step=1)

    uploaded_csv = None
    if selected_dataset == "Custom CSV Upload":
        uploaded_csv = st.file_uploader("Upload CSV File", type=["csv"])

    st.divider()

    # Section 2: MST Distance & Cutting Strategy
    st.header("2. MST Graph & Cutting Strategy")
    distance_mode = st.radio(
        "Graph Edge Weight Metric",
        options=[
            "Euclidean Distance (Standard)",
            "Mutual Reachability (HDBSCAN Anti-Chaining)",
        ],
        index=0,
        help="Mutual Reachability pushes sparse noise points away from dense manifolds before building the MST.",
    )
    use_mreach = distance_mode.startswith("Mutual")
    m_neighbors = st.slider(
        "Core Distance k-NN (m)",
        min_value=2,
        max_value=15,
        value=5,
        step=1,
        disabled=not use_mreach,
    )

    graph_construction_mode = st.selectbox(
        "MST Graph Construction",
        options=[
            "Dense Prim's Algorithm O(N²)",
            "Sparse k-NN Graph MST O(N log N)",
        ],
        index=0,
    )
    use_sparse_knn = graph_construction_mode.startswith("Sparse")
    knn_k = st.slider(
        "k-NN Graph Neighbors",
        min_value=5,
        max_value=40,
        value=15,
        step=5,
        disabled=not use_sparse_knn,
    )

    cutting_strategy = st.radio(
        "Cutting Criterion",
        options=[
            "Known k: Top (k - 1) Heaviest Edges",
            "Unknown k: Zahn's Local Inconsistency Cut",
        ],
        index=0,
    )
    is_inconsistency = cutting_strategy.startswith("Unknown k")

    default_k = DATASET_DEFAULT_K.get(selected_dataset, 2)
    target_k = st.slider(
        "Target Clusters (k)",
        min_value=2,
        max_value=5,
        value=default_k,
        step=1,
        help="Used by MST Top-(k-1) Cut and conventional baselines in Tab 2.",
    )
    t_threshold = st.slider(
        "Inconsistency Multiplier (t)",
        min_value=1.2,
        max_value=3.5,
        value=2.3,
        step=0.1,
        help="Cuts edge e if w(e) > μ + t·σ within BFS neighborhood.",
    )
    bfs_depth = st.slider(
        "BFS Neighborhood Depth (d)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )
    min_cluster_size = st.slider(
        "Min Cluster Size (Pruning / Noise)",
        min_value=2,
        max_value=15,
        value=6,
        step=1,
    )
    mark_noise = st.checkbox(
        "Label Fragments < Min Size as Noise (-1)",
        value=False,
        help="Instead of merging tiny outlier fragments into adjacent clusters, mark them as Noise (-1).",
    )

    st.divider()

    # Section 3: Display Options
    st.header("3. Display Options")
    show_mst_edges = st.checkbox("Show MST Edges", value=True)
    show_cut_midpoints = st.checkbox("Show Cut-Edge Midpoint Markers", value=True)
    use_plotly = st.checkbox("Interactive Plotly Rendering (Uncheck for Matplotlib)", value=True)


# ==============================================================================
# 4. Core Computation for Current State
# ==============================================================================
axis_labels = ("x₁ (Standardized)", "x₂ (Standardized)")

if is_synthetic:
    all_datasets = generate_all_datasets(
        n_samples=n_samples, noise=noise_level, random_state=seed_val
    )
    X_features, y_true, k_true = all_datasets[selected_dataset]
    X_plot = X_features
elif selected_dataset == "Custom CSV Upload":
    if uploaded_csv is not None:
        df_csv = pd.read_csv(uploaded_csv)
        num_cols = df_csv.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            st.error("Uploaded CSV must contain at least 2 numeric columns.")
            st.stop()
        with st.sidebar:
            feat_cols = st.multiselect("Feature Columns", options=num_cols, default=num_cols[: min(6, len(num_cols))])
            lbl_col = st.selectbox("Ground Truth Column (Optional)", options=["<None>"] + df_csv.columns.tolist())
        if len(feat_cols) < 2:
            st.warning("Please select at least 2 feature columns in the sidebar.")
            st.stop()
        X_raw = df_csv[feat_cols].dropna().values
        X_features = StandardScaler().fit_transform(X_raw)
        if lbl_col != "<None>":
            y_true = pd.factorize(df_csv.loc[df_csv[feat_cols].dropna().index, lbl_col])[0]
            k_true = int(len(np.unique(y_true)))
        else:
            y_true = None
            k_true = target_k
        if X_features.shape[1] > 2:
            pca = PCA(n_components=2, random_state=seed_val)
            X_plot = pca.fit_transform(X_features)
            vr = pca.explained_variance_ratio_
            axis_labels = (f"PC1 ({vr[0]*100:.1f}% var)", f"PC2 ({vr[1]*100:.1f}% var)")
        else:
            X_plot = X_features
    else:
        st.info("Upload a CSV file in the sidebar (or switch back to a built-in dataset) to begin.")
        st.stop()
else:
    X_features, X_plot, y_true, k_true, axis_labels = load_real_world_dataset(selected_dataset)

# Build distance matrix & MST
D = (
    compute_mutual_reachability_matrix(X_features, m_neighbors=m_neighbors)
    if use_mreach
    else compute_distance_matrix(X_features)
)

if use_sparse_knn and not use_mreach:
    mst_edges, w_custom = sparse_knn_mst(X_features, n_neighbors=knn_k)
else:
    mst_edges, w_custom = prims_mst_from_scratch(D)

w_scipy, w_diff, scipy_match = verify_mst_against_scipy(D, w_custom)

if is_inconsistency:
    labels_active, kept_active, cut_active, _, cut_stats_active = mst_cluster_inconsistency(
        len(X_features),
        mst_edges,
        t=t_threshold,
        depth=bfs_depth,
        min_cluster_size=min_cluster_size,
        mark_outliers_as_noise=mark_noise,
    )
    strategy_short = f"Inconsistency (t={t_threshold:.1f}, d={bfs_depth})"
else:
    labels_active, kept_active, cut_active = mst_cluster_top_k(
        len(X_features),
        mst_edges,
        k=target_k,
        mark_noise_below=(min_cluster_size if mark_noise else 0),
    )
    cut_stats_active = None
    strategy_short = f"Top (k-1) Cut (k={target_k})"

active_metrics = evaluate_predictions(X_features, y_true, labels_active)
discovered_k = int(len(set(labels_active) - {-1}))


# ==============================================================================
# 5. Main Dashboard Header & Tabs
# ==============================================================================
st.title("MST-Based Clustering for Non-Linearly Separable Patterns")
st.caption(
    f"Active Dataset: **{selected_dataset}** (N = {len(X_features)}, Dim = {X_features.shape[1]}D)  |  "
    f"Metric: **{'Mutual Reachability' if use_mreach else 'Euclidean'}**  |  "
    f"Active Strategy: **{strategy_short}**"
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔍 Live MST Graph & Cut Explorer",
        "📊 Method Comparison Grid & Metrics",
        "🧪 Chaining Effect & Sensitivity Lab",
        "🎓 Viva Notes & Complexity Reference",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1 — Live MST Graph & Cut Explorer
# ------------------------------------------------------------------------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        badge_text = f"SciPy Match Δ = {w_diff:.2f}" if scipy_match else f"Sparse k-NN Δ = {w_diff:.2f}"
        st.metric(
            label="Total MST Weight",
            value=f"{w_custom:.2f}",
            delta=badge_text,
            delta_color="normal" if scipy_match else "off",
        )
    with col2:
        st.metric(
            label="Discovered Clusters (k)",
            value=f"{discovered_k}",
            delta=f"Ground Truth k = {k_true}",
            delta_color="off",
        )
    with col3:
        st.metric(
            label="MST ARI",
            value=f"{active_metrics['ARI']:.3f}" if not np.isnan(active_metrics["ARI"]) else "N/A",
        )
    with col4:
        st.metric(
            label="MST NMI",
            value=f"{active_metrics['NMI']:.3f}" if not np.isnan(active_metrics["NMI"]) else "N/A",
        )

    if scipy_match:
        st.success(
            f"**Exact Prim's ↔ SciPy Verification Passed:** Custom $O(N^2)$ Prim's weight "
            f"(`{w_custom:.6f}`) matches `scipy.sparse.csgraph.minimum_spanning_tree` "
            f"(`{w_scipy:.6f}`) with difference `{w_diff:.2e}`."
        )

    if use_plotly:
        fig_mst = build_plotly_mst_explorer(
            X_plot=X_plot,
            y_true=y_true,
            labels_mst=labels_active,
            kept_edges=kept_active,
            cut_edges=cut_active,
            cut_stats=cut_stats_active,
            dataset_name=selected_dataset,
            strategy_label=strategy_short,
            show_edges=show_mst_edges,
            show_midpoints=show_cut_midpoints,
            is_inconsistency=is_inconsistency,
            axis_labels=axis_labels,
        )
        st.plotly_chart(fig_mst, width="stretch")
    else:
        fig_mpl = build_matplotlib_mst_explorer(
            X_plot=X_plot,
            y_true=y_true,
            labels_mst=labels_active,
            kept_edges=kept_active,
            cut_edges=cut_active,
            dataset_name=selected_dataset,
            strategy_label=strategy_short,
            show_edges=show_mst_edges,
            show_midpoints=show_cut_midpoints,
            axis_labels=axis_labels,
        )
        st.pyplot(fig_mpl, width="stretch")
        plt.close(fig_mpl)

    # Edge Weight Distribution Histogram
    st.plotly_chart(build_edge_weight_histogram(mst_edges, cut_active), width="stretch")

    # Interactive Step-by-Step Prim's Growth Animator
    with st.expander("🎬 Step-by-Step Prim's MST Growth Animator (Viva Walkthrough)", expanded=False):
        step_edges = st.slider(
            "Prim's Greedy Edge Addition Step (1 → N-1)",
            min_value=1,
            max_value=len(mst_edges),
            value=len(mst_edges),
            step=1,
        )
        partial_edges = mst_edges[:step_edges]
        gx, gy = [], []
        for u, v, _ in partial_edges:
            u, v = int(u), int(v)
            gx.extend([X_plot[u, 0], X_plot[v, 0], None])
            gy.extend([X_plot[u, 1], X_plot[v, 1], None])

        fig_step = go.Figure()
        fig_step.add_trace(
            go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="#1f77b4", width=1.8), name="Growing Tree")
        )
        fig_step.add_trace(
            go.Scatter(
                x=X_plot[:, 0],
                y=X_plot[:, 1],
                mode="markers",
                marker=dict(
                    color=["crimson" if i == 0 else "slategray" for i in range(len(X_plot))],
                    size=[11 if i == 0 else 6 for i in range(len(X_plot))],
                ),
                name="Nodes (Root = Node 0 in Crimson)",
            )
        )
        fig_step.update_layout(
            title=f"Prim's Greedy Tree after {step_edges} / {len(mst_edges)} Edges (Cumulative Weight = {np.sum(partial_edges[:, 2]):.2f})",
            height=400,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_step, width="stretch")

    if len(cut_active) > 0:
        with st.expander("Inspect Cut Edges Table", expanded=False):
            cut_rows = []
            for idx, (u, v, w) in enumerate(cut_active):
                row_data = {"Edge": f"({int(u)}, {int(v)})", "Weight (w)": round(float(w), 5)}
                if is_inconsistency and cut_stats_active is not None and not np.isnan(cut_stats_active[idx, 2]):
                    row_data["Local Mean (μ)"] = round(float(cut_stats_active[idx, 0]), 5)
                    row_data["Local Std (σ)"] = round(float(cut_stats_active[idx, 1]), 5)
                    row_data["Threshold (μ + tσ)"] = round(float(cut_stats_active[idx, 2]), 5)
                cut_rows.append(row_data)
            st.dataframe(pd.DataFrame(cut_rows), width="stretch", hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2 — Method Comparison Grid & Metrics
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"8-Method Cluster Structure Comparison — {selected_dataset}")
    preds_dict = run_all_clustering_methods(
        X_features,
        k=target_k,
        inconsistency_t=t_threshold,
        depth=bfs_depth,
        min_cluster_size=min_cluster_size,
        use_mreach=use_mreach,
        m_neighbors=m_neighbors,
        mark_noise=mark_noise,
        random_state=seed_val,
    )

    if use_plotly:
        fig_grid = build_plotly_method_grid(X_plot, y_true, preds_dict)
        st.plotly_chart(fig_grid, width="stretch")
    else:
        fig_grid_mpl = build_matplotlib_method_grid(X_plot, y_true, preds_dict, selected_dataset)
        st.pyplot(fig_grid_mpl, width="stretch")
        plt.close(fig_grid_mpl)

    st.divider()
    col_tbl, col_heat = st.columns([1.05, 1.15])

    metrics_rows = []
    for m_name in METHOD_ORDER:
        m_scores = evaluate_predictions(X_features, y_true, preds_dict[m_name])
        metrics_rows.append({"Method": m_name, **m_scores})
    metrics_df = pd.DataFrame(metrics_rows).set_index("Method")

    with col_tbl:
        st.subheader("Quantitative Evaluation Table")
        styled_df = metrics_df.style.format("{:.3f}", na_rep="N/A").background_gradient(
            cmap="YlGnBu", axis=0, vmin=0.0, vmax=1.0
        )
        st.dataframe(styled_df, width="stretch")
        st.caption(
            "Note: Euclidean Silhouette favors convex Voronoi cells (k-Means / k-Medoids) "
            "even when ARI/NMI on non-linear manifolds is near zero."
        )

        # Exportable Viva Pack
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            export_df = pd.DataFrame(X_plot, columns=["Dim1", "Dim2"])
            if y_true is not None:
                export_df["Ground_Truth"] = y_true
            for m_name, y_p in preds_dict.items():
                export_df[m_name] = y_p
            st.download_button(
                "📥 Download Clustered CSV",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name=f"mst_clusters_{selected_dataset.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
        with dl_col2:
            st.download_button(
                "📥 Download Metrics CSV",
                data=metrics_df.reset_index().to_csv(index=False).encode("utf-8"),
                file_name=f"mst_metrics_{selected_dataset.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )

    with col_heat:
        st.subheader("Interactive Metrics Heatmap")
        heat_fig = px.imshow(
            metrics_df.round(3),
            text_auto=".3f",
            color_continuous_scale="YlGnBu",
            zmin=0.0,
            zmax=1.0,
            aspect="auto",
            labels=dict(x="Evaluation Metric", y="Clustering Method", color="Score"),
        )
        heat_fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(heat_fig, width="stretch")

# ------------------------------------------------------------------------------
# TAB 3 — Chaining Effect & Sensitivity Lab
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Live Sensitivity, Anti-Chaining & Scalability Lab")
    col_n, col_t = st.columns(2)

    with col_n:
        st.markdown("#### 1. Noise Sweep: Euclidean MST vs. Mutual Reachability MST")
        with st.spinner("Running noise sweep (σ = 0.02 → 0.22)..."):
            noise_df, sweep_ds_name = run_noise_sweep(
                dataset_name=selected_dataset,
                n_samples=n_samples,
                k_target=target_k,
                t_val=t_threshold,
                depth=bfs_depth,
                min_cluster_size=min_cluster_size,
                m_neighbors=m_neighbors,
                base_seed=seed_val,
            )
        noise_agg = noise_df.groupby(["Noise", "Method"], as_index=False)["ARI"].mean()

        fig_noise = px.line(
            noise_agg,
            x="Noise",
            y="ARI",
            color="Method",
            markers=True,
            title=f"ARI Degradation vs. Noise σ ({sweep_ds_name}) — Mutual Reachability Mitigation",
        )
        fig_noise.add_vline(
            x=noise_level,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Current σ={noise_level:.2f}",
        )
        if sweep_ds_name == "Two Moons":
            fig_noise.add_vline(
                x=0.12,
                line_dash="dash",
                line_color="crimson",
                annotation_text="Chaining Breakdown (σ≈0.12)",
                annotation_position="bottom left",
            )
        fig_noise.update_yaxes(range=[-0.05, 1.05], title="Adjusted Rand Index (ARI)")
        fig_noise.update_xaxes(title="Gaussian Noise Level (σ)")
        fig_noise.update_layout(height=440, margin=dict(l=15, r=15, t=45, b=20))
        st.plotly_chart(fig_noise, width="stretch")

    with col_t:
        st.markdown("#### 2. Zahn's Threshold (t) Sensitivity Across Datasets")
        with st.spinner("Running threshold sweep (t = 1.3 → 3.3)..."):
            t_df = run_threshold_sweep(
                n_samples=n_samples,
                noise=noise_level,
                random_state=seed_val,
                depth=bfs_depth,
                min_cluster_size=min_cluster_size,
                use_mreach=use_mreach,
                m_neighbors=m_neighbors,
            )

        fig_t = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("ARI vs. Inconsistency Threshold (t)", "Discovered Clusters (k) vs. Threshold (t)"),
        )
        ds_colors = {
            "Two Moons": "#1f77b4",
            "Concentric Circles": "#ff7f0e",
            "Two Spirals": "#2ca02c",
            "Anisotropic Blobs": "#d62728",
        }
        for ds_name, color in ds_colors.items():
            sub = t_df[t_df["Dataset"] == ds_name]
            fig_t.add_trace(
                go.Scatter(
                    x=sub["Threshold_t"],
                    y=sub["ARI"],
                    mode="lines+markers",
                    name=ds_name,
                    line=dict(color=color),
                    legendgroup=ds_name,
                ),
                row=1,
                col=1,
            )
            fig_t.add_trace(
                go.Scatter(
                    x=sub["Threshold_t"],
                    y=sub["Discovered_k"],
                    mode="lines+markers",
                    name=ds_name,
                    line=dict(color=color, dash="dot"),
                    legendgroup=ds_name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig_t.add_vline(x=t_threshold, line_dash="dash", line_color="crimson")
        fig_t.update_yaxes(range=[-0.05, 1.05], title_text="ARI", row=1, col=1)
        fig_t.update_yaxes(title_text="Discovered k", row=2, col=1)
        fig_t.update_xaxes(title_text="Inconsistency Multiplier (t)", row=2, col=1)
        fig_t.update_layout(height=440, margin=dict(l=15, r=15, t=45, b=20))
        st.plotly_chart(fig_t, width="stretch")

    st.divider()
    st.markdown("#### 3. Empirical Runtime Scaling: Dense Prim's $O(N^2)$ vs. Sparse $k$-NN MST $O(N \\log N)$")
    scale_df = run_scalability_benchmark()
    fig_scale = px.line(
        scale_df,
        x="N",
        y="Time_ms",
        color="Method",
        markers=True,
        title="Execution Time (ms) vs. Dataset Size (N)",
    )
    fig_scale.update_xaxes(title="Number of Samples (N)")
    fig_scale.update_yaxes(title="Execution Time (ms)")
    fig_scale.update_layout(height=360, margin=dict(l=15, r=15, t=45, b=20))
    st.plotly_chart(fig_scale, width="stretch")

# ------------------------------------------------------------------------------
# TAB 4 — Viva Notes & Complexity Reference
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Academic Viva Quick-Reference & Theoretical Foundations")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="viva-card">
            <h4>1. Why MST Succeeds on Non-Convex Manifolds While k-Means Fails</h4>
            <ul>
              <li><b>k-Means &amp; k-Medoids:</b> Minimize Euclidean distance to <i>k</i> point prototypes, inducing <b>convex Voronoi cells</b> separated by linear hyperplanes. On Concentric Circles and Two Spirals, any hyperplane slices straight across the manifolds (ARI ≈ 0.00).</li>
              <li><b>MST Clustering:</b> Replaces global distance-to-center with <b>geodesic connectivity along a sparse acyclic skeleton</b>. By retaining only the shortest local edges, the MST traces 1D manifolds step-by-step and isolates inter-cluster bridges.</li>
              <li><b>Exact Single-Linkage Duality:</b> By Kruskal's theorem, cutting the <i>k - 1</i> heaviest MST edges produces the exact same partition as Single-Linkage Agglomerative Clustering (ARI = 1.000 between them).</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="viva-card-success">
            <h4>2. Why Silhouette Score Misleads on Non-Linear Shapes</h4>
            <ul>
              <li>Euclidean Silhouette measures <code>(b - a) / max(a, b)</code> using straight-line distances to cluster centers/members.</li>
              <li>Because a bisected circle or spiral forms two compact Euclidean halves, <b>k-Means can score Silhouette ≈ 0.35–0.40 even when ARI ≈ 0.00</b>.</li>
              <li>Conversely, a true spiral arm wraps around the other arm, resulting in small inter-cluster Euclidean distances and a lower Silhouette score despite 100% topological recovery.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="viva-card-warn">
            <h4>3. The Chaining Failure Mode &amp; HDBSCAN's Mutual Reachability Fix</h4>
            <ul>
              <li><b>Leaf Outlier Decapitation:</b> Isolated peripheral outliers attach via long terminal edges. Cutting the <i>k - 1</i> heaviest edges can sever singletons instead of true cluster bridges.</li>
              <li><b>The Chaining Effect (σ &gt; 0.12):</b> A single noisy point falling in the gap between two moons forms a two-hop short bridge. The MST routes through this bridge, causing ARI to collapse toward 0.</li>
              <li><b>Fix 1 — Min-Cluster-Size Pruning / Noise (-1) Labeling:</b> Reject candidate cuts that create fragments smaller than <code>min_cluster_size</code>, or mark isolated fragments as Noise (-1).</li>
              <li><b>Fix 2 — Mutual Reachability Distance (Live in Sidebar):</b> Replace Euclidean distance with <code>max{core_m(x_i), core_m(x_j), d(x_i, x_j)}</code>, pushing sparse bridge outliers away from dense manifolds before MST construction.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("4. Time & Space Complexity Reference")
    complexity_df = pd.DataFrame(
        [
            {
                "Algorithm": "Custom Dense Prim's MST + Cut",
                "Time Complexity": "O(N²d + N²)",
                "Space Complexity": "O(N²) (Distance Matrix)",
                "Geometric Assumption": "Arbitrary manifold (density-connected)",
            },
            {
                "Algorithm": "Sparse k-NN Graph MST + Cut",
                "Time Complexity": "O(N log N · d)",
                "Space Complexity": "O(N · k_nn)",
                "Geometric Assumption": "Arbitrary manifold (scalable to large N)",
            },
            {
                "Algorithm": "Mutual Reachability MST (HDBSCAN-style)",
                "Time Complexity": "O(N²d)",
                "Space Complexity": "O(N²)",
                "Geometric Assumption": "Arbitrary manifold + robust to chaining/noise",
            },
            {
                "Algorithm": "Single-Linkage (SLINK / Kd-tree)",
                "Time Complexity": "O(N²) worst-case, O(N log N) low-d",
                "Space Complexity": "O(N) to O(N²)",
                "Geometric Assumption": "Arbitrary manifold (dual of MST Top-k)",
            },
            {
                "Algorithm": "DBSCAN",
                "Time Complexity": "O(N log N) (spatial index) / O(N²)",
                "Space Complexity": "O(N)",
                "Geometric Assumption": "Arbitrary density-connected regions + noise",
            },
            {
                "Algorithm": "Spectral Clustering",
                "Time Complexity": "O(N³) (eigendecomposition)",
                "Space Complexity": "O(N²)",
                "Geometric Assumption": "Non-convex graph Laplacian manifolds",
            },
            {
                "Algorithm": "k-Means (Lloyd's)",
                "Time Complexity": "O(N · k · d · I)",
                "Space Complexity": "O(Nd + kd)",
                "Geometric Assumption": "Convex, isotropic (spherical) Voronoi cells",
            },
            {
                "Algorithm": "k-Medoids (PAM)",
                "Time Complexity": "O(N²d + k(N - k)²I)",
                "Space Complexity": "O(N²)",
                "Geometric Assumption": "Convex Voronoi cells (robust to outliers)",
            },
            {
                "Algorithm": "Gaussian Mixture (GMM)",
                "Time Complexity": "O(N · k · d³ · I)",
                "Space Complexity": "O(Nk + kd²)",
                "Geometric Assumption": "Elliptical Gaussian components",
            },
        ]
    )
    st.dataframe(complexity_df, width="stretch", hide_index=True)
