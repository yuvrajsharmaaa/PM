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

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
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

METHOD_ORDER = [
    "MST (Top k-1)",
    "MST (Inconsistency)",
    "Single-Linkage",
    "k-Means",
    "k-Medoids (PAM)",
    "GMM",
]

DATASET_DEFAULT_K = {
    "Two Moons": 2,
    "Concentric Circles": 2,
    "Two Spirals": 2,
    "Anisotropic Blobs": 3,
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
# 1. Core Algorithms (Ported from MST_Clustering_Project.ipynb)
# ==============================================================================
def make_spirals(n_samples: int = 300, noise: float = 0.04, random_state: int = SEED):
    """
    Generates two intertwined Archimedean spirals in 2D space.
    Points are sampled along angle theta so the inner tips remain separated
    while the arms wrap around each other.
    """
    rng = np.random.RandomState(random_state)
    n_per_arm = n_samples // 2

    # Square root sampling ensures roughly uniform arc-length spacing along the spiral
    theta = np.sqrt(rng.uniform(0.15, 1.0, n_per_arm)) * 2.5 * np.pi
    r = theta

    # Arm 1: (r*cos(theta), r*sin(theta))
    x1 = r * np.cos(theta) + rng.normal(0, noise * 8.0, n_per_arm)
    y1 = r * np.sin(theta) + rng.normal(0, noise * 8.0, n_per_arm)

    # Arm 2: rotated by pi radians (-r*cos(theta), -r*sin(theta))
    x2 = -r * np.cos(theta) + rng.normal(0, noise * 8.0, n_per_arm)
    y2 = -r * np.sin(theta) + rng.normal(0, noise * 8.0, n_per_arm)

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_per_arm, dtype=int), np.ones(n_per_arm, dtype=int)])
    return X, y


@st.cache_data(show_spinner=False)
def generate_all_datasets(n_samples: int = 300, noise: float = 0.05, random_state: int = SEED):
    """
    Generates and standardizes the 4 benchmark datasets along with ground-truth k.
    """
    # 1. Two Moons (k = 2)
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    # 2. Concentric Circles (k = 2, factor=0.45 keeps clear radial gap)
    X_circles, y_circles = make_circles(
        n_samples=n_samples, factor=0.45, noise=noise, random_state=random_state
    )

    # 3. Intertwined Spirals (k = 2)
    X_spirals, y_spirals = make_spirals(n_samples=n_samples, noise=noise, random_state=random_state)

    # 4. Anisotropic Blobs (Control dataset, k = 3)
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
def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Computes the pairwise Euclidean distance matrix D where D[i, j] = ||x_i - x_j||_2."""
    return cdist(X, X, metric="euclidean")


@st.cache_data(show_spinner=False)
def prims_mst_from_scratch(D: np.ndarray):
    """
    Dense O(n^2) implementation of Prim's Algorithm for complete graphs.
    Returns:
      edges: numpy array of shape (n-1, 3) where each row is [u, v, weight]
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
def verify_mst_against_scipy(D: np.ndarray, custom_weight: float):
    """Cross-checks custom Prim's MST total weight against SciPy's minimum_spanning_tree."""
    mst_scipy = minimum_spanning_tree(D)
    w_scipy = float(mst_scipy.sum())
    diff = abs(custom_weight - w_scipy)
    is_match = bool(np.isclose(custom_weight, w_scipy, atol=1e-6))
    return w_scipy, diff, is_match


def edges_to_labels(n_nodes: int, kept_edges: np.ndarray) -> np.ndarray:
    """Converts a set of retained MST edges into cluster labels via Connected Components."""
    if len(kept_edges) == 0:
        return np.arange(n_nodes)
    u = kept_edges[:, 0].astype(int)
    v = kept_edges[:, 1].astype(int)
    w = np.ones(len(u), dtype=float)

    adj = csr_matrix(
        (np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))),
        shape=(n_nodes, n_nodes),
    )
    _, labels = connected_components(csgraph=adj, directed=False)
    return labels


@st.cache_data(show_spinner=False)
def mst_cluster_top_k(n_nodes: int, mst_edges: np.ndarray, k: int):
    """Clusters an MST into exactly k clusters by cutting the k-1 heaviest edges."""
    if k <= 1:
        return np.zeros(n_nodes, dtype=int), mst_edges, np.empty((0, 3))

    sorted_idx = np.argsort(mst_edges[:, 2])
    sorted_edges = mst_edges[sorted_idx]

    kept_edges = sorted_edges[: -(k - 1)]
    cut_edges = sorted_edges[-(k - 1) :]

    labels = edges_to_labels(n_nodes, kept_edges)
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
):
    """
    Zahn's Inconsistency-Based MST Clustering (for unknown k).
    Also returns local neighborhood statistics (mu, sigma, mu + t*sigma) for every edge
    for interactive tooltip inspection.
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
    kept_stats = edge_stats[~final_cut_mask]
    cut_stats = edge_stats[final_cut_mask]
    labels = edges_to_labels(n_nodes, kept_edges)
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
    random_state: int = SEED,
):
    """Executes both MST approaches and the 4 conventional baselines on dataset X."""
    D = compute_distance_matrix(X)
    mst_edges, _ = prims_mst_from_scratch(D)

    labels_mst_k, _, _ = mst_cluster_top_k(len(X), mst_edges, k=k)
    labels_mst_inc, _, _, _, _ = mst_cluster_inconsistency(
        len(X), mst_edges, t=inconsistency_t, depth=depth, min_cluster_size=min_cluster_size
    )
    labels_single = AgglomerativeClustering(n_clusters=k, linkage="single").fit_predict(X)
    labels_kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
    labels_kmedoids = k_medoids_pam(D, k=k, random_state=random_state)
    labels_gmm = GaussianMixture(
        n_components=k, covariance_type="full", random_state=random_state
    ).fit_predict(X)

    return {
        "MST (Top k-1)": labels_mst_k,
        "MST (Inconsistency)": labels_mst_inc,
        "Single-Linkage": labels_single,
        "k-Means": labels_kmeans,
        "k-Medoids (PAM)": labels_kmedoids,
        "GMM": labels_gmm,
    }


def compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes cluster purity: sum of majority class counts divided by total samples."""
    c_matrix = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix))


def evaluate_predictions(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    """Computes ARI, NMI, Purity, and Silhouette Score (safely handling single-cluster edge cases)."""
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    purity = compute_purity(y_true, y_pred)

    n_clusters = len(np.unique(y_pred))
    if 1 < n_clusters < len(X):
        sil = float(silhouette_score(X, y_pred, metric="euclidean"))
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
    base_seed: int,
):
    """Sweeps Gaussian noise sigma from 0.02 to 0.22 across 5 trials per level."""
    noise_levels = np.linspace(0.02, 0.22, 9)
    records = []
    for nl in noise_levels:
        for trial_seed in range(base_seed, base_seed + 5):
            ds_dict = generate_all_datasets(
                n_samples=n_samples, noise=float(nl), random_state=int(trial_seed)
            )
            X_s, y_s, k_true = ds_dict[dataset_name]
            k_use = k_true if dataset_name != "Anisotropic Blobs" else k_target
            preds = run_all_clustering_methods(
                X_s,
                k=k_use,
                inconsistency_t=t_val,
                depth=depth,
                min_cluster_size=min_cluster_size,
                random_state=int(trial_seed),
            )
            for m_name, y_p in preds.items():
                records.append(
                    {
                        "Noise": round(float(nl), 4),
                        "Method": m_name,
                        "ARI": float(adjusted_rand_score(y_s, y_p)),
                    }
                )
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def run_threshold_sweep(
    n_samples: int,
    noise: float,
    random_state: int,
    depth: int,
    min_cluster_size: int,
):
    """Sweeps Zahn's inconsistency threshold t from 1.3 to 3.3 across all 4 datasets."""
    t_values = np.linspace(1.3, 3.3, 11)
    datasets = generate_all_datasets(n_samples=n_samples, noise=noise, random_state=random_state)
    records = []
    for ds_name, (X, y_true, _) in datasets.items():
        D = compute_distance_matrix(X)
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


# ==============================================================================
# 2. Visualization Helpers
# ==============================================================================
def _color_for_label(label: int) -> str:
    return CLUSTER_PALETTE[int(label) % len(CLUSTER_PALETTE)]


def build_plotly_mst_explorer(
    X: np.ndarray,
    y_true: np.ndarray,
    labels_mst: np.ndarray,
    kept_edges: np.ndarray,
    cut_edges: np.ndarray,
    cut_stats: np.ndarray,
    dataset_name: str,
    strategy_label: str,
    show_edges: bool,
    show_midpoints: bool,
    is_inconsistency: bool,
):
    """Builds the interactive side-by-side Plotly figure: Ground Truth vs. Constructed MST."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{dataset_name} — Ground Truth (k = {len(np.unique(y_true))})",
            f"Constructed MST — {strategy_label} (k = {len(np.unique(labels_mst))})",
        ),
        horizontal_spacing=0.07,
    )

    # Left panel: Ground Truth scatter
    gt_colors = [_color_for_label(lbl) for lbl in y_true]
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                color=gt_colors,
                size=7,
                line=dict(width=0.6, color="black"),
            ),
            text=[f"Node {i}<br>True Cluster: {lbl}" for i, lbl in enumerate(y_true)],
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
            edge_x.extend([X[u, 0], X[v, 0], None])
            edge_y.extend([X[u, 1], X[v, 1], None])

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
            cut_x.extend([X[u, 0], X[v, 0], None])
            cut_y.extend([X[u, 1], X[v, 1], None])

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
            mid_x.append((X[u, 0] + X[v, 0]) / 2.0)
            mid_y.append((X[u, 1] + X[v, 1]) / 2.0)

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
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                color=mst_colors,
                size=7,
                line=dict(width=0.6, color="black"),
            ),
            text=[
                f"Node {i}<br>Discovered Cluster: {lbl}<br>True Cluster: {y_true[i]}"
                for i, lbl in enumerate(labels_mst)
            ],
            hoverinfo="text",
            name="MST Nodes",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="x₁ (Standardized)", row=1, col=1)
    fig.update_yaxes(title_text="x₂ (Standardized)", row=1, col=1)
    fig.update_xaxes(title_text="x₁ (Standardized)", row=1, col=2)
    fig.update_yaxes(title_text="x₂ (Standardized)", row=1, col=2)

    fig.update_layout(
        height=540,
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
        hovermode="closest",
    )
    return fig


def build_matplotlib_mst_explorer(
    X: np.ndarray,
    y_true: np.ndarray,
    labels_mst: np.ndarray,
    kept_edges: np.ndarray,
    cut_edges: np.ndarray,
    dataset_name: str,
    strategy_label: str,
    show_edges: bool,
    show_midpoints: bool,
):
    """Builds the crisp Matplotlib side-by-side figure for Ground Truth vs. Constructed MST."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=120)

    # Left: Ground Truth
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=26, edgecolor="k", linewidth=0.3)
    axes[0].set_title(
        f"{dataset_name} — Ground Truth (k = {len(np.unique(y_true))})",
        fontsize=11.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("$x_1$ (Standardized)")
    axes[0].set_ylabel("$x_2$ (Standardized)")
    axes[0].grid(alpha=0.3)

    # Right: Constructed MST
    ax = axes[1]
    if show_edges and len(kept_edges) > 0:
        for u, v, _ in kept_edges:
            u, v = int(u), int(v)
            ax.plot([X[u, 0], X[v, 0]], [X[u, 1], X[v, 1]], color="slategray", alpha=0.55, lw=1.0, zorder=1)

    if show_edges and len(cut_edges) > 0:
        for idx_c, (u, v, _) in enumerate(cut_edges):
            u, v = int(u), int(v)
            lbl = "Cut Edge" if idx_c == 0 else None
            ax.plot(
                [X[u, 0], X[v, 0]],
                [X[u, 1], X[v, 1]],
                color="crimson",
                ls="--",
                lw=2.6,
                zorder=3,
                label=lbl,
            )
            if show_midpoints:
                mid_x, mid_y = (X[u, 0] + X[v, 0]) / 2.0, (X[u, 1] + X[v, 1]) / 2.0
                ax.scatter([mid_x], [mid_y], color="crimson", marker="x", s=75, lw=2, zorder=4)

    ax.scatter(X[:, 0], X[:, 1], c=labels_mst, cmap="tab10", s=26, edgecolor="k", linewidth=0.3, zorder=2)
    ax.set_title(
        f"Constructed MST — {strategy_label} (k = {len(np.unique(labels_mst))})",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel("$x_1$ (Standardized)")
    ax.set_ylabel("$x_2$ (Standardized)")
    ax.grid(alpha=0.3)
    if show_edges and len(cut_edges) > 0:
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig


def build_plotly_method_grid(X: np.ndarray, y_true: np.ndarray, preds_dict: dict):
    """Builds a 2x3 interactive Plotly grid comparing all 6 clustering methods."""
    titles = []
    for m_name in METHOD_ORDER:
        ari_val = adjusted_rand_score(y_true, preds_dict[m_name])
        titles.append(f"<b>{m_name}</b><br><span style='font-size:12px'>ARI = {ari_val:.3f}</span>")

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )

    for idx, m_name in enumerate(METHOD_ORDER):
        row = idx // 3 + 1
        col = idx % 3 + 1
        y_pred = preds_dict[m_name]
        colors = [_color_for_label(lbl) for lbl in y_pred]
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
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
        height=640,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
    )
    return fig


def build_matplotlib_method_grid(X: np.ndarray, y_true: np.ndarray, preds_dict: dict, dataset_name: str):
    """Builds a 2x3 Matplotlib grid comparing all 6 clustering methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True, dpi=115)
    for idx, m_name in enumerate(METHOD_ORDER):
        ax = axes[idx // 3, idx % 3]
        y_pred = preds_dict[m_name]
        ari_val = adjusted_rand_score(y_true, y_pred)
        ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab10", s=20, edgecolor="k", linewidth=0.25)
        ax.set_title(m_name, fontsize=11, fontweight="bold")
        ax.text(
            0.04,
            0.88,
            f"ARI = {ari_val:.3f}",
            transform=ax.transAxes,
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
        )
        ax.grid(alpha=0.3)
        if idx // 3 == 1:
            ax.set_xlabel("$x_1$")
        if idx % 3 == 0:
            ax.set_ylabel("$x_2$")

    fig.suptitle(f"Cluster Structure Comparison on {dataset_name} (6 Methods)", fontsize=13.5, fontweight="bold")
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
        "Benchmark Topology",
        options=["Two Moons", "Concentric Circles", "Two Spirals", "Anisotropic Blobs"],
        index=0,
    )
    n_samples = st.slider("Number of Samples (N)", min_value=100, max_value=600, value=300, step=20)
    noise_level = st.slider(
        "Gaussian Noise (σ)", min_value=0.01, max_value=0.25, value=0.05, step=0.01
    )
    seed_val = st.slider("Random Seed", min_value=0, max_value=200, value=SEED, step=1)

    st.divider()

    # Section 2: MST Cutting Strategy
    st.header("2. MST Cutting Strategy")
    cutting_strategy = st.radio(
        "Cutting Criterion",
        options=[
            "Known k: Top (k - 1) Heaviest Edges",
            "Unknown k: Zahn's Local Inconsistency Cut",
        ],
        index=0,
    )
    is_inconsistency = cutting_strategy.startswith("Unknown k")

    default_k = DATASET_DEFAULT_K[selected_dataset]
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
        "Min Cluster Size (Pruning)",
        min_value=2,
        max_value=15,
        value=6,
        step=1,
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
all_datasets = generate_all_datasets(
    n_samples=n_samples, noise=noise_level, random_state=seed_val
)
X, y_true, k_true = all_datasets[selected_dataset]

D = compute_distance_matrix(X)
mst_edges, w_custom = prims_mst_from_scratch(D)
w_scipy, w_diff, scipy_match = verify_mst_against_scipy(D, w_custom)

if is_inconsistency:
    labels_active, kept_active, cut_active, _, cut_stats_active = mst_cluster_inconsistency(
        len(X), mst_edges, t=t_threshold, depth=bfs_depth, min_cluster_size=min_cluster_size
    )
    strategy_short = f"Inconsistency (t={t_threshold:.1f}, d={bfs_depth})"
else:
    labels_active, kept_active, cut_active = mst_cluster_top_k(len(X), mst_edges, k=target_k)
    cut_stats_active = None
    strategy_short = f"Top (k-1) Cut (k={target_k})"

active_metrics = evaluate_predictions(X, y_true, labels_active)
discovered_k = int(len(np.unique(labels_active)))


# ==============================================================================
# 5. Main Dashboard Header & Tabs
# ==============================================================================
st.title("MST-Based Clustering for Non-Linearly Separable Patterns")
st.caption(
    f"Active Dataset: **{selected_dataset}** (N = {len(X)}, σ = {noise_level:.2f}, Seed = {seed_val})  |  "
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
        badge_text = f"SciPy Match Δ = {w_diff:.2f}" if scipy_match else f"Mismatch Δ = {w_diff:.4f}"
        st.metric(
            label="Total MST Weight (Prim's O(N²))",
            value=f"{w_custom:.2f}",
            delta=badge_text,
            delta_color="normal" if scipy_match else "inverse",
        )
    with col2:
        st.metric(
            label="Discovered Clusters (k)",
            value=f"{discovered_k}",
            delta=f"Ground Truth k = {k_true}",
            delta_color="off",
        )
    with col3:
        st.metric(label="MST ARI", value=f"{active_metrics['ARI']:.3f}")
    with col4:
        st.metric(label="MST NMI", value=f"{active_metrics['NMI']:.3f}")

    if scipy_match:
        st.success(
            f"**Exact Prim's ↔ SciPy Verification Passed:** Custom $O(N^2)$ Prim's weight "
            f"(`{w_custom:.6f}`) matches `scipy.sparse.csgraph.minimum_spanning_tree` "
            f"(`{w_scipy:.6f}`) with difference `{w_diff:.2e}`."
        )

    if use_plotly:
        fig_mst = build_plotly_mst_explorer(
            X=X,
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
        )
        st.plotly_chart(fig_mst, width="stretch")
    else:
        fig_mpl = build_matplotlib_mst_explorer(
            X=X,
            y_true=y_true,
            labels_mst=labels_active,
            kept_edges=kept_active,
            cut_edges=cut_active,
            dataset_name=selected_dataset,
            strategy_label=strategy_short,
            show_edges=show_mst_edges,
            show_midpoints=show_cut_midpoints,
        )
        st.pyplot(fig_mpl, width="stretch")
        plt.close(fig_mpl)

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
    st.subheader(f"6-Method Cluster Structure Comparison — {selected_dataset}")
    preds_dict = run_all_clustering_methods(
        X,
        k=target_k,
        inconsistency_t=t_threshold,
        depth=bfs_depth,
        min_cluster_size=min_cluster_size,
        random_state=seed_val,
    )

    if use_plotly:
        fig_grid = build_plotly_method_grid(X, y_true, preds_dict)
        st.plotly_chart(fig_grid, width="stretch")
    else:
        fig_grid_mpl = build_matplotlib_method_grid(X, y_true, preds_dict, selected_dataset)
        st.pyplot(fig_grid_mpl, width="stretch")
        plt.close(fig_grid_mpl)

    st.divider()
    col_tbl, col_heat = st.columns([1.05, 1.15])

    metrics_rows = []
    for m_name in METHOD_ORDER:
        m_scores = evaluate_predictions(X, y_true, preds_dict[m_name])
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
        heat_fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(heat_fig, width="stretch")

# ------------------------------------------------------------------------------
# TAB 3 — Chaining Effect & Sensitivity Lab
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Live Sensitivity & Failure-Mode Experiments")
    col_n, col_t = st.columns(2)

    with col_n:
        st.markdown(f"#### 1. Noise Sensitivity & The Chaining Effect ({selected_dataset})")
        with st.spinner("Running noise sweep (σ = 0.02 → 0.22)..."):
            noise_df = run_noise_sweep(
                dataset_name=selected_dataset,
                n_samples=n_samples,
                k_target=target_k,
                t_val=t_threshold,
                depth=bfs_depth,
                min_cluster_size=min_cluster_size,
                base_seed=seed_val,
            )
        noise_agg = noise_df.groupby(["Noise", "Method"], as_index=False)["ARI"].mean()

        fig_noise = px.line(
            noise_agg,
            x="Noise",
            y="ARI",
            color="Method",
            markers=True,
            title=f"ARI Degradation vs. Gaussian Noise σ ({selected_dataset})",
        )
        fig_noise.add_vline(
            x=noise_level,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Current σ={noise_level:.2f}",
        )
        if selected_dataset == "Two Moons":
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
              <li><b>Fix 1 — Min-Cluster-Size Pruning:</b> Reject candidate cuts that create fragments smaller than <code>min_cluster_size</code> (implemented in our Zahn's cut).</li>
              <li><b>Fix 2 — Mutual Reachability Distance (HDBSCAN):</b> Replace Euclidean distance with <code>max{core_m(x_i), core_m(x_j), d(x_i, x_j)}</code>, pushing sparse bridge outliers away from dense manifolds before MST construction.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("4. Time & Space Complexity Reference")
    complexity_df = pd.DataFrame(
        [
            {
                "Algorithm": "Custom Prim's MST + Cut",
                "Time Complexity": "O(N²d + N²)",
                "Space Complexity": "O(N²) (Distance Matrix)",
                "Geometric Assumption": "Arbitrary manifold (density-connected)",
            },
            {
                "Algorithm": "Single-Linkage (SLINK / Kd-tree)",
                "Time Complexity": "O(N²) worst-case, O(N log N) low-d",
                "Space Complexity": "O(N) to O(N²)",
                "Geometric Assumption": "Arbitrary manifold (dual of MST Top-k)",
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
