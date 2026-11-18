# MST-Based Clustering for Non-Linearly Separable Patterns

An interactive research project and live academic viva dashboard exploring **Minimum Spanning Tree (MST) Clustering** for recovering non-convex, density-connected manifolds where centroid-based methods ($k$-Means, $k$-Medoids, GMM) fail.

---

## Repository Structure

```text
PM/
├── MST_Clustering_Project.ipynb   # Research notebook with derivations, experiments & benchmarks
├── web_app/
│   ├── app.py                     # Interactive Streamlit dashboard for live demos & viva
│   └── requirements.txt           # Python dependencies
└── README.md
```

---

## Key Features

### 1. Core MST & Anti-Chaining Algorithms
- **Custom Prim's Algorithm ($O(N^2)$):** Implemented from scratch and cross-verified against `scipy.sparse.csgraph.minimum_spanning_tree` ($\Delta = 0.00$).
- **Scalable Sparse $k$-NN Graph MST ($O(N \log N)$):** Uses `kneighbors_graph` with automatic inter-component bridge fallback for larger datasets.
- **Mutual Reachability Distance (Anti-Chaining):** HDBSCAN-style edge weighting
  $$d_{\text{mreach}}(x_i, x_j) = \max\{\text{core}_m(x_i),\ \text{core}_m(x_j),\ d(x_i, x_j)\}$$
  to push sparse noise/bridge points away from dense manifolds and prevent chaining collapse.
- **Two Edge-Cutting Strategies:**
  - **Known $k$:** Global cut of the $k - 1$ heaviest edges (theoretically dual to Single-Linkage Agglomerative Clustering).
  - **Unknown $k$ (Zahn's Local Inconsistency Cut):** Cuts edges where $w(e) > \mu + t\sigma$ within a BFS neighborhood of depth $d$, with `min_cluster_size` pruning or explicit **Noise (`-1`)** outlier labeling.

### 2. Datasets & Multi-Dimensional Support
- **Synthetic Topologies:** `Two Moons`, `Concentric Circles`, `Two Spirals`, and `Anisotropic Blobs` (standardized via `StandardScaler`).
- **Real-World Benchmarks:** `Iris` (4D), `Wine` (13D), and `Breast Cancer` (30D) — MST is built in the full standardized feature space and projected to 2D via PCA for visualization.
- **Custom CSV Upload:** Upload any numeric `.csv`, select feature columns and an optional ground-truth label column.

### 3. 8-Method Comparison & Evaluation
Compares **MST (Top $k-1$)** and **MST (Inconsistency)** against **Single-Linkage**, **DBSCAN**, **Spectral Clustering**, **$k$-Means**, **Pure-NumPy $k$-Medoids (PAM)**, and **GMM (full covariance)** using:
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**
- **Cluster Purity**
- **Silhouette Score**

### 4. Interactive Dashboard Tabs (`web_app/app.py`)
1. **Live MST Graph & Cut Explorer:** KPI cards with SciPy verification badge, interactive Plotly (or Matplotlib) Ground Truth vs. MST plot with edge-weight & local $\mu + t\sigma$ hover tooltips, MST edge-weight distribution histogram, and a step-by-step Prim's growth animator.
2. **Method Comparison Grid & Metrics:** $2 \times 4$ cluster comparison grid, styled metrics table, interactive heatmap, and one-click CSV downloads for clustered labels and metrics.
3. **Chaining Effect & Sensitivity Lab:** Live Gaussian noise sweep ($\sigma \in [0.02, 0.22]$) comparing Euclidean MST vs. Mutual Reachability MST, Zahn's threshold sweep ($t \in [1.3, 3.3]$), and empirical runtime scaling ($O(N^2)$ vs. $O(N \log N)$).
4. **Viva Notes & Complexity Reference:** Concise theoretical notes on Voronoi hyperplanes vs. geodesic skeletons, Silhouette pitfalls on non-convex shapes, chaining mitigation, and Big-O time/space complexity reference.

---

## Quick Start

### 1. Create a Virtual Environment & Install Dependencies
```powershell
python -m venv web_app\.venv
web_app\.venv\Scripts\activate
pip install -r web_app\requirements.txt
```

### 2. Launch the Streamlit Dashboard
```powershell
streamlit run web_app/app.py
```

---

## Complexity Reference

| Algorithm | Time Complexity | Space Complexity | Geometric Assumption |
| :--- | :--- | :--- | :--- |
| **Custom Dense Prim's MST + Cut** | $O(N^2 d + N^2)$ | $O(N^2)$ | Arbitrary manifold (density-connected) |
| **Sparse $k$-NN Graph MST + Cut** | $O(N \log N \cdot d)$ | $O(N \cdot k_{nn})$ | Arbitrary manifold (scalable) |
| **Mutual Reachability MST** | $O(N^2 d)$ | $O(N^2)$ | Arbitrary manifold + anti-chaining |
| **Single-Linkage** | $O(N^2)$ worst-case | $O(N)$ to $O(N^2)$ | Arbitrary manifold (dual of MST Top-$k$) |
| **DBSCAN** | $O(N \log N)$ / $O(N^2)$ | $O(N)$ | Density-connected regions + noise |
| **Spectral Clustering** | $O(N^3)$ | $O(N^2)$ | Non-convex graph Laplacian manifolds |
| **$k$-Means (Lloyd's)** | $O(N \cdot k \cdot d \cdot I)$ | $O(Nd + kd)$ | Convex Voronoi cells |
| **$k$-Medoids (PAM)** | $O(N^2 d + k(N-k)^2 I)$ | $O(N^2)$ | Convex Voronoi cells |
| **Gaussian Mixture (GMM)** | $O(N \cdot k \cdot d^3 \cdot I)$ | $O(Nk + kd^2)$ | Elliptical Gaussian components |
