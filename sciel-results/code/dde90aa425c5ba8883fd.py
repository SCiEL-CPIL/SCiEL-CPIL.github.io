"""Improved program for Maximum Independent Set via enhanced spectral partitioning and local feature search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_decode(adjacency, scores):
    """Greedy decoder: pick vertices in descending score order, skipping neighbours."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    chosen = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if chosen[v] or forbidden[v]:
            continue
        chosen[v] = True
        count += 1
        forbidden[adjacency[v]] = True
    return count


def _greedy_decode_vectorized(adjacency, scores):
    """Fast greedy decode using precomputed neighbor lists for speed."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    chosen = np.zeros(n, dtype=bool)
    forbidden = np.zeros(n, dtype=bool)
    # Precompute neighbor indices for faster access
    neighbor_lists = [np.where(adjacency[v])[0] for v in range(n)]
    count = 0
    for v in order:
        if chosen[v] or forbidden[v]:
            continue
        chosen[v] = True
        count += 1
        for u in neighbor_lists[v]:
            forbidden[u] = True
    return count


def _compute_degree(adjacency):
    return adjacency.sum(axis=1).astype(float)


def _compute_neighbor_degree_sum(adjacency):
    deg = adjacency.sum(axis=1)
    return (adjacency @ deg).astype(float)


def _compute_coreness(adjacency):
    """Compute k-core number for each vertex using iterative peeling."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(int).copy()
    core = np.zeros(n, dtype=int)
    removed = np.zeros(n, dtype=bool)
    current_deg = deg.copy()
    
    active_count = n
    for _ in range(n):
        if active_count == 0:
            break
        active_mask = ~removed
        active_indices = np.where(active_mask)[0]
        if len(active_indices) == 0:
            break
        min_deg = current_deg[active_indices].min()
        candidates = active_indices[current_deg[active_indices] == min_deg]
        v = candidates[0]
        removed[v] = True
        core[v] = min_deg
        active_count -= 1
        for u in np.where(adjacency[v])[0]:
            if not removed[u]:
                current_deg[u] -= 1

    return core.astype(float)


def _compute_effective_degree(adjacency):
    """
    Effective degree: degree(v) - edges_in_N(v)
    Lower effective degree means neighbors are less interconnected.
    """
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    return degree - edges_in_N


def _compute_neighborhood_sparsity(adjacency):
    """
    Neighborhood sparsity ratio: fraction of neighbor pairs that are NON-adjacent.
    """
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    d_v = degree
    total_pairs = d_v * (d_v - 1.0) / 2.0
    sparsity_ratio = np.where(total_pairs > 0,
                              1.0 - edges_in_N / np.maximum(total_pairs, 1.0),
                              1.0)
    return sparsity_ratio


def _compute_mean_neighbor_degree(adjacency):
    """Mean degree of neighbors for each vertex."""
    degree = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degree
    return neighbor_deg_sum / np.maximum(degree, 1.0)


def _compute_spectral_features(adjacency):
    """
    Compute spectral features focusing on partitioning structure.
    Returns:
        - fiedler_vec_centered: Centered Fiedler vector
        - small_side_bonus: 1.0 for vertices in the smaller partition, 0.0 otherwise
        - dist_from_cut: Normalized distance from the cut boundary (high = far from cut)
        - cross_cut_degree: Number of neighbors on the opposite side of the cut
    """
    n = adjacency.shape[0]
    if n < 2:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n))

    A = adjacency.astype(float)
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n))

    # Find the Fiedler vector (eigenvector of the smallest positive eigenvalue)
    tol = 1e-8
    fiedler_idx = 1
    for i in range(len(eigvals)):
        if eigvals[i] > tol:
            fiedler_idx = i
            break
    else:
        fiedler_idx = 1

    fiedler_vec = eigvecs[:, fiedler_idx].copy()
    fiedler_vec_centered = fiedler_vec - fiedler_vec.mean()

    # Determine partition sides
    if np.all(fiedler_vec_centered >= 0):
        median_val = np.median(fiedler_vec_centered)
        side = np.where(fiedler_vec_centered >= median_val, 1.0, -1.0)
    elif np.all(fiedler_vec_centered <= 0):
        median_val = np.median(fiedler_vec_centered)
        side = np.where(fiedler_vec_centered <= median_val, -1.0, 1.0)
    else:
        side = np.where(fiedler_vec_centered >= 0, 1.0, -1.0)

    # Identify the smaller partition
    pos_count = np.sum(side == 1.0)
    neg_count = np.sum(side == -1.0)
    
    # We want to prioritize the smaller partition to minimize cut edges
    if pos_count <= neg_count:
        small_side_mask = (side == 1.0)
    else:
        small_side_mask = (side == -1.0)
    
    small_side_bonus = np.where(small_side_mask, 1.0, 0.0)

    # Distance from cut: how far is the vertex from the boundary (0 value)?
    # Vertices far from the cut are "safer" to pick as they have fewer cross-cut neighbors.
    max_abs = np.max(np.abs(fiedler_vec_centered))
    if max_abs > 1e-12:
        dist_from_cut = np.abs(fiedler_vec_centered) / max_abs
    else:
        dist_from_cut = np.zeros(n)

    # Cross-cut degree: number of neighbors on the opposite side (vectorized)
    # side is 1 or -1. Cross-cut if side[u] * side[v] == -1
    # For each v: sum over neighbors u of (side[v] * side[u] == -1)
    # = sum over neighbors u of (1 - side[v]*side[u]) / 2
    # = deg(v)/2 - sum over neighbors u of side[v]*side[u] / 2
    # = deg(v)/2 - side[v] * (sum over neighbors u of side[u]) / 2
    side_float = side.astype(float)
    neighbor_side_sum = adjacency.astype(float) @ side_float
    cross_cut_degree = 0.5 * degree - 0.5 * side_float * neighbor_side_sum

    return (fiedler_vec_centered, small_side_bonus, dist_from_cut, cross_cut_degree)


def _search_best_scores(adjacency, time_limit=110.0):
    """
    Search for a good static scoring function.
    Focuses on combining local density metrics with global spectral partitioning insights.
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute base features
    degree = _compute_degree(adjacency)
    nds = _compute_neighbor_degree_sum(adjacency)
    coreness = _compute_coreness(adjacency)
    effective_deg = _compute_effective_degree(adjacency)
    sparsity_ratio = _compute_neighborhood_sparsity(adjacency)
    mean_neighbor_deg = _compute_mean_neighbor_degree(adjacency)

    # Compute enhanced spectral features
    fiedler_vec, small_side_bonus, dist_from_cut, cross_cut_degree = _compute_spectral_features(adjacency)

    # Compute clustering coefficient
    def _compute_clustering_coefficient(adjacency):
        A_int = adjacency.astype(np.int32)
        degree = A_int.sum(axis=1).astype(float)
        M = A_int @ A_int.T
        triangles = np.diag(M).astype(float) / 2.0
        denom = degree * (degree - 1.0)
        clustering = np.where(denom > 0, triangles / denom, 0.0)
        return clustering

    clustering_coeff = _compute_clustering_coefficient(adjacency)

    # Normalize features
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    deg_norm = normalize(degree)
    eff_deg_norm = normalize(effective_deg)
    sparsity_norm = normalize(sparsity_ratio)
    core_norm = normalize(coreness)
    mnd_norm = normalize(mean_neighbor_deg)
    cluster_norm = normalize(clustering_coeff)
    
    # Spectral features
    cross_cut_norm = normalize(cross_cut_degree)

    features = {
        '-deg': -deg_norm,
        '-eff': -eff_deg_norm,
        'sparsity': sparsity_norm,
        'core': core_norm,
        'mnd': mnd_norm,
        'cluster': cluster_norm,
        'small_side': small_side_bonus,
        'dist_cut': dist_from_cut,
        '-cross_cut': -cross_cut_norm,
    }

    feature_names = list(features.keys())
    feature_arrays = [features[name] for name in feature_names]
    num_features = len(feature_arrays)

    # Initial best: simple degree-based
    best_scores = -degree
    best_count = _greedy_decode(adjacency, best_scores)

    # Structured combinations emphasizing spectral partitioning
    structured_combos = [
        # (w_deg, w_eff, w_sparsity, w_core, w_mnd, w_cluster, w_small, w_dist, w_cross)
        # Local only baselines
        (-1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, -1, 0, 0, 0, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0),
        (-0.7, -0.3, 0.1, 0, 0, 0, 0, 0, 0),
        
        # Spectral heavy
        (0, 0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 0, -1),
        (0, 0, 0, 0, 0, 0, 1, 1, -1),
        (0, 0, 0, 0, 0, 0, 1, 0.5, -0.5),
        (0, 0, 0, 0, 0, 0, 0.5, 1, -0.5),
        
        # Mixed: Local + Spectral
        (-0.5, -0.5, 0, 0, 0, 0, 0.5, 0.5, -0.5),
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.4, 0.4, -0.4),
        (-0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6, -0.6),
        (-0.4, -0.4, 0.1, 0.1, 0, 0, 0.5, 0.5, -0.5),
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.5, 0.5, -0.5),
        (-0.2, -0.2, 0.3, 0.2, 0.2, 0.2, 0.4, 0.4, -0.4),
        
        # Balanced
        (-0.4, -0.4, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, -0.3),
        (-0.3, -0.3, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, -0.4),
        (-0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, -0.5),
        (-0.1, -0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, -0.4),
    ]

    for weights in structured_combos:
        if time.time() > deadline - 5:
            break
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Random search with bias towards spectral features
    rng = np.random.default_rng(42)
    num_random_tries = 8000
    
    for i in range(num_random_tries):
        if time.time() > deadline - 5:
            break
        
        # Weights for local features (negative to prefer low degree/effective degree)
        w_deg = rng.uniform(-0.8, 0)
        w_eff = rng.uniform(-0.8, 0)
        w_sparsity = rng.uniform(0, 0.5)
        w_core = rng.uniform(0, 0.4)
        w_mnd = rng.uniform(0, 0.4)
        w_cluster = rng.uniform(0, 0.4)
        
        # Weights for spectral features (positive to prefer small side, high dist, low cross-cut)
        w_small = rng.uniform(0, 1.5)
        w_dist = rng.uniform(0, 1.2)
        w_cross = rng.uniform(-1.2, 0)
        
        weights = [w_deg, w_eff, w_sparsity, w_core, w_mnd, w_cluster, w_small, w_dist, w_cross]
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode_vectorized(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Local search to polish the best found scores
    local_search_steps = 5000
    current_scores = best_scores.copy()
    current_count = best_count
    step_size = 0.08

    for i in range(local_search_steps):
        if time.time() > deadline - 2:
            break
        
        # Perturb the scores with adaptive step size
        noise = rng.normal(0, step_size, size=n)
        perturbed = current_scores + noise
        count = _greedy_decode_vectorized(adjacency, perturbed)
        
        if count > current_count:
            current_count = count
            current_scores = perturbed.copy()
            if count > best_count:
                best_count = count
                best_scores = perturbed.copy()
            step_size = min(0.15, step_size * 1.03)
        elif i % 5 == 0:
            step_size = max(0.01, step_size * 0.92)

    return best_scores


def priority(adjacency):
    """
    Give every vertex a score. The greedy decoder picks vertices in descending
    score order, skipping neighbours of picked vertices.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]

    if n == 0:
        return np.array([], dtype=float)

    start_time = time.time()
    time_limit = 110.0

    try:
        scores = _search_best_scores(adjacency, time_limit=time_limit)
    except Exception:
        # Fallback to simple degree-based scoring
        scores = -adjacency.sum(axis=1).astype(float)

    return scores
# EVOLVE-BLOCK-END
