"""Improved program for Maximum Independent Set via spectral + local feature search."""
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

    for _ in range(n):
        active_mask = ~removed
        if not active_mask.any():
            break
        active_indices = np.where(active_mask)[0]
        min_deg = current_deg[active_indices].min()
        candidates = active_indices[current_deg[active_indices] == min_deg]
        v = candidates[0]
        removed[v] = True
        core[v] = min_deg
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
    Compute spectral features using the Fiedler vector (eigenvector of λ₂ of Laplacian).
    Returns:
        - fiedler_vec: the Fiedler vector values (centered)
        - partition_side: +1 for vertices on the "smaller" side, -1 for the larger side
        - distance_from_cut: |fiedler_vec| normalized, indicating how far from the cut boundary
        - cross_cut_degree: number of neighbors on the opposite side of the cut
        - spectral_gap: the second-smallest eigenvalue (connectivity measure)
    """
    n = adjacency.shape[0]
    if n < 2:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), 0.0)

    A = adjacency.astype(float)
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), 0.0)

    # Find the index of the smallest positive eigenvalue (Fiedler value)
    tol = 1e-8
    fiedler_idx = 1
    for i in range(len(eigvals)):
        if eigvals[i] > tol:
            fiedler_idx = i
            break
    else:
        fiedler_idx = 1

    fiedler_vec = eigvecs[:, fiedler_idx].copy()
    spectral_gap = float(eigvals[fiedler_idx]) if fiedler_idx < len(eigvals) else 0.0

    fiedler_vec_centered = fiedler_vec - fiedler_vec.mean()

    if np.all(fiedler_vec_centered >= 0):
        median_val = np.median(fiedler_vec_centered)
        side = np.where(fiedler_vec_centered >= median_val, 1.0, -1.0)
    elif np.all(fiedler_vec_centered <= 0):
        median_val = np.median(fiedler_vec_centered)
        side = np.where(fiedler_vec_centered <= median_val, -1.0, 1.0)
    else:
        side = np.where(fiedler_vec_centered >= 0, 1.0, -1.0)

    pos_count = np.sum(side == 1.0)
    neg_count = np.sum(side == -1.0)
    if pos_count <= neg_count:
        small_side = 1.0
    else:
        small_side = -1.0

    partition_side = np.where(side == small_side, 1.0, -1.0)

    max_abs = np.max(np.abs(fiedler_vec_centered))
    if max_abs > 1e-12:
        distance_from_cut = np.abs(fiedler_vec_centered) / max_abs
    else:
        distance_from_cut = np.zeros(n)

    cross_cut_degree = np.zeros(n, dtype=float)
    for v in range(n):
        neighbors = np.where(adjacency[v])[0]
        if len(neighbors) > 0:
            cross_cut_degree[v] = np.sum(side[neighbors] != side[v])

    return (fiedler_vec_centered, partition_side, distance_from_cut, cross_cut_degree, spectral_gap)


def _compute_clustering_coefficient(adjacency):
    """Compute local clustering coefficient for each vertex."""
    n = adjacency.shape[0]
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    total_pairs = degree * (degree - 1.0) / 2.0
    cc = np.where(total_pairs > 0, edges_in_N / np.maximum(total_pairs, 1.0), 0.0)
    return cc


def _search_best_scores(adjacency, time_limit=110.0):
    """
    Search for a good static scoring function by trying various linear combinations
    of degree, neighbor degree sum, coreness, effective degree, neighborhood sparsity,
    spectral features, and clustering coefficient.
    Returns the best scores array found.
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
    clustering_coef = _compute_clustering_coefficient(adjacency)

    # Compute spectral features
    (fiedler_vec, partition_side, distance_from_cut,
     cross_cut_degree, spectral_gap) = _compute_spectral_features(adjacency)

    # Normalize features to [0, 1] range for stable combination
    def normalize(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def normalize_signed(x):
        """Normalize signed features to [-1, 1] range."""
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        mid = (mx + mn) / 2.0
        half_range = (mx - mn) / 2.0
        if half_range < 1e-12:
            return np.zeros_like(x)
        return (x - mid) / half_range

    deg_norm = normalize(degree)
    nds_norm = normalize(nds)
    core_norm = normalize(coreness)
    eff_deg_norm = normalize(effective_deg)
    sparsity_norm = normalize(sparsity_ratio)
    mnd_norm = normalize(mean_neighbor_deg)
    cc_norm = normalize(clustering_coef)

    # Spectral features
    fiedler_norm = normalize_signed(fiedler_vec)
    part_side_norm = partition_side
    dist_cut_norm = normalize(distance_from_cut)
    cross_cut_norm = normalize(cross_cut_degree)

    best_scores = -degree
    best_count = _greedy_decode(adjacency, best_scores)

    # Feature vectors
    features = {
        '-deg': -deg_norm,
        '-eff': -eff_deg_norm,
        'sparsity': sparsity_norm,
        'core': core_norm,
        'mnd': mnd_norm,
        'part_side': part_side_norm,
        'dist_cut': dist_cut_norm,
        '-cross_cut': -cross_cut_norm,
        '-cc': -cc_norm,
    }

    feature_names = list(features.keys())
    feature_arrays = [features[name] for name in feature_names]
    num_features = len(feature_arrays)

    # Structured combinations - now with 9 features
    structured_combos = [
        # (w_deg, w_eff, w_sparsity, w_core, w_mnd, w_part, w_dist, w_cross, w_cc)
        (-1, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, -1, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0),
        (-0.3, -0.3, 0.2, 0, 0, 0, 0, 0, 0),
        (-0.2, -0.4, 0.2, 0, 0, 0, 0, 0, 0),
        (-0.4, -0.2, 0.2, 0, 0, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0.2, 0, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0.2, 0, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0.3, 0, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0.3, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0.3, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0.3),
        # Mixed local
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0, 0, 0, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0, 0, 0, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0, 0, 0, 0.1),
        # Add spectral
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1),
        # Heavier spectral
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.4, 0.2, 0.2, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.4, 0.2, 0.2, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.4, 0.2, 0.2, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1, 0.4, 0.2, 0.2, 0.1),
        # Distance from cut emphasis
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.2, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.2, 0.4, 0.2, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.2, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1, 0.2, 0.4, 0.2, 0.1),
        # Cross-cut degree emphasis
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.2, 0.2, 0.4, 0.1),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.2, 0.2, 0.4, 0.1),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.4, 0.1),
        (-0.5, -0.5, 0.2, 0.1, 0.1, 0.2, 0.2, 0.4, 0.1),
        # Clustering emphasis
        (-0.3, -0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3),
        (-0.2, -0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3),
        (-0.4, -0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3),
        # All features moderate
        (-0.3, -0.3, 0.2, 0.15, 0.15, 0.25, 0.15, 0.15, 0.15),
        (-0.2, -0.3, 0.25, 0.15, 0.15, 0.25, 0.15, 0.15, 0.15),
        (-0.3, -0.2, 0.25, 0.15, 0.15, 0.25, 0.15, 0.15, 0.15),
    ]

    for weights in structured_combos:
        if time.time() > deadline - 5:
            break
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Random search with more diverse weight ranges
    rng = np.random.default_rng(42)

    num_random_tries = 4000
    for i in range(num_random_tries):
        if time.time() > deadline - 5:
            break
        w1 = rng.uniform(-1.2, 0)
        w2 = rng.uniform(-1.2, 0)
        w3 = rng.uniform(0, 0.6)
        w4 = rng.uniform(0, 0.6)
        w5 = rng.uniform(0, 0.4)
        w6 = rng.uniform(-0.3, 0.6)
        w7 = rng.uniform(0, 0.5)
        w8 = rng.uniform(-0.5, 0)
        w9 = rng.uniform(-0.4, 0)
        weights = [w1, w2, w3, w4, w5, w6, w7, w8, w9]
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Local search with adaptive step size
    local_search_steps = 2500
    current_scores = best_scores.copy()
    current_count = best_count
    step_size = 0.08

    for i in range(local_search_steps):
        if time.time() > deadline - 5:
            break
        noise = rng.normal(0, step_size, size=n)
        perturbed = current_scores + noise
        count = _greedy_decode(adjacency, perturbed)
        if count > current_count:
            current_count = count
            current_scores = perturbed.copy()
            if count > best_count:
                best_count = count
                best_scores = perturbed.copy()
            step_size = min(0.15, step_size * 1.1)
        elif i % 20 == 0:
            step_size = max(0.03, step_size * 0.85)

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
        scores = -adjacency.sum(axis=1).astype(float)

    return scores
# EVOLVE-BLOCK-END
