"""Maximum Independent Set via multi-objective score search with spectral structure."""
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
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    return degree - edges_in_N


def _compute_neighborhood_sparsity(adjacency):
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    total_pairs = degree * (degree - 1.0) / 2.0
    sparsity_ratio = np.where(total_pairs > 0,
                              1.0 - edges_in_N / np.maximum(total_pairs, 1.0),
                              1.0)
    return sparsity_ratio


def _compute_mean_neighbor_degree(adjacency):
    degree = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degree
    return neighbor_deg_sum / np.maximum(degree, 1.0)


def _compute_spectral_features(adjacency):
    n = adjacency.shape[0]
    if n < 2:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), 0.0)

    A = adjacency.astype(float)
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n), 0.0)

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
    neighbors_of_v = A
    for v in range(n):
        neighbors = np.where(adjacency[v])[0]
        if len(neighbors) > 0:
            cross_cut_degree[v] = np.sum(side[neighbors] != side[v])

    # New: bipartite fitness — fraction of neighbors on the SAME side
    same_side_count = np.zeros(n, dtype=float)
    for v in range(n):
        neighbors = np.where(adjacency[v])[0]
        if len(neighbors) > 0:
            same_side_count[v] = np.sum(side[neighbors] == side[v])
            same_side_count[v] /= len(neighbors)
        else:
            same_side_count[v] = 1.0

    return (fiedler_vec_centered, partition_side, distance_from_cut,
            cross_cut_degree, same_side_count, spectral_gap)


def _compute_clustering_coefficient(adjacency):
    n = adjacency.shape[0]
    A_int = adjacency.astype(np.int32)
    degree = A_int.sum(axis=1).astype(float)
    M = A_int @ A_int.T
    S = (A_int * M).sum(axis=1).astype(float)
    edges_in_N = 0.5 * S
    total_pairs = degree * (degree - 1.0) / 2.0
    cc = np.where(total_pairs > 0, edges_in_N / np.maximum(total_pairs, 1.0), 0.0)
    return cc


def _compute_betweenness_proxy(adjacency):
    """Approximate betweenness centrality using 2-hop connectivity."""
    n = adjacency.shape[0]
    A_int = adjacency.astype(np.float64)
    two_hop = (A_int @ A_int).astype(np.float64)
    two_hop = np.where(adjacency, 0.0, two_hop)
    return two_hop.sum(axis=1).astype(float)


def _normalize(x, signed=False):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    if signed:
        mid = (mx + mn) / 2.0
        half = (mx - mn) / 2.0
        if half < 1e-12:
            return np.zeros_like(x)
        return (x - mid) / half
    return (x - mn) / (mx - mn)


def _search_best_scores(adjacency, time_limit=125.0):
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
    betweenness = _compute_betweenness_proxy(adjacency)

    (fiedler_vec, partition_side, distance_from_cut,
     cross_cut_degree, same_side_frac, spectral_gap) = _compute_spectral_features(adjacency)

    # Normalize features
    deg_norm = _normalize(degree)
    nds_norm = _normalize(nds)
    core_norm = _normalize(coreness)
    eff_deg_norm = _normalize(effective_deg)
    sparsity_norm = _normalize(sparsity_ratio)
    mnd_norm = _normalize(mean_neighbor_deg)
    cc_norm = _normalize(clustering_coef)
    btw_norm = _normalize(betweenness)
    fiedler_norm = _normalize(fiedler_vec, signed=True)
    part_side_norm = partition_side
    dist_cut_norm = _normalize(distance_from_cut)
    cross_cut_norm = _normalize(cross_cut_degree)
    same_side_norm = _normalize(same_side_frac)

    feature_arrays = [
        -deg_norm,
        -eff_deg_norm,
        sparsity_norm,
        core_norm,
        mnd_norm,
        part_side_norm,
        dist_cut_norm,
        -cross_cut_norm,
        -cc_norm,
        same_side_norm,
        -btw_norm,
    ]

    feature_names = [
        '-deg', '-eff', 'sparsity', 'core', 'mnd',
        'part_side', 'dist_cut', '-cross_cut', '-cc',
        'same_side', '-btw'
    ]

    num_features = len(feature_arrays)

    best_scores = -degree
    best_count = _greedy_decode(adjacency, best_scores)

    # Phase A: Structured combinations
    structured_combos = [
        # Individual features
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        # Pairs
        [0.6, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.3, 0.3, 0.4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2, 0.3, 0.2, 0.3, 0, 0, 0, 0, 0, 0, 0],
        [0.4, 0.2, 0.4, 0, 0, 0, 0, 0, 0, 0, 0],
        # With spectral
        [0.3, 0.3, 0.2, 0, 0, 0.2, 0.1, 0.1, 0, 0.1, 0],
        [0.2, 0.3, 0.2, 0, 0, 0.1, 0.1, 0.1, 0, 0.3, 0],
        [0.3, 0.2, 0.2, 0, 0, 0.1, 0.1, 0.1, 0, 0.2, 0.1],
        # Heavy spectral
        [0.2, 0.2, 0.1, 0, 0, 0.3, 0.2, 0.2, 0.1, 0.3, 0.1],
        [0.1, 0.2, 0.1, 0, 0, 0.4, 0.2, 0.2, 0.1, 0.3, 0.1],
        # All moderate
        [0.2, 0.2, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.15, 0.1],
        [0.15, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15],
        # Same-side emphasis
        [0.2, 0.2, 0.1, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1],
        [0.1, 0.2, 0.1, 0, 0, 0.2, 0.1, 0.1, 0.1, 0.4, 0.1],
    ]

    for weights in structured_combos:
        if time.time() > deadline - 8:
            break
        weights = np.array(weights, dtype=float)
        # Normalize weights to sum to 1
        wsum = np.abs(weights).sum()
        if wsum > 0:
            weights = weights / wsum
        scores = sum(w * f for w, f in zip(weights, feature_arrays))
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Phase B: Random search with diverse weights
    rng = np.random.default_rng(42)

    num_random_tries = 3000
    for i in range(num_random_tries):
        if time.time() > deadline - 8:
            break
        weights = np.array([
            rng.uniform(-1.5, -0.2),   # deg
            rng.uniform(-1.5, -0.2),   # eff
            rng.uniform(0.2, 0.8),     # sparsity
            rng.uniform(0, 0.6),       # core
            rng.uniform(0, 0.5),       # mnd
            rng.uniform(-0.5, 0.8),    # part_side
            rng.uniform(0, 0.6),       # dist_cut
            rng.uniform(-0.6, 0),      # cross_cut
            rng.uniform(-0.5, 0),      # cc
            rng.uniform(0, 0.8),       # same_side
            rng.uniform(-0.5, 0),      # btw
        ])
        scores = weights @ np.column_stack(feature_arrays)
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Phase C: Local search with adaptive step size
    local_search_steps = 2000
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
            step_size = min(0.2, step_size * 1.08)
        elif i % 15 == 0:
            step_size = max(0.02, step_size * 0.88)

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
    time_limit = 125.0

    try:
        scores = _search_best_scores(adjacency, time_limit=time_limit)
    except Exception:
        scores = -adjacency.sum(axis=1).astype(float)

    return scores
# EVOLVE-BLOCK-END
