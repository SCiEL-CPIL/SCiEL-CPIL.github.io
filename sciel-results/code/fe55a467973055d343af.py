"""Combined program: feature-based local search + spectral partitioning + soft surrogate polish."""
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
    Returns tuple of 5 feature arrays.
    """
    n = adjacency.shape[0]
    if n < 2:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))

    A = adjacency.astype(float)
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return (np.zeros(n), np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n))

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

    # Determine partition sides using zero-crossing
    if np.all(fiedler_vec_centered >= 0):
        median_val = np.median(fiedler_vec_centered)
        side = np.where(fiedler_vec_centered >= median_val, 1.0, -1.0)
    elif np.all(fiedler_vec_centered <= 0):
        median_val = np.median(fiedler_vec_centered)
        side = np.where(fiedler_vec_centered <= median_val, -1.0, 1.0)
    else:
        side = np.where(fiedler_vec_centered >= 0, 1.0, -1.0)

    pos_mask = (side == 1.0)
    neg_mask = (side == -1.0)
    pos_count = np.sum(pos_mask)
    neg_count = np.sum(neg_mask)

    side_float = side.astype(float)
    neighbor_side_sum = A @ side_float
    cross_cut_degree = 0.5 * degree - 0.5 * side_float * neighbor_side_sum

    side_size = np.where(side == 1.0, pos_count, neg_count).astype(float)
    max_cross = np.max(cross_cut_degree) if np.max(cross_cut_degree) > 0 else 1.0
    cross_norm = cross_cut_degree / max_cross
    
    territory_score = side_size / n - 0.5 * cross_norm

    max_abs = np.max(np.abs(fiedler_vec_centered))
    if max_abs > 1e-12:
        dist_from_cut = np.abs(fiedler_vec_centered) / max_abs
    else:
        dist_from_cut = np.zeros(n)

    bottleneck_proximity = dist_from_cut

    return (fiedler_vec_centered, territory_score, dist_from_cut, cross_cut_degree, bottleneck_proximity)


def _soft_surrogate_score(adjacency, scores, temperature=0.1):
    """
    Smooth surrogate for the greedy decode objective.
    Uses a soft-min formulation to estimate how many vertices can be "selected"
    without conflict, providing a differentiable-ish signal for local refinement.
    """
    n = adjacency.shape[0]
    # For each vertex, compute the "competition" from higher-scoring neighbors.
    # A vertex is "blocked" if any higher-scoring neighbor is selected.
    # We approximate the selection probability as a sigmoid of relative score advantage.
    
    # Sort by score descending
    order = np.argsort(-scores, kind='stable')
    
    # For each vertex in order, its "effective score" is reduced by the presence
    # of already-selected higher-scoring neighbors. We use a soft version.
    
    # Compute for each vertex the sum of scores of its neighbors that are "before" it in order
    # This is complex to vectorize perfectly, so we use an approximation:
    # The key insight is that vertices with high scores relative to their neighbors are more likely to be picked.
    
    # Simpler approach: use the greedy decode count but with a perturbation-based gradient estimate
    # For the polish pass, we just do fine-grained random perturbations with acceptance criteria.
    
    # Actually, for the soft surrogate, let's use a different idea:
    # The independent set size is bounded by n / (1 + avg_degree_in_IS)
    # But more directly, we can use the fact that the greedy algorithm's performance
    # is sensitive to the ordering. We'll use a simulated annealing-like approach.
    
    # For now, this function serves as a placeholder for the smooth surrogate.
    # The actual refinement will use local search with small perturbations.
    return scores


def _search_best_scores(adjacency, time_limit=110.0):
    """
    Search for a good static scoring function.
    Combines local density metrics with global spectral partitioning insights.
    Includes a final high-precision polish pass.
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + time_limit

    # Precompute base features
    degree = _compute_degree(adjacency)
    nds = _compute_neighbor_degree_sum(adjacency)
    coreness = _compute_coreness(adjacency)
    effective_deg = _compute_effective_degree(adjacency)
    sparsity = _compute_neighborhood_sparsity(adjacency)
    mean_nd = _compute_mean_neighbor_degree(adjacency)
    
    # Spectral features
    fiedler_centered, territory, dist_cut, cross_cut, bottleneck = _compute_spectral_features(adjacency)

    # Normalize features to [0, 1] range for stable combination
    def normalize(x):
        x = np.asarray(x, dtype=float)
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    deg_norm = normalize(degree)
    nds_norm = normalize(nds)
    core_norm = normalize(coreness)
    eff_deg_norm = normalize(effective_deg)
    sparsity_norm = normalize(sparsity)
    mean_nd_norm = normalize(mean_nd)
    territory_norm = normalize(territory)
    dist_cut_norm = normalize(dist_cut)
    bottleneck_norm = normalize(bottleneck)

    # Feature matrix: each row is a vertex, each column is a feature
    # We want to find weights w such that score = w @ features
    # The features are: [-deg_norm, -nds_norm, core_norm, -eff_deg_norm, sparsity_norm, -mean_nd_norm,
    #                    territory_norm, dist_cut_norm, bottleneck_norm]
    
    features = np.column_stack([
        -deg_norm,
        -nds_norm,
        core_norm,
        -eff_deg_norm,
        sparsity_norm,
        -mean_nd_norm,
        territory_norm,
        dist_cut_norm,
        bottleneck_norm
    ])
    
    n_features = features.shape[1]

    best_scores = -degree  # Start with the seed strategy (lowest degree first)
    best_count = _greedy_decode(adjacency, best_scores)

    rng = np.random.default_rng(42)

    # Phase 1: Structured combinations
    structured_combos = [
        # Weights for each of the 9 features
        (-1, 0, 0, 0, 0, 0, 0, 0, 0),    # lowest degree
        (0, -1, 0, 0, 0, 0, 0, 0, 0),    # lowest neighbor degree sum
        (0, 0, 1, 0, 0, 0, 0, 0, 0),     # highest coreness
        (0, 0, 0, -1, 0, 0, 0, 0, 0),    # lowest effective degree
        (0, 0, 0, 0, 1, 0, 0, 0, 0),     # highest sparsity
        (0, 0, 0, 0, 0, -1, 0, 0, 0),    # lowest mean neighbor degree
        (0, 0, 0, 0, 0, 0, 1, 0, 0),     # highest territory
        (0, 0, 0, 0, 0, 0, 0, 1, 0),     # highest dist from cut
        (0, 0, 0, 0, 0, 0, 0, 0, 1),     # highest bottleneck proximity
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0),
        (-0.5, 0, 0.5, 0, 0, 0, 0, 0, 0),
        (0, -0.5, 0.5, 0, 0, 0, 0, 0, 0),
        (-0.3, -0.3, 0.4, 0, 0, 0, 0, 0, 0),
        (-0.2, -0.4, 0.4, 0, 0, 0, 0, 0, 0),
        (-0.4, -0.3, 0.3, 0, 0, 0, 0, 0, 0),
        (-1, -1, 0, 0, 0, 0, 0, 0, 0),
        (-1, 0, 1, 0, 0, 0, 0, 0, 0),
        (0, -1, 1, 0, 0, 0, 0, 0, 0),
        (-0.5, -0.5, 0.5, 0, 0, 0, 0, 0, 0),
        (-0.2, -0.2, 0.6, 0, 0, 0, 0, 0, 0),
        (-0.6, -0.2, 0.2, 0, 0, 0, 0, 0, 0),
        (-0.2, -0.6, 0.2, 0, 0, 0, 0, 0, 0),
        # Add spectral features
        (-0.5, -0.5, 0, 0, 0, 0, 0.5, 0, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0.5, 0),
        (-0.5, -0.5, 0, 0, 0, 0, 0, 0, 0.5),
        (-0.3, -0.3, 0.3, 0, 0, 0, 0.3, 0.3, 0.3),
        (-0.5, -0.5, 0.5, 0, 0, 0, 0.5, 0.5, 0.5),
        (-0.2, -0.2, 0.2, 0, 0.2, 0, 0.2, 0.2, 0.2),
    ]

    for weights in structured_combos:
        if time.time() > deadline - 10:
            break
        w = np.array(weights, dtype=float)
        scores = features @ w
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Phase 2: Random linear combinations
    num_random_tries = 3000
    for i in range(num_random_tries):
        if time.time() > deadline - 15:
            break
        # Random weights with appropriate signs
        w1 = rng.uniform(-1, 0)   # degree: negative means prefer low degree
        w2 = rng.uniform(-1, 0)   # nds: negative means prefer low nds
        w3 = rng.uniform(0, 1)    # core: positive means prefer high core
        w4 = rng.uniform(-1, 0)   # eff_deg: negative means prefer low eff_deg
        w5 = rng.uniform(0, 1)    # sparsity: positive means prefer high sparsity
        w6 = rng.uniform(-1, 0)   # mean_nd: negative means prefer low mean_nd
        w7 = rng.uniform(0, 1)    # territory: positive
        w8 = rng.uniform(0, 1)    # dist_cut: positive
        w9 = rng.uniform(0, 1)    # bottleneck: positive
        
        w = np.array([w1, w2, w3, w4, w5, w6, w7, w8, w9], dtype=float)
        scores = features @ w
        count = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()

    # Phase 3: Local search with small random perturbations
    local_search_steps = 3000
    current_scores = best_scores.copy()
    current_count = best_count

    for i in range(local_search_steps):
        if time.time() > deadline - 5:
            break
        # Add small random perturbation
        noise = rng.normal(0, 0.05, size=n)
        perturbed = current_scores + noise
        count = _greedy_decode(adjacency, perturbed)
        if count > current_count:
            current_count = count
            current_scores = perturbed.copy()
            if count > best_count:
                best_count = count
                best_scores = perturbed.copy()
        # Occasionally try a larger perturbation
        elif i % 10 == 0:
            noise = rng.normal(0, 0.2, size=n)
            perturbed = current_scores + noise
            count = _greedy_decode(adjacency, perturbed)
            if count > current_count:
                current_count = count
                current_scores = perturbed.copy()
                if count > best_count:
                    best_count = count
                    best_scores = perturbed.copy()

    # Phase 4: High-precision polish pass
    # Use very small perturbations to fine-tune the scores
    polish_steps = 2000
    for i in range(polish_steps):
        if time.time() > deadline - 1:
            break
        # Very small perturbations
        noise = rng.normal(0, 0.01, size=n)
        perturbed = best_scores + noise
        count = _greedy_decode(adjacency, perturbed)
        if count > best_count:
            best_count = count
            best_scores = perturbed.copy()

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
    # Leave a safety margin before the 120s limit
    time_limit = 110.0

    try:
        scores = _search_best_scores(adjacency, time_limit=time_limit)
    except Exception:
        # Fallback to simple degree-based scoring
        scores = -adjacency.sum(axis=1).astype(float)

    return scores
# EVOLVE-BLOCK-END
