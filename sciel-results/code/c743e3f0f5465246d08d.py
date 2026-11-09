"""
Maximum Independent Set via priority scoring.
Strategy:
1. Fast static scoring: combine degree, neighbor-degree sum, coreness, and spectral gap features.
2. Multi-start local search on the vertex ordering (swap-based) to refine the greedy outcome.
3. Optional local search on the independent set itself for final polishing.
"""
import numpy as np
import time

def _greedy_decode(adjacency, order):
    """
    Run the fixed greedy decoder given a specific vertex order (permutation).
    Returns: (count, selected_mask)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        available[adjacency[v]] = False
        
    return count, selected

def _evaluate_scores(adjacency, scores):
    """
    Evaluate a set of scores by sorting them and running the greedy decoder.
    Returns: count
    """
    order = np.argsort(-scores, kind='stable')
    count, _ = _greedy_decode(adjacency, order)
    return count

def _compute_core_numbers(adjacency):
    """
    Compute the core number for each vertex in the graph.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    
    deg = degrees.copy()
    core = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if not np.any(remaining):
            break
        masked_degrees = np.where(remaining, deg, np.inf)
        v = np.argmin(masked_degrees)
        core[v] = deg[v]
        remaining[v] = False
        neighbors = np.where(adjacency[v])[0]
        active_neighbors = neighbors[remaining[neighbors]]
        deg[active_neighbors] -= 1
        
    return core

def _compute_spectral_features(adjacency):
    """
    Compute spectral features: use the Fiedler vector (eigenvector of the second smallest
    eigenvalue of the Laplacian) to identify a spectral cut. Vertices on one side of the cut
    are prioritized to minimize crossing edges.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    # Laplacian L = D - A
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        return fiedler
    except np.linalg.LinAlgError:
        # Fallback: return zeros
        return np.zeros(n)

def _baseline_scores(adjacency):
    """
    Compute a combined baseline score using degree, neighbor-degree sum, coreness, and spectral features.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    coreness = _compute_core_numbers(adjacency).astype(float)
    spectral = _compute_spectral_features(adjacency)
    
    # Normalize each feature
    deg_norm = degrees / (degrees.max() + 1e-9)
    nds_norm = neighbor_deg_sum / (neighbor_deg_sum.max() + 1e-9)
    core_norm = coreness / (coreness.max() + 1e-9)
    spec_norm = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-9)
    
    # We want low degree, low neighbor-degree sum, low coreness, and low spectral value (one side of cut)
    # to have HIGH priority. So we negate them.
    # Weighted combination:
    alpha, beta, gamma, delta = 1.0, 0.5, 0.5, 0.3
    scores = -(alpha * deg_norm + beta * nds_norm + gamma * core_norm + delta * spec_norm)
    
    return scores

def _local_search_order(adjacency, initial_order, deadline):
    """
    Multi-start local search on the vertex ordering. We try swaps and 2-opt moves to improve
    the greedy outcome.
    """
    n = adjacency.shape[0]
    current_order = initial_order.copy()
    current_count, _ = _greedy_decode(adjacency, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # Try random swaps
        for _ in range(100):
            if time.time() > deadline:
                break
            i, j = np.random.choice(n, 2, replace=False)
            current_order[i], current_order[j] = current_order[j], current_order[i]
            new_count, _ = _greedy_decode(adjacency, current_order)
            if new_count > current_count:
                current_count = new_count
                improved = True
                if new_count > best_count:
                    best_count = new_count
                    best_order = current_order.copy()
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
    return best_order, best_count

def _local_search_set(adjacency, initial_set, deadline):
    """
    Local search on the independent set itself. Try to add vertices that don't conflict,
    and try swaps (remove one, add two) if possible.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = np.sum(current_set)
    
    best_set = current_set.copy()
    best_size = current_size
    
    def get_blocked_mask(s):
        if not np.any(s):
            return np.zeros(n, dtype=bool)
        neighbor_counts = adjacency @ s
        return s | (neighbor_counts > 0)
    
    blocked = get_blocked_mask(current_set)
    
    while time.time() < deadline:
        candidates = np.where(~blocked)[0]
        
        if len(candidates) == 0:
            # Try to remove a vertex to unblock others
            set_vertices = np.where(current_set)[0]
            if len(set_vertices) == 0:
                break
            v_to_remove = set_vertices[np.random.randint(len(set_vertices))]
            current_set[v_to_remove] = False
            current_size -= 1
            blocked = get_blocked_mask(current_set)
            continue
        
        # Try to add a random candidate
        v = candidates[np.random.randint(len(candidates))]
        current_set[v] = True
        current_size += 1
        blocked = get_blocked_mask(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set.copy()
        
        # Occasionally try a swap: remove a vertex, see if we can add two
        if np.random.random() < 0.1:
            set_vertices = np.where(current_set)[0]
            if len(set_vertices) > 1:
                v_remove = set_vertices[np.random.randint(len(set_vertices))]
                current_set[v_remove] = False
                current_size -= 1
                blocked = get_blocked_mask(current_set)
                candidates_after = np.where(~blocked)[0]
                if len(candidates_after) >= 2:
                    # Add two random candidates
                    v1, v2 = np.random.choice(candidates_after, 2, replace=False)
                    current_set[v1] = True
                    current_set[v2] = True
                    current_size += 2
                    blocked = get_blocked_mask(current_set)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
    
    return best_set, best_size

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # 1. Baseline scores
    best_scores = _baseline_scores(adjacency)
    best_count = _evaluate_scores(adjacency, best_scores)
    
    # 2. Local search on the ordering
    if time.time() < deadline - 20.0:
        initial_order = np.argsort(-best_scores, kind='stable')
        ls_order, ls_count = _local_search_order(adjacency, initial_order, deadline - 10.0)
        if ls_count > best_count:
            best_count = ls_count
            # Convert order to scores
            best_scores = np.zeros(n)
            for idx, v in enumerate(ls_order):
                best_scores[v] = n - idx
    
    # 3. Local search on the independent set
    if time.time() < deadline - 5.0:
        order = np.argsort(-best_scores, kind='stable')
        _, init_set = _greedy_decode(adjacency, order)
        ls_set, ls_count = _local_search_set(adjacency, init_set, deadline - 1.0)
        
        if ls_count > best_count:
            # Convert set to scores: vertices in the set get high scores
            best_scores = np.zeros(n)
            best_scores[ls_set] = 1.0
            best_count = ls_count
    
    return best_scores
