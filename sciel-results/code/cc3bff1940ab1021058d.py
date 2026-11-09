# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_decode(adjacency, scores):
    """Run the fixed greedy decoder: pick vertices in descending score order,
    skipping neighbours of already-picked vertices."""
    n = adjacency.shape[0]
    order = np.argsort(-scores, kind='stable')
    selected = np.zeros(n, dtype=bool)
    picked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if selected[v]:
            continue
        picked[v] = True
        count += 1
        # Mark all neighbors as selected (blocked)
        neighbors = np.where(adjacency[v])[0]
        selected[neighbors] = True
    return count, picked

def _compute_coreness(adjacency):
    """Compute k-core decomposition coreness for each vertex."""
    n = adjacency.shape[0]
    adj = adjacency.copy()
    degrees = adj.sum(axis=1).astype(int).copy()
    coreness = np.zeros(n, dtype=int)
    
    # Simple k-core decomposition
    in_core = np.ones(n, dtype=bool)
    current_k = 0
    
    while in_core.any():
        # Find vertices with minimum degree among remaining
        remaining = np.where(in_core)[0]
        if len(remaining) == 0:
            break
        min_deg = degrees[remaining].min()
        if min_deg <= current_k:
            current_k = min_deg
            # Assign coreness to all remaining vertices with degree == min_deg
            to_remove = remaining[degrees[remaining] == min_deg]
            for v in to_remove:
                coreness[v] = max(coreness[v], min_deg)
                in_core[v] = False
                for u in np.where(adj[v])[0]:
                    if in_core[u]:
                        degrees[u] -= 1
        else:
            # All remaining have degree > current_k, set their coreness
            coreness[remaining] = np.maximum(coreness[remaining], current_k + 1)
            break
    
    return coreness

def _search_best_scores(adjacency):
    """Search over parameterized scoring functions to find the best greedy outcome."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees  # sum of degrees of neighbors
    coreness = _compute_coreness(adjacency).astype(float)
    
    # Normalize for better conditioning
    deg_norm = degrees / (degrees.max() + 1e-9)
    nds_norm = neighbor_deg_sum / (neighbor_deg_sum.max() + 1e-9)
    core_norm = coreness / (coreness.max() + 1e-9)
    
    best_count = -1
    best_scores = None
    
    # Grid search over alpha, beta, gamma
    alphas = [0, 0.5, 1, 2, 3, 5, 10]
    betas = [0, 0.1, 0.5, 1, 2, 5]
    gammas = [0, 0.1, 0.5, 1]
    
    # Start with the seed: lowest degree
    base_scores = -degrees
    count, _ = _greedy_decode(adjacency, base_scores)
    best_count = count
    best_scores = base_scores.copy()
    
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                if time.time() > deadline:
                    return best_scores
                
                scores = -(alpha * deg_norm + beta * nds_norm + gamma * core_norm)
                count, _ = _greedy_decode(adjacency, scores)
                if count > best_count:
                    best_count = count
                    best_scores = scores.copy()
    
    # Try some randomized perturbations around the best found
    for _ in range(200):
        if time.time() > deadline:
            return best_scores
        
        # Small random perturbation
        pert = np.random.uniform(-0.1, 0.1, n)
        scores = best_scores + pert
        count, _ = _greedy_decode(adjacency, scores)
        if count > best_count:
            best_count = count
            best_scores = scores.copy()
    
    return best_scores

def priority(adjacency):
    """Give every vertex a score; greedy decoder picks in descending order."""
    adjacency = np.asarray(adjacency)
    
    # Try the search-based approach
    try:
        scores = _search_best_scores(adjacency)
        if scores is not None and np.all(np.isfinite(scores)):
            return scores
    except Exception:
        pass
    
    # Fallback: use the proven improvement with neighbor degree
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    # Combine: low degree is good, but also consider blocking high-degree neighbors
    # score = -(degree + 0.3 * mean_neighbor_degree)
    mean_neighbor_deg = neighbor_deg_sum / (degrees + 1e-9)
    scores = -(degrees + 0.3 * mean_neighbor_deg)
    
    return scores
# EVOLVE-BLOCK-END
