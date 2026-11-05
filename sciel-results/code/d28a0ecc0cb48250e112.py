import numpy as np
import time
import random

def _greedy_score(adj, scores):
    """Run the fixed greedy decoder and return the independent set size."""
    n = adj.shape[0]
    order = np.argsort(-scores, kind='stable')
    chosen = set()
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        v = int(v)
        if not blocked[v]:
            chosen.add(v)
            count += 1
            # block v and its neighbors
            blocked[v] = True
            nbrs = np.nonzero(adj[v])[0]
            if len(nbrs) > 0:
                blocked[nbrs] = True
    return count

def _local_search(adj, scores, max_iter=500, time_limit=None):
    """Improve the score vector via pairwise swaps that increase greedy set size."""
    n = adj.shape[0]
    best = _greedy_score(adj, scores)
    cur = scores.copy()
    start = time.time()
    
    # Precompute neighbor lists for fast blocking
    # adj is bool matrix, n x n
    # For each vertex, store its neighbors as a list/array
    # Actually, for speed, we'll use the adjacency matrix directly in greedy_score
    
    it = 0
    no_improve = 0
    while it < max_iter:
        if time_limit is not None and time.time() - start > time_limit:
            break
        
        # Random swap of two vertices' scores
        i, j = random.sample(range(n), 2)
        cur[i], cur[j] = cur[j], cur[i]
        
        new_score = _greedy_score(adj, cur)
        if new_score > best:
            best = new_score
            no_improve = 0
        else:
            # Revert
            cur[i], cur[j] = cur[j], cur[i]
            no_improve += 1
            if no_improve > 200:
                break
        it += 1
    
    return cur, best

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool).astype(float)
    n = adj.shape[0]
    
    # Initial heuristic: combination of degree and eigenvector centrality
    # Compute eigenvector centrality via power iteration
    x = np.ones(n)
    x /= np.linalg.norm(x)
    for _ in range(50):
        new_x = adj @ x
        norm = np.linalg.norm(new_x)
        if norm < 1e-10:
            break
        new_x /= norm
        if np.linalg.norm(new_x - x) < 1e-8:
            x = new_x
            break
        x = new_x
    
    centrality = x
    deg = adj.sum(axis=1)
    
    # Initial score: low degree + low centrality is preferred
    # But also consider neighbor structure
    neighbor_cent_sum = adj @ centrality
    safe_deg = deg + 1.0
    avg_neighbor_cent = neighbor_cent_sum / safe_deg
    
    # Score: want to pick peripheral, low-degree nodes
    score_init = -0.5 * centrality - 0.5 * (deg / (n / 2)) + 0.3 * avg_neighbor_cent
    
    # Normalize scores to [0, 1] for stability
    score_min = score_init.min()
    score_max = score_init.max()
    if score_max - score_min > 1e-10:
        score_init = (score_init - score_min) / (score_max - score_min)
    else:
        score_init = np.zeros(n)
    
    # Time budget: use ~100 seconds for local search
    time_for_ls = 95.0
    elapsed = time.time() - start_time
    if elapsed > time_for_ls:
        return score_init
    
    # Multiple restarts with local search
    best_score = _greedy_score(adj, score_init)
    best_scores = score_init.copy()
    
    num_restarts = 3
    time_per_restart = (time_for_ls - elapsed) / num_restarts
    
    for r in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_for_ls - 5:  # leave safety margin
            break
        
        # Generate a perturbed initial score
        if r == 0:
            cur_scores = score_init.copy()
        else:
            # Random perturbation
            noise = np.random.normal(0, 0.5, n)
            cur_scores = score_init + noise
        
        # Local search
        cur_scores, cur_best = _local_search(adj, cur_scores, max_iter=300, time_limit=time_per_restart)
        
        if cur_best > best_score:
            best_score = cur_best
            best_scores = cur_scores.copy()
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > 110:
        return best_scores
    
    return best_scores
