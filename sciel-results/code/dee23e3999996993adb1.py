import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

def _simulate_aco_fast(d, eta, n_ants=20, n_iterations=30, seed=None):
    """
    Fast simulation of the ACO algorithm to estimate heuristic quality.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = d.shape[0]
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for _ in range(n_iterations):
        all_tours = []
        all_lens = []
        
        for _ in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            current = 0
            visited[0] = True
            tour_len = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                probs[current] = 0.0
                
                total = np.sum(probs)
                if total <= 1e-15:
                    unvisited = np.where(~visited)[0]
                    next_node = unvisited[np.random.randint(len(unvisited))]
                else:
                    cum_probs = np.cumsum(probs)
                    cum_probs /= total
                    r = np.random.rand()
                    next_node = np.searchsorted(cum_probs, r)
                    if next_node >= n:
                        next_node = n - 1
                
                tour_len += d[current, next_node]
                tour.append(next_node)
                current = next_node
                visited[current] = True
            
            tour_len += d[current, 0]
            all_tours.append(tour)
            all_lens.append(tour_len)
            
            if tour_len < best_len:
                best_len = tour_len
        
        tau *= 0.9
        for i, tour in enumerate(all_tours):
            L = all_lens[i]
            if L < 1e-15:
                continue
            delta = 1.0 / L
            for k in range(len(tour)):
                u = tour[k]
                v = tour[(k + 1) % len(tour)]
                tau[u, v] += delta
                
    return best_len

def _compute_mst_edges(d):
    """Compute MST edge set from distance matrix."""
    n = d.shape[0]
    # Create a sparse graph from the distance matrix
    from scipy.sparse import csr_matrix
    d_sparse = csr_matrix(d)
    mst = minimum_spanning_tree(d_sparse)
    # Get the non-zero entries (MST edges)
    row, col = mst.nonzero()
    edges = set()
    for i in range(len(row)):
        u, v = row[i], col[i]
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return edges

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    mask = np.eye(n, dtype=bool)
    
    start_time = time.time()
    time_limit = 280.0  # Safety margin for 300s limit
    
    # Compute MST edges
    mst_edges = _compute_mst_edges(d)
    
    # Strategy 1: Pure MST-based heuristic
    # High value for MST edges, low for others
    eta_mst = np.full((n, n), 1e-9)
    for (u, v) in mst_edges:
        eta_mst[u, v] = 10.0
        eta_mst[v, u] = 10.0
    eta_mst[mask] = 1e-9
    
    # Strategy 2: Hybrid - MST edges get boost, others use inverse distance
    d_safe = d.copy()
    d_safe[mask] = 1.0
    base_eta = 1.0 / (d_safe ** 1.5)
    base_eta[mask] = 1e-9
    
    eta_hybrid = base_eta.copy()
    for (u, v) in mst_edges:
        eta_hybrid[u, v] *= 5.0
        eta_hybrid[v, u] *= 5.0
    eta_hybrid[mask] = 1e-9
    
    # Strategy 3-5: Pure inverse distance with different exponents
    candidates = []
    candidates.append(('mst_pure', eta_mst))
    candidates.append(('hybrid_mst', eta_hybrid))
    
    for p in [1.0, 1.5, 2.0, 2.5, 3.0]:
        d_safe = d.copy()
        d_safe[mask] = 1.0
        eta = 1.0 / (d_safe ** p)
        eta[mask] = 1e-9
        candidates.append((f'inv_d_p{p}', eta))
    
    # Test each candidate with a quick simulation
    best_eta = None
    best_score = float('inf')
    
    # First, measure time for one simulation
    t0 = time.time()
    _simulate_aco_fast(d, candidates[0][1], seed=42)
    t1 = time.time()
    sim_time = t1 - t0
    
    # Determine how many seeds we can afford
    remaining_time = time_limit - (time.time() - start_time)
    num_seeds = max(1, min(5, int(remaining_time / (sim_time * len(candidates) * 1.5))))
    
    for name, eta in candidates:
        if time.time() - start_time > time_limit:
            break
        
        scores = []
        for s in range(num_seeds):
            seed_val = s * 7919 + 13
            score = _simulate_aco_fast(d, eta, seed=seed_val)
            scores.append(score)
        
        avg_score = np.mean(scores)
        
        if avg_score < best_score:
            best_score = avg_score
            best_eta = eta.copy()
    
    if best_eta is None:
        # Default fallback
        d_safe = d.copy()
        d_safe[mask] = 1.0
        best_eta = 1.0 / (d_safe ** 1.5)
        best_eta[mask] = 1e-9
    
    # Ensure all values are positive
    best_eta[~np.isfinite(best_eta)] = 1e-9
    best_eta[best_eta <= 0] = 1e-9
    
    return best_eta
