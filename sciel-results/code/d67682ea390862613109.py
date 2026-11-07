import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    Uses a local RNG for reproducibility.
    """
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def _compute_mst_edges(distance_matrix):
    """
    Compute MST edges from the distance matrix using scipy.
    Returns a boolean matrix where True indicates an MST edge.
    """
    n = distance_matrix.shape[0]
    # Create a sparse graph representation
    from scipy.sparse import csr_matrix
    # Use the distance matrix as weights
    # We need to handle the diagonal (self-loops) - set to inf or large value
    dm = distance_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    
    # Convert to sparse
    sparse_dm = csr_matrix(dm)
    
    # Compute MST
    try:
        mst = minimum_spanning_tree(sparse_dm)
        mst_coo = mst.tocoo()
        mst_edges = np.zeros((n, n), dtype=bool)
        for i, j in zip(mst_coo.row, mst_coo.col):
            mst_edges[i, j] = True
            mst_edges[j, i] = True
        return mst_edges
    except Exception:
        # Fallback: return empty matrix
        return np.zeros((n, n), dtype=bool)

def _compute_knn_edges(distance_matrix, k=5):
    """
    Compute k-nearest neighbor edges.
    Returns a boolean matrix where True indicates a kNN edge.
    """
    n = distance_matrix.shape[0]
    knn_edges = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        dists = distance_matrix[i, :].copy()
        dists[i] = np.inf
        # Get k nearest neighbors
        if k < n:
            nearest = np.argpartition(dists, k)[:k]
        else:
            nearest = np.argsort(dists)
        
        for j in nearest:
            knn_edges[i, j] = True
    
    return knn_edges

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Compute MST edges
    mst_edges = _compute_mst_edges(distance_matrix)
    
    # Compute kNN edges
    knn_edges = _compute_knn_edges(distance_matrix, k=5)
    
    # Strategy: Combine inverse distance with structural priors
    # For MST edges and kNN edges, boost the heuristic significantly
    # This forces the ACO to prioritize the structural backbone
    
    # Try different combinations of structural priors
    best_score = float('inf')
    best_heuristic = base_heur.copy()
    
    # Candidate 1: Pure inverse distance
    candidates = []
    
    # Add pure inverse distance as baseline
    candidates.append(('base', base_heur.copy()))
    
    # Add MST-boosted versions
    for boost in [2.0, 5.0, 10.0, 20.0]:
        h = base_heur.copy()
        h[mst_edges] *= boost
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        candidates.append((f'mst_{boost}', h))
    
    # Add kNN-boosted versions
    for boost in [2.0, 5.0, 10.0]:
        h = base_heur.copy()
        h[knn_edges] *= boost
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        candidates.append((f'knn_{boost}', h))
    
    # Add combined MST+kNN boosted versions
    for mst_boost in [5.0, 10.0]:
        for knn_boost in [2.0, 5.0]:
            h = base_heur.copy()
            h[mst_edges] *= mst_boost
            h[knn_edges] *= knn_boost
            h[h <= 0] = 1e-9
            h[~np.isfinite(h)] = 1e-9
            candidates.append((f'mst{knn}_{mst_boost}_{knn_boost}', h))
    
    # Evaluate each candidate
    for name, h in candidates:
        if time.time() > deadline - 30.0:
            break
        
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=42)
        
        if score < best_score:
            best_score = score
            best_heuristic = h.copy()
    
    # If time permits, also try a few power-law exponents on the base heuristic
    if time.time() < deadline - 40.0:
        for beta in [1.5, 2.0, 2.5, 3.0]:
            if time.time() > deadline - 30.0:
                break
            h = base_heur ** beta
            h[h <= 0] = 1e-9
            h[~np.isfinite(h)] = 1e-9
            score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=42)
            
            if score < best_score:
                best_score = score
                best_heuristic = h.copy()
    
    # Ensure final heuristic is valid
    final_heuristic = best_heuristic.copy()
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
