# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Safety margin

def _spectral_gap_penalty(D, coords):
    """
    Compute Fiedler vector of the graph Laplacian derived from inverse-distance weights.
    Return a penalty matrix where entries are < 1 for edges connecting nodes with large
    differences in their Fiedler coordinates.
    """
    n = D.shape[0]
    if n < 4:
        return None
    
    # Build weighted adjacency matrix using inverse distance
    # W[i,j] = 1/D[i,j] for i != j, 0 otherwise
    W = np.zeros((n, n))
    mask = D > 1e-9
    W[mask] = 1.0 / D[mask]
    np.fill_diagonal(W, 0.0)
    
    # Compute degree matrix
    degrees = W.sum(axis=1)
    # Avoid division by zero
    degrees[degrees == 0] = 1e-9
    
    # Compute unnormalized Laplacian: L = D - W
    L = np.diag(degrees) - W
    
    # Symmetrize to handle numerical errors
    L = (L + L.T) / 2.0
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        return None
    
    # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
    # Sort eigenvalues in ascending order
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    if n < 2:
        return None
    
    fiedler_vec = eigvecs[:, 1]
    
    # Compute pairwise differences in Fiedler coordinates
    diff = np.abs(fiedler_vec[:, np.newaxis] - fiedler_vec[np.newaxis, :])
    
    # Normalize differences to [0, 1]
    max_diff = diff.max()
    if max_diff < 1e-9:
        return None
    
    diff_norm = diff / max_diff
    
    # Create penalty: edges with large Fiedler differences get penalized
    # Use a smoother penalty function, e.g., exponential decay
    # penalty = exp(-alpha * diff_norm)
    alpha = 1.5
    penalty = np.exp(-alpha * diff_norm)
    
    # Zero out diagonal
    np.fill_diagonal(penalty, 1.0)
    
    return penalty

def _mds_coordinates(D):
    """
    Recover 2D coordinates from a Euclidean distance matrix using Classical MDS.
    """
    n = D.shape[0]
    if n < 3:
        return None
    
    if not np.allclose(D, D.T, atol=1e-8):
        return None
    if np.any(D < -1e-9):
        return None

    J = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * (J @ D2 @ J)
    B = (B + B.T) / 2.0
    
    try:
        w, V = np.linalg.eigh(B)
    except:
        return None
        
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    
    if w[0] < 1e-9:
        return None
        
    coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
    
    # Verify reconstruction quality
    rec_D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
    corr = np.corrcoef(rec_D.flatten(), D.flatten())[0, 1]
    if np.isnan(corr) or corr < 0.99:
        return None
        
    return coords

def _tour_length(tour, D):
    n = len(tour)
    return sum(D[tour[i], tour[(i+1) % n]] for i in range(n))

def _fast_two_opt(tour, D):
    """
    Optimized 2-opt improvement.
    """
    n = len(tour)
    if n < 4:
        return list(tour)
    
    best_tour = list(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = best_tour[i], best_tour[i+1]
                c, d = best_tour[j], best_tour[(j+1) % n]
                
                old_len = D[a, b] + D[c, d]
                new_len = D[a, c] + D[b, d]
                
                if new_len < old_len - 1e-12:
                    best_tour[i+1:j+1] = best_tour[i+1:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best_tour

def _improve_tour(tour, D):
    """
    Combined 2-opt improvement.
    """
    current_tour = list(tour)
    # Just run 2-opt until no improvement. It's fast for N=50.
    for _ in range(10): # Max 10 passes of full 2-opt just in case
        prev_len = _tour_length(current_tour, D)
        current_tour = _fast_two_opt(current_tour, D)
        curr_len = _tour_length(current_tour, D)
        if curr_len >= prev_len - 1e-9:
            break
    return current_tour

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 4
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        heuristic = 1.0 / (d + 1e-10) ** 3
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Base heuristic: inverse distance
    # Using power 3 strengthens bias towards local edges, as found in observations
    heuristic = 1.0 / (d + 1e-10) ** 3
    
    # 2. Find a high-quality reference tour
    # Use multiple greedy starts with 2-opt improvement
    
    best_tour = None
    best_len = np.inf
    
    # Try multiple starting nodes for greedy construction
    num_starts = 5 # Reduced to save time, 2-opt is fast now
    if n > num_starts:
        starts = list(np.linspace(0, n-1, num_starts, dtype=int))
    else:
        starts = list(range(n))
    
    for start in starts:
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Nearest Neighbor tour from start
        visited = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        while unvisited:
            min_dist = np.inf
            next_node = -1
            for node in unvisited:
                if d[current, node] < min_dist:
                    min_dist = d[current, node]
                    next_node = node
            visited.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        improved_tour = _improve_tour(visited, d)
        tour_len = _tour_length(improved_tour, d)
        if tour_len < best_len:
            best_len = tour_len
            best_tour = improved_tour
            
    # 3. Boost edges from the best tour
    # Use a softer boost to avoid over-constraining the ACO
    tour_boost = 5.0
    if best_tour:
        for i in range(n):
            u = best_tour[i]
            v = best_tour[(i + 1) % n]
            heuristic[u, v] *= tour_boost
            heuristic[v, u] *= tour_boost

    # 4. Geometric boosts if MDS succeeded
    coords = _mds_coordinates(d)
    if coords is not None and time.time() < _DEADLINE - 5.0:
        try:
            # Spectral gap penalty using Fiedler vector
            spectral_penalty = _spectral_gap_penalty(d, coords)
            if spectral_penalty is not None:
                heuristic *= spectral_penalty
            
            # Convex Hull boost
            # Hull edges are almost certainly in the optimal tour.
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 2.0 # Softer boost
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
                
        except Exception:
            pass

    # 5. K-nearest neighbors boost
    # Short edges are generally good.
    k = 4
    knn_boost = 1.5 # Softer boost
    for i in range(n):
        dists = d[i].copy()
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        for j in nn_indices:
            heuristic[i, j] *= knn_boost
            heuristic[j, i] *= knn_boost

    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
