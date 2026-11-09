"""Optimized TSP heuristic: Cached MDS + Vectorized Directional Nearest Neighbor."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s hard kill

# Cache for coordinates derived from distance matrix (Euclidean assumption)
# Key: id(distance_matrix) -> (coords, success_flag)
_COORD_CACHE = {}


def _get_coordinates(distance_matrix):
    """
    Recover 2D coordinates from a Euclidean distance matrix using MDS.
    Returns (coords, success).
    """
    key = id(distance_matrix)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    
    n = distance_matrix.shape[0]
    try:
        # Use classical MDS to recover coordinates
        D2 = distance_matrix ** 2
        # Centering matrix
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take top 2 eigenvalues
        idx = np.argsort(eigenvalues)[::-1][:2]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Coordinates
        coords = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        _COORD_CACHE[key] = (coords, True)
        return coords, True
    except Exception:
        _COORD_CACHE[key] = (None, False)
        return None, False


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a fast, vectorized heuristic.
    1. If coordinates are available, use a directional nearest-neighbor score.
    2. Fallback to 2-step lookahead if geometry is unavailable.
    """
    # Time guard: if we are close to the limit, just pick the nearest neighbor
    if time.time() - _START_TIME > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distance from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Get distance from all unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Simple 2-step lookahead score: d(current, next) + d(next, dest)
    # This is a strong baseline. We will try to improve it with geometry.
    lookahead_scores = d_curr + d_dest
    
    # Try to use geometric information
    coords, success = _get_coordinates(distance_matrix)
    
    if success:
        curr_pos = coords[current_node]
        dest_pos = coords[destination_node]
        unvisited_pos = coords[unvisited_nodes]
        
        # Vector from current to destination
        dir_vec = dest_pos - curr_pos
        dir_norm_sq = np.dot(dir_vec, dir_vec)
        
        if dir_norm_sq > 1e-10:
            # Compute dot product of (unvisited - current) with dir_vec
            # This measures how "forward" each candidate is relative to the destination direction
            # Higher dot product means more aligned with destination direction
            vectors = unvisited_pos - curr_pos
            dots = vectors @ dir_vec
            
            # Normalize dots to be comparable to distances?
            # Distances are in units of length. Dots are in units of length^2.
            # We need a weight. Let's normalize dots by the norm of dir_vec.
            dir_norm = np.sqrt(dir_norm_sq)
            proj_lengths = dots / dir_norm
            
            # We want to prefer nodes that are close (low d_curr) AND forward (high proj_lengths).
            # Let's normalize both to [0, 1] range roughly.
            
            # Normalize d_curr
            d_min = np.min(d_curr)
            d_max = np.max(d_curr)
            if d_max > d_min:
                d_norm = (d_curr - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d_curr)
                
            # Normalize proj_lengths
            p_min = np.min(proj_lengths)
            p_max = np.max(proj_lengths)
            if p_max > p_min:
                p_norm = (proj_lengths - p_min) / (p_max - p_min)
            else:
                p_norm = np.zeros_like(proj_lengths)
                
            # Score: low distance is good (0), high forward projection is good (1).
            # So we want to minimize distance and maximize projection.
            # Score = w1 * d_norm - w2 * p_norm
            # Let's use weights that balance them.
            # Usually distance is the dominant factor.
            w_dist = 1.0
            w_dir = 0.3
            
            geo_scores = w_dist * d_norm - w_dir * p_norm
            
            # Combine with lookahead?
            # The lookahead score already accounts for the return trip.
            # The geometric score adds a "smoothness" penalty for turning back.
            # Let's blend them.
            # Normalize lookahead scores to [0,1]
            l_min = np.min(lookahead_scores)
            l_max = np.max(lookahead_scores)
            if l_max > l_min:
                l_norm = (lookahead_scores - l_min) / (l_max - l_min)
            else:
                l_norm = np.zeros_like(lookahead_scores)
                
            # Final score: primarily lookahead, secondarily geometric smoothness
            final_scores = l_norm + 0.2 * geo_scores
            
            best_idx = np.argmin(final_scores)
            return int(unvisited_nodes[best_idx])
            
    # Fallback: Pure 2-step lookahead
    best_idx = np.argmin(lookahead_scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
