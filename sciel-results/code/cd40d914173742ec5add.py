import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    # Initialize with true distances
    d_out = edge_distance.copy()
    
    # Find max distance to scale penalties
    max_d = np.max(d_out)
    if max_d == 0:
        return d_out

    # 1. Build tour mask
    tour_mask = np.zeros((n, n), dtype=float)
    try:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            if 0 <= u < n and 0 <= v < n:
                tour_mask[u, v] = 1.0
                tour_mask[v, u] = 1.0
    except Exception:
        return edge_distance.copy()

    # Geometry-aware inverse distance weighting
    # Use squared inverse distance (1/d^2) for strong repulsive force on short edges
    safe_dist = np.where(edge_distance > 1e-9, edge_distance, 1.0)
    inv_d_sq = 1.0 / (safe_dist * safe_dist)
    
    # Normalize by mean to focus penalty on short edges relative to the average
    # This avoids the penalty being dominated by a single extremely short edge
    inv_d_sq_diag_masked = inv_d_sq.copy()
    np.fill_diagonal(inv_d_sq_diag_masked, 0.0)
    
    mean_inv_d_sq = np.mean(inv_d_sq_diag_masked)
    if mean_inv_d_sq > 0:
        inv_d_sq_norm = inv_d_sq / mean_inv_d_sq
    else:
        inv_d_sq_norm = np.ones_like(edge_distance)
        
    # Zero out diagonal for penalty calculation
    np.fill_diagonal(inv_d_sq_norm, 0.0)

    # 2. Cumulative penalty
    # Use x / (1 + x) for a faster-growing penalty for frequently used edges
    # This helps escape local optima that persist over many iterations
    cum_sat = edge_n_used / (1.0 + edge_n_used)
    
    # Adjust alpha_cum to balance the increased weight from mean normalization
    # and the higher saturation ceiling
    alpha_cum = 0.20
    penalty = alpha_cum * max_d * cum_sat * inv_d_sq_norm
    
    # 3. Current tour penalty
    # Explicit penalty for edges in the current local optimum
    alpha_curr = 0.05
    penalty += alpha_curr * max_d * tour_mask * inv_d_sq_norm
    
    # Add penalty to distances
    d_out += penalty
    
    # Ensure symmetry just in case
    d_out = (d_out + d_out.T) / 2.0

    # Safety: ensure finite
    if not np.all(np.isfinite(d_out)):
        return edge_distance.copy()
        
    return d_out
