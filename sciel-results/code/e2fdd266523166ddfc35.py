"""Guided local search penalty rule using a continuous repulsive force field."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_DEADLINE = 290.0  # Safety margin before 300s limit

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Returns a modified distance matrix implementing a repulsive force field.
    
    Strategy:
    1. Reconstruct the geometric coordinates of the points from the distance matrix.
       This is possible because the points are in a known space (unit square) and the
       distance matrix is Euclidean. We use the first 3 points to define the coordinate
       system and project all other points.
    2. Identify the edges in the current local_opt_tour.
    3. For every edge (i, j) NOT in the current tour, calculate the minimum Euclidean
       distance from the midpoint of (i, j) to any segment in the current tour.
    4. Apply a penalty to D_mod[i, j] that decays exponentially with this distance.
       This creates a "soft" exclusion zone around the current tour.
    5. Also apply a small history-based penalty to edges frequently used in the past.
    """
    # Safety check
    if time.monotonic() - _START_TIME > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        return D.copy()
    
    n = D.shape[0]
    if n < 3:
        return D.copy()

    # 1. Reconstruct coordinates
    # We need to embed the points in 2D.
    # Let p0 = (0,0), p1 = (d01, 0).
    # p2 = (x2, y2) where x2 = (d02^2 + d01^2 - d12^2) / (2*d01)
    # y2 = sqrt(d02^2 - x2^2)
    
    d01 = D[0, 1]
    d02 = D[0, 2]
    d12 = D[1, 2]
    
    if d01 < 1e-10:
        # Degenerate case, points 0 and 1 are too close. Try different pivot.
        # For simplicity in this specific task context (random points), this is unlikely.
        # If it happens, fall back to simple penalty.
        return _simple_penalty_fallback(D, local_opt_tour, edge_n_used)

    x2 = (d02**2 + d01**2 - d12**2) / (2.0 * d01)
    y2_sq = d02**2 - x2**2
    if y2_sq < 0:
        y2_sq = 0.0
    y2 = np.sqrt(y2_sq)
    
    coords = np.zeros((n, 2))
    coords[0] = [0.0, 0.0]
    coords[1] = [d01, 0.0]
    coords[2] = [x2, y2]
    
    # Project remaining points using trilateration
    for i in range(3, n):
        d0i = D[0, i]
        d1i = D[1, i]
        d2i = D[2, i]
        
        # Distance to p0 and p1
        x_i_01 = (d0i**2 - d1i**2 + d01**2) / (2.0 * d01)
        y_i_01_sq = d0i**2 - x_i_01**2
        if y_i_01_sq < 0: y_i_01_sq = 0.0
        y_i_01 = np.sqrt(y_i_01_sq)
        
        # Distance to p0 and p2
        d02_len = d02 # distance between p0 and p2
        x_i_02 = (d0i**2 - d2i**2 + d02_len**2) / (2.0 * d02_len)
        y_i_02_sq = d0i**2 - x_i_02**2
        if y_i_02_sq < 0: y_i_02_sq = 0.0
        y_i_02 = np.sqrt(y_i_02_sq)
        
        # Solve for coordinates relative to p0=(0,0) and p2=(x2, y2)
        # x_i^2 + y_i^2 = d0i^2
        # (x_i - x2)^2 + (y_i - y2)^2 = d2i^2
        # x_i^2 - 2*x_i*x2 + x2^2 + y_i^2 - 2*y_i*y2 + y2^2 = d2i^2
        # d0i^2 - 2*(x_i*x2 + y_i*y2) + d02_len^2 = d2i^2
        # x_i*x2 + y_i*y2 = (d0i^2 + d02_len^2 - d2i^2) / 2
        
        # This is a system of linear equations for x_i, y_i if we had the angle.
        # Easier: Use the angle from p0.
        # Let theta be the angle of p_i from p0.
        # x_i = d0i * cos(theta), y_i = d0i * sin(theta)
        
        # We can find the angle using the law of cosines in triangle 0-2-i.
        # cos(angle_0_2_i) = (d02_len^2 + d0i^2 - d2i^2) / (2 * d02_len * d0i)
        if d0i < 1e-10:
            coords[i] = [0.0, 0.0]
            continue
            
        cos_alpha = (d02_len**2 + d0i**2 - d2i**2) / (2.0 * d02_len * d0i)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        
        # Angle of p2 from p0
        angle_p2 = np.arctan2(y2, x2)
        
        # There are two possible positions for p_i: angle_p2 + alpha or angle_p2 - alpha.
        # We can determine the correct one by checking distance to p1.
        # Pos 1:
        angle_1 = angle_p2 + alpha
        x_cand1 = d0i * np.cos(angle_1)
        y_cand1 = d0i * np.sin(angle_1)
        
        # Pos 2:
        angle_2 = angle_p2 - alpha
        x_cand2 = d0i * np.cos(angle_2)
        y_cand2 = d0i * np.sin(angle_2)
        
        # Check which candidate is closer to p1=(d01, 0) in terms of squared distance matching d1i^2
        d1i_sq = d1i**2
        dist_sq_1 = (x_cand1 - d01)**2 + y_cand1**2
        dist_sq_2 = (x_cand2 - d01)**2 + y_cand2**2
        
        if abs(dist_sq_1 - d1i_sq) < abs(dist_sq_2 - d1i_sq):
            coords[i] = [x_cand1, y_cand1]
        else:
            coords[i] = [x_cand2, y_cand2]
            
    # 2. Identify edges in current tour
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v: u, v = v, u
        tour_edges.add((u, v))
        
    # 3. Calculate penalty for non-tour edges
    D_mod = D.copy()
    
    # Parameters for the force field
    # We want the penalty to be significant for edges close to the tour.
    # Let's say the "radius" of influence is roughly the average edge length.
    triu_idx = np.triu_indices(n, k=1)
    avg_edge_len = np.mean(D[triu_idx])
    
    # Decay constant. Smaller sigma = sharper peak.
    # If sigma is avg_edge_len, then at distance avg_edge_len, penalty is exp(-1) ~ 0.37 of max.
    sigma = 0.5 * avg_edge_len 
    
    # Max penalty magnitude. Should be enough to discourage but not dominate.
    # 0.5 * avg_edge_len seems reasonable.
    max_penalty = 0.5 * avg_edge_len
    
    # Pre-compute tour segments for distance calculation
    # We need min distance from midpoint of (i,j) to any segment (u,v) in tour.
    # To speed this up, we can use a grid or just loop if n is small (100).
    # n=100, edges ~ 5000. Tour edges = 100.
    # 5000 * 100 = 500,000 distance calculations. This is fast in Python if vectorized per edge.
    
    # Let's vectorize the distance calculation for all non-tour edges at once?
    # It might be memory heavy. 5000 edges * 100 segments * 2 coords.
    # Let's do it in chunks or just loop over edges and vectorize over segments.
    
    # List of non-tour edges
    non_tour_edges = []
    for u, v in zip(triu_idx[0], triiu_idx[1]):
        if (u, v) not in tour_edges:
            non_tour_edges.append((u, v))
            
    if not non_tour_edges:
        return D_mod
        
    # Convert to numpy array
    edge_arr = np.array(non_tour_edges)
    
    # Midpoints of these edges
    midpoints = (coords[edge_arr[:, 0]] + coords[edge_arr[:, 1]]) / 2.0
    
    # Tour segments endpoints
    tour_u = np.array([local_opt_tour[i] for i in range(n)])
    tour_v = np.array([local_opt_tour[(i + 1) % n] for i in range(n)])
    seg_start = coords[tour_u]
    seg_end = coords[tour_v]
    
    # Calculate distance from each midpoint to each segment
    # We do this in batches to manage memory
    batch_size = 500
    penalties = np.zeros(len(edge_arr))
    
    for start in range(0, len(edge_arr), batch_size):
        end = min(start + batch_size, len(edge_arr))
        mids = midpoints[start:end] # Shape (B, 2)
        
        # Vectorized distance from points to segments
        # mids: (B, 2)
        # seg_start: (N, 2)
        # seg_end: (N, 2)
        
        # Expand dims: mids -> (B, 1, 2), seg_start -> (1, N, 2), seg_end -> (1, N, 2)
        m = mids[:, np.newaxis, :]
        s0 = seg_start[np.newaxis, :, :]
        s1 = seg_end[np.newaxis, :, :]
        
        # Vector v = s1 - s0 (1, N, 2)
        v = s1 - s0
        # Vector w = m - s0 (B, N, 2)
        w = m - s0
        
        # Projection factor t = dot(w, v) / dot(v, v)
        dot_wv = np.sum(w * v, axis=2) # (B, N)
        dot_vv = np.sum(v * v, axis=2) # (N,)
        
        # Avoid division by zero
        dot_vv_safe = np.where(dot_vv > 1e-10, dot_vv, 1e-10)
        t = dot_wv / dot_vv_safe[np.newaxis, :] # (B, N)
        
        # Clamp t to [0, 1]
        t = np.clip(t, 0.0, 1.0)
        
        # Closest point on segment: s0 + t * v
        # s0: (1, N, 2), t: (B, N, 1)
        closest = s0 + t[:, :, np.newaxis] * v
        
        # Distance squared
        diff = m - closest # (B, N, 2)
        dist_sq = np.sum(diff * diff, axis=2) # (B, N)
        
        # Min distance for each midpoint
        min_dist_sq = np.min(dist_sq, axis=1) # (B,)
        min_dist = np.sqrt(min_dist_sq)
        
        # Calculate penalty
        # Penalty = max_penalty * exp(-min_dist / sigma)
        # But we only want to penalize if it's close?
        # The exponential decay handles this naturally.
        # However, we should not penalize edges that are part of the tour (already excluded).
        # We should also consider that 2-opt swaps involve 4 edges.
        # If we penalize edges near the tour, we discourage the tour from staying similar.
        
        # Apply decay based on round?
        max_used = np.max(edge_n_used)
        if max_used < 1:
            round_idx = 1
        else:
            round_idx = int(max_used)
            
        # Decay the strength of the field as we progress
        decay_factor = 0.95 ** round_idx
        current_max_penalty = max_penalty * decay_factor
        
        batch_penalties = current_max_penalty * np.exp(-min_dist / sigma)
        
        # Add to D_mod
        for k in range(len(edge_arr[start:end])):
            u, v = edge_arr[start + k]
            D_mod[u, v] += batch_penalties[k]
            D_mod[v, u] += batch_penalties[k]
            
    # 4. Add history-based penalty (small, to break ties or reinforce avoidance of old bad edges)
    if max_used > 0:
        norm_used = edge_n_used / max_used
        # Only apply to non-zero usage to save time? No, vectorized is fast.
        # Small factor
        history_factor = 0.1 * avg_edge_len * (0.95 ** round_idx)
        D_mod += history_factor * norm_used
        
    # 5. Ensure symmetry and validity
    D_mod = (D_mod + D_mod.T) / 2.0
    np.fill_diagonal(D_mod, 0.0)
    
    if not np.all(np.isfinite(D_mod)):
        return D.copy()
        
    return D_mod

def _simple_penalty_fallback(D, local_opt_tour, edge_n_used):
    """Fallback if coordinate reconstruction fails or is degenerate."""
    D_mod = D.copy()
    n = D.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    med_len = np.median(D[triu_idx])
    
    max_used = np.max(edge_n_used)
    if max_used < 1:
        round_idx = 1
    else:
        round_idx = int(max_used)
        
    decay = 0.95 ** round_idx
    base_penalty = 0.5 * med_len * decay
    
    for i in range(n):
        j = local_opt_tour[(i + 1) % n]
        D_mod[i, j] += base_penalty
        D_mod[j, i] += base_penalty
        
    if max_used > 0:
        norm_used = edge_n_used / max_used
        history_factor = 0.2 * med_len * decay
        D_mod += history_factor * norm_used
        
    D_mod = (D_mod + D_mod.T) / 2.0
    np.fill_diagonal(D_mod, 0.0)
    return D_mod
# EVOLVE-BLOCK-END
