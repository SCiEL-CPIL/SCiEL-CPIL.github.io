import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    D = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    d = np.asarray(demands, dtype=np.float64)
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    
    # Base heuristic: inverse squared distance
    D_safe = np.where(D > eps, D, 1.0)
    base = 1.0 / (D_safe ** 2)
    
    # Compute savings matrix: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    D0 = D[0, :]
    D0_i = D0[:, np.newaxis]
    D0_j = D0[np.newaxis, :]
    
    S = D0_i + D0_j - D
    Denom = D0_i + D0_j
    Denom_safe = np.where(Denom > eps, Denom, 1.0)
    S_norm = S / Denom_safe
    
    # Clip savings to be non-negative
    S_norm = np.maximum(S_norm, 0.0)
    
    # Normalize savings to [0, 1] range
    max_savings = np.max(S_norm)
    if max_savings > 0:
        S_norm = S_norm / max_savings
    else:
        S_norm = np.zeros_like(S_norm)
    
    # Demand weight: prioritize consolidating high-demand customers
    d_j = d[np.newaxis, :]  # Shape (1, N)
    demand_weight = 1.0 + 0.1 * d_j
    
    # Savings weight
    savings_weight = 1.0 + 2.0 * S_norm
    
    # Angular continuity: reward edges that maintain smooth route direction
    # For edge (i,j), compute the angle between vectors (depot->i) and (i->j)
    depot_coords = coords[0]
    angular_score = np.ones((n, n))
    
    for i in range(1, n):
        vec_depot_i = coords[i] - depot_coords
        norm_di = np.linalg.norm(vec_depot_i)
        if norm_di < 1e-10:
            continue
        vec_depot_i_unit = vec_depot_i / norm_di
        
        for j in range(1, n):
            if i == j:
                continue
            vec_i_j = coords[j] - coords[i]
            norm_ij = np.linalg.norm(vec_i_j)
            if norm_ij < 1e-10:
                continue
            vec_i_j_unit = vec_i_j / norm_ij
            
            # Cosine of angle between depot->i and i->j
            cos_angle = np.dot(vec_depot_i_unit, vec_i_j_unit)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Reward small angles (smooth continuation), penalize large angles
            # Map cos_angle from [-1, 1] to [0.5, 1.5] so it's always positive
            # and peaks at 1.0 when angles match perfectly
            angular_score[i, j] = 0.5 + 0.5 * (cos_angle + 1.0) / 2.0
            angular_score[j, i] = 0.5 + 0.5 * (cos_angle + 1.0) / 2.0
    
    # Combine all components
    eta = base * demand_weight * savings_weight * angular_score
    
    # Ensure positive
    eta = np.maximum(eta, 1e-9)
    
    # Replace non-finite
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    return eta
