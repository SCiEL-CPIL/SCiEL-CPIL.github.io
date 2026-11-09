import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Extract return costs to depot (node 0)
    d_j0 = distance_matrix[:, 0]
    
    # Minimum path length from depot to node i
    d_0i = distance_matrix[0, :]
    
    # Feasibility: min path 0->i->j->0 must be within max_len
    min_path = d_0i[:, None] + distance_matrix + d_j0[None, :]
    feasible = min_path <= max_len
    
    # Denominator: total budget consumed by transition i->j and return j->0
    denom = distance_matrix + d_j0[None, :]
    
    # Numerator: prize of destination node j
    num = prizes[None, :]
    
    # Safe division
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    
    eta = num / safe_denom
    
    # Apply feasibility mask
    eta = np.where(feasible, eta, 1e-9)
    
    # Enforce non-positive values become 1e-9
    eta = np.where(eta > 0, eta, 1e-9)
    
    # Normalize to [0, 1] to balance with pheromone (starts at 1)
    max_eta = np.max(eta)
    if max_eta > 1e-12:
        eta = eta / max_eta
    
    # Sharpen with power 1.5 to emphasize high-value edges
    eta = eta ** 1.5
    
    # Re-apply minimum after power
    eta = np.where(eta > 1e-9, eta, 1e-9)
    
    return eta
