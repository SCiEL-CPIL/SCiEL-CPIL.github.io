import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization of orienteering.
    
    Uses a 'Local Density of Value' heuristic that rewards moves into high-value clusters.
    For each node j, we compute a prize-weighted average distance to its top-k nearest neighbors.
    The edge attractiveness eta[i,j] is proportional to prize[j] divided by this local density,
    which explicitly rewards collecting multiple high-prize nodes that are geographically close.
    
    We also incorporate return-cost awareness (d[i,j] + d[j,0]) and a distance penalty
    relative to max_len to balance immediate prize value against path length constraints.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    eps = 1e-9
    
    # --- Step 1: Compute local density of value for each node ---
    # For each node j, find its top-k nearest neighbors (excluding itself)
    # and compute a prize-weighted average distance to them.
    # This creates a "local density" metric: low density = isolated, high density = clustered.
    
    k = min(5, n - 1)  # top-k neighbors
    
    # Compute pairwise distances (already given as distance_matrix)
    # For each node j, find k nearest neighbors
    local_density = np.zeros(n, dtype=float)
    
    for j in range(n):
        # Get distances from all other nodes to j
        dists_to_j = distance_matrix[:, j].copy()
        dists_to_j[j] = np.inf  # Exclude self
        
        # Find top-k nearest neighbors
        if n > 1:
            k_actual = min(k, n - 1)
            nearest_indices = np.argpartition(dists_to_j, k_actual)[:k_actual]
            nearest_dists = dists_to_j[nearest_indices]
            nearest_prizes = prizes[nearest_indices]
            
            # Prize-weighted average distance: sum(prize_m * dist_m) / sum(prize_m)
            # If all prizes are 0, fall back to simple average distance
            total_prize = np.sum(nearest_prizes)
            if total_prize > eps:
                local_density[j] = np.sum(nearest_prizes * nearest_dists) / total_prize
            else:
                local_density[j] = np.mean(nearest_dists) if len(nearest_dists) > 0 else 1.0
        else:
            local_density[j] = 1.0
    
    # Ensure local_density is positive and finite
    local_density = np.where(np.isfinite(local_density) & (local_density > 0), local_density, 1.0)
    
    # --- Step 2: Compute effective cost for each edge (i, j) ---
    # d[i,j] + d[j,0] accounts for the full detour cost
    d_to_start = distance_matrix[:, 0].copy()  # shape (n,)
    eff_cost = distance_matrix + d_to_start[None, :]  # shape (n, n)
    eff_cost_safe = eff_cost + eps
    
    # --- Step 3: Compute the heuristic ---
    # eta[i,j] = (prize[j] / local_density[j])^alpha * (1 / eff_cost[i,j])^gamma * dist_penalty
    
    # Parameters
    alpha = 1.5  # Amplify prize/density ratio
    gamma = 2.0  # Amplify cost efficiency
    beta = 1.5   # Distance penalty strength
    
    # Prize divided by local density (reward high-value clusters)
    prize_density = (prizes[None, :] + eps) / local_density[None, :]  # shape (1, n)
    
    # Cost efficiency component
    cost_eff = (1.0 / eff_cost_safe) ** gamma
    
    # Distance penalty: penalize edges that are long relative to max_len
    dist_penalty = (1.0 / (1.0 + distance_matrix / max_len)) ** beta
    
    # Combine all components
    eta = (prize_density ** alpha) * cost_eff * dist_penalty
    
    # Set eta[i,0] = 1e-9 for all i (no point visiting node 0)
    eta[:, 0] = 1e-9
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check time
    if time.time() > deadline:
        # Return a simple fallback
        d_j0 = distance_matrix[:, 0]
        costs = distance_matrix + d_j0[None, :] + eps
        fallback = (prizes[None, :] + eps) / costs
        fallback[:, 0] = 1e-9
        fallback = np.where(np.isfinite(fallback) & (fallback > 0), fallback, 1e-9)
        return fallback
    
    return eta
