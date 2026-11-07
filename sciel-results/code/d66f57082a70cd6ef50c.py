import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Deadline guard
    start_time = time.time()
    
    # 1. Precompute feasibility and potential value for each node j
    # For a node j to be a good next step from i, it must satisfy:
    # current_path_len + d[i,j] + d[j,0] <= max_len
    # Since current_path_len varies, we approximate the "commitment" of visiting j
    # by the cost to go from i to j and then back to 0.
    # However, the heuristic is static. We can precompute the "potential" of node j:
    # How much prize can we potentially collect if we visit j?
    # A simple proxy is the sum of prizes of nodes k that are close to j and 0.
    # Or better: the prize of j itself, weighted by how much "slack" it leaves.
    
    # Let's try a different structural approach:
    # Heuristic[i,j] = (Prize[j] + Sum of prizes of nodes reachable from j) / (d[i,j] + d[j,0])^alpha
    # This rewards nodes that are not only valuable themselves but also hubs to other value.
    
    # Precompute "hub potential" for each node j
    # A node k is reachable from j if d[j,k] + d[k,0] <= max_len
    # This is a necessary condition for k to be visited after j (assuming we start fresh from j? No, path length matters).
    # But for a static heuristic, this is a reasonable proxy for "connectivity to value".
    
    hub_potential = np.zeros(n)
    for j in range(1, n):
        # Sum of prizes of nodes k that are "compatible" with j
        # Compatible means: if we are at j, can we still go to k and return to 0?
        # The most optimistic case is if we just arrived at j from 0 (path len = d[0,j]).
        # Then remaining budget for j->k->0 is max_len - d[0,j].
        # So condition: d[j,k] + d[k,0] <= max_len - d[0,j]
        # If this holds, k is a "likely" next node after j.
        
        total_potential = prizes[j]
        for k in range(1, n):
            if k == j:
                continue
            # Check if k is reachable from j given we came from 0
            if distance_matrix[j, k] + distance_matrix[k, 0] <= max_len - distance_matrix[0, j]:
                total_potential += prizes[k]
        hub_potential[j] = total_potential
    
    # Normalize hub potential to [0, 1] to avoid scale issues
    max_hp = np.max(hub_potential)
    if max_hp > 0:
        hub_potential = hub_potential / max_hp
    
    # 2. Compute the heuristic matrix
    eps = 1e-9
    heuristic = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 1e-9
                continue
            
            d_ij = distance_matrix[i, j]
            d_j0 = distance_matrix[j, 0]
            
            # Feasibility check: Is it even possible to visit j from i and return to 0?
            # Even if we start at i with 0 path length, we need d[i,j] + d[j,0] <= max_len
            # If not, this edge is effectively useless for any ant.
            if d_ij + d_j0 > max_len:
                heuristic[i, j] = 1e-9
            else:
                # The "cost" of taking this edge is d_ij + d_j0 (the commitment)
                # The "benefit" is the hub potential of j
                # We use a ratio, but soften the denominator to not be too harsh on distance
                cost = d_ij + d_j0
                benefit = hub_potential[j]
                
                if benefit <= 0:
                    heuristic[i, j] = 1e-9
                else:
                    # Use a power to control the trade-off
                    # Higher alpha means more sensitive to cost
                    alpha = 0.5
                    heuristic[i, j] = benefit / (cost + eps)**alpha
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() - start_time > 290:
        # Safety margin: return immediately
        pass
    
    return heuristic
