import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony orienteering.
    
    Structural change: Rank-Based Heuristic.
    Replaces the continuous power-law heuristic with a discrete rank-based
    attractiveness. For each node i, the potential next nodes j are ranked
    by their efficiency (prize[j] / detour[i,j]). The heuristic value is
    the inverse of the rank (1/rank).
    
    This provides a smoother gradient of attractiveness compared to the
    cubic power, potentially allowing the ACO to explore a broader set of
    candidates and escape local minima more effectively.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Calculate efficiency: prize[j] / (d[i,j] + d[j,0])
    detour = distance_matrix + distance_matrix[:, 0][None, :]
    detour_safe = np.where(detour > 1e-9, detour, 1e-9)
    
    # prize[j] broadcast along rows
    efficiency = prizes[None, :] / detour_safe
    
    # Initialize heuristic
    heuristic = np.full_like(efficiency, 1e-9)
    
    # Compute ranks for each node i
    for i in range(n):
        # Exclude self
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            continue
            
        # Get efficiencies for valid neighbors
        eff_vals = efficiency[i, valid_indices]
        
        # Sort indices by efficiency descending
        sorted_local = np.argsort(-eff_vals)
        
        # Assign heuristic values based on rank
        # Rank 0 -> 1st place -> 1/1
        # Rank 1 -> 2nd place -> 1/2
        for rank, local_idx in enumerate(sorted_local):
            j = valid_indices[local_idx]
            heuristic[i, j] = 1.0 / (rank + 1)
            
            # Deadline check
            if time.time() > deadline:
                return np.ones_like(distance_matrix) * 1e-9

    # Ensure validity
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
