"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Uses a combined heuristic that includes:
    1. Base: Prize / (Travel Cost + Return Cost)
    2. Regret factor: biases towards nodes that are significantly better than alternatives
    
    The heuristic is computed once per instance, before the colony runs.
    Non-positive entries are set to 1e-9.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n != distance_matrix.shape[0] or n != distance_matrix.shape[1]:
        raise ValueError("Shape mismatch")
    
    # Ensure prizes at node 0 is 0 (as per task description)
    # prizes[0] = 0.0
    
    # Compute base heuristic: prize / (distance from i to j + distance from j to 0)
    # This represents the "value density" of visiting j from i
    dist_to_j = distance_matrix  # n x n
    return_cost = distance_matrix[:, 0]  # n x 1 -> broadcast to n x n
    total_cost = dist_to_j + return_cost[None, :]  # n x n
    
    # Avoid division by zero
    total_cost_safe = np.maximum(total_cost, 1e-9)
    
    base_heuristic = prizes[None, :] / total_cost_safe
    
    # Now add a regret-based refinement:
    # For each row i, compute for each node j a "regret" factor that is higher
    # if j's prize is much larger than other reachable nodes' prizes
    # We use a soft-max style approach: for each i, compute the max prize among
    # other nodes k (k != j, k != 0) that are reachable from i within budget
    
    # First, determine which nodes are reachable from each i within the budget
    # A node j is reachable from i if: d[i,j] + d[j,0] <= max_len (roughly)
    # More precisely, the constraint is: current_length + d[i,j] + d[j,0] <= max_len
    # But since we don't know current_length at heuristic computation time,
    # we use a conservative estimate: assume current_length is 0, so d[i,j] + d[j,0] <= max_len
    
    reachable = (total_cost <= max_len).astype(np.float64)  # n x n
    reachable[:, 0] = 0.0  # Node 0 is not a valid destination
    
    # For each i, compute a "regret" factor for each j
    # regret[i, j] = 1 / (1 + max_{k != j, k != 0, reachable[i,k]} prize[k])
    # This makes j more attractive if other reachable nodes have lower prizes
    
    # To compute this efficiently, for each i, find the max prize among reachable nodes
    # excluding j itself
    max_prize_excl = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        # Get prizes of reachable nodes from i
        reachable_mask = reachable[i] > 0.5
        if not np.any(reachable_mask):
            continue
        
        # For each j, compute max prize among reachable nodes excluding j
        reachable_prizes = prizes * reachable_mask
        # Set prize of j to 0 for each j, then take max
        # This is O(n^2) per i, so O(n^3) total. For n=50, this is 125,000 ops, fine.
        for j in range(n):
            if j == 0:
                continue
            if not reachable_mask[j]:
                continue
            # Temporarily set j's prize to 0
            temp_prizes = reachable_prizes.copy()
            temp_prizes[j] = 0.0
            max_prize_excl[i, j] = np.max(temp_prizes) if np.any(temp_prizes > 0) else 0.0
    
    # Regret factor: higher when max_prize_excl is lower
    # Use a smooth version: regret = 1 / (1 + alpha * max_prize_excl)
    alpha = 1.0  # Tune this parameter
    regret_factor = 1.0 / (1.0 + alpha * max_prize_excl)
    
    # Combine base heuristic with regret factor
    heuristic = base_heuristic * regret_factor
    
    # Ensure all entries are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return heuristic
    
    return heuristic
# EVOLVE-BLOCK-END
