# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Regret-based heuristic for ant colony orienteering.
    
    The heuristic H[i,j] rewards moves to node j that:
    1. Have high prize value
    2. Are close to i (low travel cost)
    3. Preserve future routing flexibility (many feasible continuations from j)
    4. Don't trap the ant (avoid dead-ends within the tight length budget)
    
    Key insight: with max_len=3 and 50 nodes, the feasible set is very constrained.
    A move to j is only good if there are still many nodes k that can be visited
    after j while keeping total path length <= max_len. We compute for each node j
    the number of feasible continuation nodes, and use this as a flexibility score.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    
    # Ensure valid shape
    if D.shape != (n, n):
        raise ValueError("Invalid distance matrix shape")
    
    # Base attractiveness: prize_j / (d_ij + d_j0)
    # This accounts for the cost of going to j and returning to origin
    # d_j0 is the cost to return to start, critical for feasibility
    d_to_origin = D[:, 0]  # cost from each node to origin
    d_from_origin = D[0, :]  # cost from origin to each node
    
    # For each pair (i,j), the remaining budget after visiting j is:
    # max_len - (current_length + d[i,j] + d[j,0])
    # But since current_length varies, we use a proxy:
    # The "effective cost" of visiting j from i is d[i,j] + d[j,0]
    # Nodes with lower effective cost have more remaining budget
    
    # Compute flexibility score for each node j:
    # For node j, count how many other nodes k (k != j, k != 0) can be visited
    # after j while staying within max_len, assuming we came from some typical node.
    # We approximate by checking: d[j,k] + d[k,0] <= max_len - min_cost_to_j
    # where min_cost_to_j is the minimum d[i,j] over i != j, i != 0
    
    # Simpler approach: for each node j, count feasible continuations
    # assuming the path length so far is approximately d[0,j] (came from origin)
    # Remaining budget: max_len - d[0,j]
    # A node k is feasible after j if d[j,k] + d[k,0] <= remaining_budget
    
    flexibility = np.zeros(n)
    for j in range(1, n):
        remaining = max_len - d_from_origin[j]
        if remaining <= 0:
            flexibility[j] = 1.0  # No flexibility, but at least visitable
            continue
        # Count nodes k != j, k != 0 that can be visited after j
        feasible = 0
        for k in range(1, n):
            if k == j:
                continue
            if D[j, k] + d_to_origin[k] <= remaining + 1e-9:
                feasible += 1
        flexibility[j] = feasible + 1.0  # +1 to avoid zero
    
    # Flexibility ratio: higher means more options after visiting j
    # We want to reward nodes that preserve flexibility, so use 1/flexibility
    # Actually, we want to reward nodes with HIGH flexibility (many options)
    # So use flexibility directly as a multiplier
    
    # Base heuristic: prize_j / (d_ij + d_j0)
    # This is n x n
    effective_cost = D + d_to_origin[None, :]  # d[i,j] + d[j,0]
    base_heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                base_heuristic[i, j] = 0.0
                continue
            if effective_cost[i, j] < 1e-9:
                base_heuristic[i, j] = 0.0
                continue
            base_heuristic[i, j] = prizes[j] / effective_cost[i, j]
    
    # Apply flexibility: multiply by flexibility[j] / max_flexibility
    # Nodes with high flexibility (many continuations) get boosted
    max_flex = np.max(flexibility[1:]) if n > 1 else 1.0
    flex_factor = flexibility / max_flex  # normalized to [~0, 1]
    
    # Combine: base * flex_factor
    # This rewards moves to high-prize, close nodes that also preserve flexibility
    heuristic = base_heuristic * flex_factor[None, :]
    
    # Ensure node 0 column is zero (can't return to start in ACO sense, 
    # but the problem says ant starts at 0 and moves to unvisited nodes)
    # Actually, node 0 is the start, so moves TO node 0 don't happen in the ACO
    # The heuristic matrix should have zeros on the diagonal
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive (non-positive become 1e-9)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Check for non-finite values
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Verify shape
    if heuristic.shape != (n, n):
        raise ValueError("Heuristic shape mismatch")
    
    return heuristic
# EVOLVE-BLOCK-END
