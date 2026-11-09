import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = len(prizes)
    eps = 1e-9
    
    # Pre-compute distances to depot
    dist_to_depot = distance_matrix[:, 0]  # shape (n,)
    
    # Compute "remaining budget" feasibility for each node j
    # Node j is reachable from i if d[i,j] + d[j,0] <= max_len
    # We create a mask for feasible transitions
    # dist_matrix[i,j] + dist_to_depot[j] <= max_len
    
    # For the heuristic, we want to emphasize:
    # 1. High prize nodes
    # 2. Nodes that are close to current node i (low d[i,j])
    # 3. Nodes that leave enough budget to return to depot (d[j,0] small)
    
    # Strategy: H[i,j] = prize[j] / (d[i,j] + d[j,0] + eps)
    # This is the "greedy ratio" suggested in the directive.
    # It maximizes prize density relative to the total cost of visiting j and returning.
    
    # Let's also consider a combined approach that was successful in inspirations:
    # Global value: prize[j] / d[j,0] (rewards valuable nodes close to depot)
    # Local penalty: 1 / d[i,j] (rewards close transitions)
    
    # Let's try the directive's suggestion first: prizes[j] / (dist[i,j] + dist[j,0])
    # This treats the edge choice as maximizing immediate prize density relative to remaining length budget.
    
    # Compute the denominator: dist[i,j] + dist[j,0]
    # dist[i,j] is distance_matrix
    # dist[j,0] is dist_to_depot[j], which we broadcast as a row
    denom = distance_matrix + dist_to_depot[np.newaxis, :]
    
    # Avoid division by zero
    safe_denom = denom + eps
    
    # H[i,j] = prizes[j] / safe_denom[i,j]
    prize_row = prizes[np.newaxis, :]  # shape (1, n)
    H = prize_row / safe_denom
    
    # Ensure non-finite values are handled
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure positive
    H = np.maximum(H, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return H
    
    # Optional: Refine with centrality or other factors if time permits
    # For now, this simple greedy ratio might be strong.
    # Let's also try combining with a centrality factor if it improves things.
    # The previous best used centrality. Let's see if adding it helps.
    
    # Calculate centrality: inverse of mean distance to other nodes
    dist_sums = distance_matrix.sum(axis=1) - distance_matrix[np.arange(n), np.arange(n)]
    mean_dists = dist_sums / (n - 1)
    centrality = 1.0 / (mean_dists + eps)
    
    # Combine: H_combined[i,j] = H[i,j] * centrality[j]
    # This rewards nodes that are central (easy to reach from anywhere)
    centrality_row = centrality[np.newaxis, :]
    H_combined = H * centrality_row
    
    # Ensure non-finite values are handled
    H_combined = np.where(np.isfinite(H_combined), H_combined, 1e-9)
    H_combined = np.maximum(H_combined, 1e-9)
    
    # Check time again
    if time.time() > deadline:
        return H_combined
    
    # Return the combined heuristic
    return H_combined
