import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    t0 = time.time()
    _DEADLINE = 322.0
    
    prizes = np.asarray(prizes, dtype=np.float64)
    d = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    n = len(prizes)
    
    eps = 1e-9
    
    # Ensure prizes[0] is 0
    prizes = prizes.copy()
    prizes[0] = 0.0
    
    # 1. Compute Cluster Value for each node j
    # Cluster value is the sum of prizes of the k nearest neighbors
    k = 5
    if n > k:
        # Get k nearest neighbors for each node
        # d is (n, n). For each row i, find indices of k smallest values (excluding self)
        cluster_val = np.zeros(n)
        for j in range(n):
            # Exclude self
            d_j = d[j].copy()
            d_j[j] = np.inf
            # Get k smallest
            if k < n:
                k_nearest = np.argpartition(d_j, k)[:k]
            else:
                k_nearest = np.arange(n)
                k_nearest = k_nearest[k_nearest != j]
            
            cluster_val[j] = np.sum(prizes[k_nearest])
    else:
        cluster_val = prizes.copy()
        cluster_val[0] = 0.0
        
    # 2. Compute Incremental Cost Matrix
    # cost[i, j] = d[i, j] + d[j, 0] - d[i, 0]
    # This represents the additional length required to go from i to j and then to depot,
    # compared to going from i to depot directly.
    d_to_depot = d[:, 0]
    incremental_cost = d + d_to_depot[None, :] - d_to_depot[:, None]
    
    # Ensure costs are not negative (triangle inequality should hold, but be safe)
    incremental_cost = np.maximum(incremental_cost, 0.0)
    
    # 3. Construct Heuristic
    # eta[i, j] = (prize[j] + beta * cluster_val[j]) / (incremental_cost[i, j] + eps)
    beta = 0.5  # Weight of cluster value vs direct prize
    
    numerator = prizes[None, :] + beta * cluster_val[None, :]
    # Numerator is (1, n), broadcast to (n, n)
    
    eta = numerator / (incremental_cost + eps)
    
    # 4. Feasibility Pre-filtering (Soft)
    # If a move is clearly infeasible (cost > max_len - current_min_len), it's bad.
    # But the ant checks exact feasibility. 
    # However, if incremental_cost[i,j] > max_len, it's likely infeasible from start.
    # We can dampen these heavily.
    # The max possible current length is 0 (at start).
    # So if d[i,j] + d[j,0] > max_len, it's infeasible from start.
    # For i > 0, current length > 0, so constraint is tighter.
    
    # Let's apply a hard mask for moves that are infeasible even from the start (i=0 case)
    # or generally very expensive.
    # Actually, the ant logic: 
    # if current_len + d[i,j] + d[j,0] <= max_len:
    # So d[i,j] + d[j,0] <= max_len - current_len.
    
    # Since heuristic is static, we don't know current_len.
    # But we know d[i,j] + d[j,0] is the "commitment".
    # If commitment > max_len, it's infeasible from start.
    commitment = d + d_to_depot[None, :]
    infeasible_from_start = commitment > max_len
    eta[infeasible_from_start] = eps
    
    # Diagonal
    np.fill_diagonal(eta, eps)
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    eta = np.clip(eta, eps, 1e6)
    
    if time.time() - t0 > _DEADLINE:
        return eta
        
    return eta
