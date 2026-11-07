import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290.0

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    eps = 1e-9
    
    # 1. Compute distance from start (node 0) to all nodes
    d_0 = distance_matrix[0, :]
    
    # 2. Compute Max Prize Reachable from each node i
    # Assumption: Path 0 -> i -> k -> 0 is feasible if d[0,i] + d[i,k] + d[k,0] <= max_len
    # So d[i,k] + d[k,0] <= max_len - d[0,i]
    
    max_p_reachable = np.zeros(n)
    
    # Vectorized approach for Max P
    # For each i, we need max P[k] where d[i,k] + d[k,0] <= max_len - d[0,i]
    
    # d_k0 is distance from k to 0
    d_k0 = distance_matrix[:, 0]
    
    # Matrix of d[i,k] + d[k,0]
    # d[i,k] is distance_matrix
    # d[k,0] is d_k0[None, :]
    total_cost = distance_matrix + d_k0[None, :]
    
    # Budget remaining for each i
    budget_remaining = max_len - d_0
    
    # Feasibility mask: total_cost[i,k] <= budget_remaining[i]
    feasible = total_cost <= budget_remaining[:, None]
    
    # Set prizes of infeasible nodes to 0 for max calculation
    # We want max P[k] for feasible k.
    # Create a matrix where infeasible entries are 0 (or -inf if using log, but P>=0 so 0 is fine)
    prizes_masked = np.where(feasible, prizes[None, :], 0.0)
    
    # Max prize for each row i
    max_p_reachable = np.max(prizes_masked, axis=1)
    
    # If no feasible node (max is 0), set to a small value to avoid division by zero later
    # But if max is 0, it means no node k (including i itself?) is feasible?
    # Note: k can be i? No, we move to unvisited j. 
    # But for the heuristic calculation, we consider all j.
    # If max_p_reachable[i] is 0, it means no node is reachable from i within budget.
    # In that case, the ant shouldn't move. But the heuristic is static.
    # We set max_p_reachable to eps to avoid div by 0.
    max_p_reachable[max_p_reachable < eps] = eps
    
    # 3. Compute Dual Distance Cost C_ij
    # C_ij = d[i,j] * d[j,0] + d[i,j] + d[j,0]
    d_j0_row = d_k0[None, :]
    prod_term = distance_matrix * d_j0_row
    sum_term = distance_matrix + d_j0_row
    cost_matrix = prod_term + sum_term + eps
    
    # 4. Compute Heuristic
    # eta_ij = (P_j^2) / (MaxP_i * C_ij)
    # P_j is prizes[None, :]
    # MaxP_i is max_p_reachable[:, None]
    
    numerator = (prizes[None, :] ** 2) + eps
    denominator = (max_p_reachable[:, None] * cost_matrix) + eps
    
    heuristic = numerator / denominator
    
    # Clean up
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    if time.time() > deadline:
        return (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)
        
    return heuristic
