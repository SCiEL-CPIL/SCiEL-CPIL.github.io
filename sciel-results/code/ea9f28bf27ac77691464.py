import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    n = len(prizes)
    d = np.asarray(distance_matrix, dtype=float)
    prizes = np.asarray(prizes, dtype=float)
    
    # Initialize heuristic
    heuristic = np.zeros((n, n))
    
    # Budget-Aware Expected Utility Strategy:
    # eta[i, j] = (prize[j] / (d[i,j] + d[j,0])^1.5) * exp(-lambda * (d[i,j] + d[j,0]) / max_len)
    #
    # Rationale:
    # 1. d[i,j] + d[j,0] is the "effective cost" of visiting j from i, 
    #    because the ant must eventually return to 0. This term represents 
    #    the total distance budget consumed by the decision to go to j.
    # 2. The power 1.5 penalizes long effective costs more than linearly, 
    #    favoring high-prize nodes that are "cheap" in terms of total round-trip impact.
    # 3. The exponential term exp(-lambda * cost / L_max) explicitly penalizes 
    #    moves that consume a large fraction of the remaining length budget.
    #    This prevents ants from getting stuck far from the depot with insufficient 
    #    range to return or visit other nodes, a common failure mode in orienteering.
    
    # Vectorized computation
    # d[i, j] is the distance from i to j
    # d[j, 0] is the distance from j back to start
    # We need a matrix E where E[i, j] = d[i, j] + d[j, 0]
    
    # d is (n, n)
    # d[:, 0] is (n, 1) -> distance from any node to 0
    # We want E[i, j] = d[i, j] + d[j, 0]
    # d[j, 0] for a fixed j is the j-th element of the vector d[:, 0]
    # So we can broadcast:
    # d is (n, n)
    # d_to_start is (n,) -> d[j, 0] for each j
    d_to_start = d[:, 0]  # Shape (n,)
    
    # E[i, j] = d[i, j] + d_to_start[j]
    # d[i, j] is accessed as d[i, :]
    # We add d_to_start (broadcasted across rows) to each row of d
    effective_cost = d + d_to_start[np.newaxis, :]  # Shape (n, n)
    
    # Avoid division by zero or very small numbers
    # effective_cost[i, i] might be d[i,i] + d[i,0]. d[i,i] is usually 0.
    # If i=0, effective_cost[0,0] = 0 + 0 = 0.
    # We set diagonal to 0 in the final heuristic anyway, but need to avoid 0/0 in intermediate steps.
    safe_cost = np.maximum(effective_cost, 1e-9)
    
    # Calculate the base utility: prize[j] / (cost^1.5)
    # prize[j] is broadcasted across rows
    prize_vec = prizes[np.newaxis, :]  # Shape (1, n)
    
    base_utility = prize_vec / (safe_cost ** 1.5)
    
    # Calculate the exponential penalty
    # lambda controls the strength of the budget penalty.
    # max_len is the total budget.
    # If cost is close to max_len, the penalty is strong.
    # Let's choose lambda = 2.0 as a reasonable starting point for unit square distances.
    # max_len is given as 3.0 in the task description, but we use the parameter.
    lambda_penalty = 2.0
    
    # cost / max_len
    # If max_len is 0, we have issues, but task says max_len=3.
    ratio = effective_cost / max_len
    
    # exp(-lambda * ratio)
    penalty = np.exp(-lambda_penalty * ratio)
    
    # Combine
    heuristic = base_utility * penalty
    
    # Zero out diagonal (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure non-negative and finite
    heuristic = np.maximum(heuristic, 0.0)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 0.0)
    
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    # This implies we should replace 0 (and negative) with 1e-9.
    # However, if we set all non-tour edges to 1e-9, the ACO might be too stochastic.
    # The current program's score of 1.43 suggests that some structure is good.
    # But this new strategy provides a *smooth* landscape, not a sparse one.
    # Let's normalize the heuristic to a reasonable scale.
    # The ACO uses pheromone * heuristic. Pheromone starts at 1.
    # If heuristic values are very small (e.g., 1e-5), the choice is dominated by pheromone initially.
    # If they are large, heuristic dominates.
    # Let's scale the heuristic so that the maximum value is around 10.
    # This gives heuristic a strong initial influence but allows pheromone to adapt.
    
    max_h = heuristic.max()
    if max_h > 0:
        heuristic = (heuristic / max_h) * 10.0
    else:
        heuristic = np.ones((n, n)) * 1e-9
    
    # Apply the rule: entries not positive become 1e-9
    # Since we normalized to positive, only zeros remain.
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Final sanity check
    if time.time() > deadline:
        return np.ones((n, n)) * 1e-9
        
    return heuristic
