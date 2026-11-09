import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    d = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Effective cost: distance to node + distance back to depot
    # d[j, 0] is the return cost for node j
    return_cost = d[:, 0] # shape (n,)
    
    # For move i -> j, the cost is d[i, j] + d[j, 0]
    # We want to maximize prize / cost
    # To avoid division by zero and handle infeasible moves (though ACO checks feasibility, 
    # a low heuristic for infeasible moves is good practice)
    
    # Construct the cost matrix: cost[i, j] = d[i, j] + d[j, 0]
    cost = d + return_cost[None, :]
    
    # Ensure no zeros
    cost = np.where(cost < 1e-9, 1e-9, cost)
    
    # Base heuristic
    H = (prizes[None, :] + 1e-9) / cost
    
    # Sharpen the heuristic to make the ACO more greedy/exploitative
    # This often helps in small problems
    H = H ** 1.2
    
    # Ensure positivity
    H = np.where(H > 0, H, 1e-9)
    
    return H
