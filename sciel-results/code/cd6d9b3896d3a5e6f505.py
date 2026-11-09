import numpy as np
def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Effective cost to visit j and return to 0
    # Note: d[i,j] is cost from i to j. d[j,0] is cost from j to 0.
    # The feasibility check is len + d[i,j] + d[j,0] <= max_len.
    # So the "incremental cost" in terms of budget consumption is d[i,j] + d[j,0].
    cost = distance_matrix + distance_matrix[:, 0][None, :] # Wait, distance_matrix[j, 0] is return cost.
    # distance_matrix is symmetric? Usually yes for Euclidean.
    # d[j,0] is column 0 of distance_matrix.
    return_cost = distance_matrix[:, 0] # shape (n,)
    # We need d[i,j] + d[j,0] for all i,j.
    # cost[i,j] = d[i,j] + d[j,0]
    cost_matrix = distance_matrix + return_cost[None, :]
    
    # Avoid division by zero
    cost_matrix = np.maximum(cost_matrix, 1e-9)
    
    # Heuristic: Prize / Cost
    # To make it more attractive for high prizes, maybe power?
    # Let's try simple ratio first.
    heuristic = prizes[None, :] / cost_matrix
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
