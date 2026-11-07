import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    # Compute savings: d(i,0) + d(0,j) - d(i,j)
    # d(i,0) is distance_matrix[i, 0]
    # d(0,j) is distance_matrix[0, j]
    # d(i,j) is distance_matrix[i, j]
    
    # Vectorized calculation of savings matrix
    # savings[i, j] = d[i, 0] + d[0, j] - d[i, j]
    d_i_0 = distance_matrix[:, 0] # shape (n,)
    d_0_j = distance_matrix[0, :] # shape (n,)
    savings = d_i_0[:, np.newaxis] + d_0_j[np.newaxis, :] - distance_matrix
    
    # We only care about savings for i != j and i != 0, j != 0?
    # The ACO runs from depot to customers. 
    # The heuristic is used for transitions i -> j.
    # If i is depot (0), savings[0, j] = d(0,0) + d(0,j) - d(0,j) = 0.
    # So savings from depot are 0. This is correct, no "saving" is made by leaving the depot.
    # If j is depot (0), savings[i, 0] = d(i,0) + d(0,0) - d(i,0) = 0.
    # Returning to depot is not a "saving" edge in the same way.
    
    # Identify valid savings (positive and not involving depot in a trivial way?)
    # Actually, just use the positive savings.
    max_saving = np.max(savings)
    
    if max_saving > 0:
        # Boost factor
        # boost[i, j] = 1 + savings[i, j] / max_saving if savings[i, j] > 0 else 1
        # We can use np.where
        boost = np.where(savings > 0, 1 + savings / max_saving, 1.0)
    else:
        boost = np.ones((n, n))
        
    # Base heuristic: 1 / distance
    # Avoid division by zero
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Combine
    heuristic = base_heuristic * boost
    
    # Ensure no zeros or negatives
    # The prompt says "Heuristic entries that are not positive become 1e-9"
    # Our formula produces positive values.
    # But just in case:
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
