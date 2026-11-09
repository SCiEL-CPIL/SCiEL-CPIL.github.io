import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Extract return costs to depot (node 0)
    # d_j0[j] is distance from j to 0
    d_j0 = distance_matrix[:, 0]
    
    # Construct denominator: distance from i to j + distance from j to 0
    # This represents the total distance budget consumed by this transition 
    # and the mandatory return leg from the new node.
    denom = distance_matrix + d_j0[None, :]
    
    # Numerator: prize of the destination node j
    num = prizes[None, :]
    
    # Calculate heuristic
    # We want to avoid division by zero. 
    # If denom is 0, it means i=j and j=0 (since d_00=0, d_00=0).
    # In that case, prize[0] is 0, so 0/0 is NaN.
    # We handle this by using a safe denominator.
    safe_denom = np.where(denom > 1e-12, denom, 1.0)
    
    eta = num / safe_denom
    
    # Enforce the rule: non-positive values become 1e-9
    # This handles prizes=0 and any numerical issues
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
