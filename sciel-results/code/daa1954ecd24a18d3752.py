import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix that combines inverse distance with a demand-aware penalty.
    
    The heuristic encourages:
    1. Short distances (inverse distance)
    2. Avoiding routes where adding a customer would leave little remaining capacity
       (which could force a return to depot sooner than optimal)
    
    H_ij = 1 / (d(i, j) + epsilon) * (1 + demand_j / capacity)^alpha
    where alpha is a small positive exponent to gently prefer lower-demand customers
    when distances are comparable, but this is subtle. Actually, let's think differently.
    
    A better approach: Use inverse distance but also consider that going to a customer
    with high demand early might be worse if it leaves little room. However, the ACO
    algorithm already handles capacity constraints by only allowing feasible moves.
    
    Let's try: H_ij = 1 / (d(i, j) + epsilon)^beta where beta > 1 to make the heuristic
    more sensitive to distance differences, helping ants distinguish between good and bad
    moves more sharply.
    """
    eps = 1e-9
    # Use a power > 1 to amplify differences in distances
    beta = 1.5
    heuristic = 1.0 / (distance_matrix + eps) ** beta
    
    # Ensure no non-finite values and positivity
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
