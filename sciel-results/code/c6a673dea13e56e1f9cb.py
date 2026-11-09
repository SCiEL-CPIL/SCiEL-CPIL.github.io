import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    # distance_matrix is n x n
    # demands is n, capacity is float
    
    # Calculate savings matrix
    # S_ij = d_i0 + d_0j - d_ij
    # d_i0 is distance_matrix[i, 0]
    # d_0j is distance_matrix[0, j]
    
    # Vectorized calculation
    # d_i0 is shape (n, 1)
    # d_0j is shape (1, n)
    d_i0 = distance_matrix[:, 0:1]
    d_0j = distance_matrix[0:1, :]
    
    savings = d_i0 + d_0j - distance_matrix
    
    # Ensure non-negative (should be for Euclidean, but just in case)
    savings = np.maximum(savings, 0)
    
    # Heuristic: (Savings + eps) / (Distance + eps)
    # This favors edges that are short and have high savings.
    eps = 1e-9
    heuristic = (savings + eps) / (distance_matrix + eps)
    
    # The problem statement says: "Heuristic entries that are not positive become 1e-9"
    # Our calculation yields positive values.
    # We should ensure the shape is correct.
    
    # One potential issue: The diagonal d_ii = 0. Savings_ii = d_i0 + d_0i - 0 = 2*d_i0.
    # Heuristic_ii = (2*d_i0) / eps -> Huge.
    # But ants don't pick visited nodes, and i != j in the transition logic usually?
    # "picks an unvisited customer". So diagonal doesn't matter for selection, 
    # but it's part of the matrix.
    # However, if the ACO implementation checks `i != j`, it's fine.
    # If it doesn't, a huge diagonal might cause issues if not masked.
    # Standard ACO masks visited nodes.
    
    # Let's double check if we should dampen the diagonal.
    # Usually, heuristics are defined for i != j.
    # We can set diagonal to 1e-9 to be safe, or 0.
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic
