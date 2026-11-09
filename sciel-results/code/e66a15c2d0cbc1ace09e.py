import numpy as np

def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Create matrix of sums
    sums = demands[:, None] + demands[None, :]
    
    # Remaining space if item j is placed after item i
    # rem = capacity - (d_i + d_j)
    rem = capacity - sums
    
    # Mask for valid placements
    valid = rem >= 0
    
    # Initialize with a small value for invalid
    H = np.full((n, n), 1e-9, dtype=float)
    
    # Calculate heuristic for valid placements
    # Heuristic: Bias towards larger items (d_j) and tighter fits (smaller rem)
    # H_ij = d_j / (rem_ij + 1)
    
    # We only calculate for valid entries
    # rem[valid] is the array of remainders
    # demands[None, :][valid] gets the d_j values? 
    # Let's be careful with indexing.
    
    # demands is shape (n,)
    # demands[None, :] is shape (1, n) -> broadcast to (n, n)
    # d_j values are in columns.
    # So we need to index the columns correctly.
    
    # Let's create a matrix of d_j values
    d_j_matrix = np.tile(demands[None, :], (n, 1))
    
    # Now apply to valid entries
    H[valid] = d_j_matrix[valid] / (rem[valid] + 1.0)
    
    # Ensure no negative or zero values slip through (though formula is positive)
    # And ensure non-finite are handled (though inputs are integers/floats 20-99)
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
