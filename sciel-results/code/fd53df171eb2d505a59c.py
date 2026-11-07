"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    if n == 0:
        return np.array([])
        
    # K nearest neighbors parameter
    # Tuned to approximate Delaunay triangulation density (avg degree ~6) 
    # with a small buffer for robustness
    K = 8
    
    # 1. Determine the set of "allowed" edges using the Union of K-NN graphs
    # This restricts the search space to geometrically proximal pairs,
    # preventing long-range crossing edges that are typically suboptimal in VRP.
    allowed = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        # Get indices of the K nearest neighbors for node i
        # distance_matrix[i][i] is 0, so it will be the first element in argsort
        # We take indices 1 to K+1 to exclude self
        neighbors = np.argsort(distance_matrix[i])[1:K+1]
        
        # Mark edges i->j as allowed
        allowed[i, neighbors] = True
        # Enforce symmetry: if i considers j, j considers i
        allowed[neighbors, i] = True
        
    # 2. Construct the heuristic matrix
    # Initialize with a very small positive value (1e-9) for disallowed edges
    heuristic = np.full((n, n), 1e-9)
    
    # For allowed edges, use inverse distance as attractiveness
    # We add 1e-9 to the denominator to avoid division by zero (if coordinates are identical)
    # and to ensure finite values.
    # We apply this only to the allowed mask.
    
    # To avoid modifying the original distance_matrix, we create a safe version for calculation
    # Or we can just use boolean indexing which is safe.
    
    # Get the distances for the allowed edges
    # We need to be careful: distance_matrix[allowed] returns a 1D array of distances
    # We compute 1/(d + eps) for these
    safe_distances = distance_matrix[allowed] + 1e-9
    heuristic_values = 1.0 / safe_distances
    
    # Assign back to the matrix
    heuristic[allowed] = heuristic_values
    
    # Ensure the diagonal (self-loops) is set to 1e-9, as ants cannot move to themselves
    # The mask 'allowed' might have diagonal as False (since we excluded self in KNN), 
    # so it should already be 1e-9. But let's be explicit.
    np.fill_diagonal(heuristic, 1e-9)
    
    # Validation: Ensure all values are positive and finite
    # 1e-9 is positive.
    # 1/(d+1e-9) is positive and finite for finite d.
    # If d is inf, 1/inf is 0. 0 is not positive. 
    # The prompt says "Heuristic entries that are not positive become 1e-9".
    # So we should check for values <= 0 or non-finite and replace them.
    
    # Check for non-finite (NaN, Inf)
    # If distance_matrix contains Inf, safe_distances is Inf, heuristic_values is 0.
    # 0 is not positive.
    # So we need to handle this.
    
    # Replace non-positive or non-finite values with 1e-9
    # np.isfinite checks for NaN and Inf
    # heuristic > 0 checks for positive
    
    invalid_mask = ~(np.isfinite(heuristic) & (heuristic > 0))
    if np.any(invalid_mask):
        heuristic[invalid_mask] = 1e-9
        
    return heuristic
# EVOLVE-BLOCK-END
