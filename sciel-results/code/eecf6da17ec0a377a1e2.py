import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a heuristic matrix for Ant Colony Optimization on CVRP.
    Uses a combination of inverse distance (proximity) and Clarke-Wright savings
    to guide ants toward edges that consolidate routes effectively.
    """
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Initialize heuristic with inverse distance to the power of 1.5
    # This provides a strong preference for local, short edges.
    # We add eps to avoid division by zero for self-loops (though ants don't pick self-loops, 
    # the matrix is full).
    D = distance_matrix + eps
    H = 1.0 / (D ** 1.5)
    
    # Compute Clarke-Wright savings modulation for customer-customer pairs
    # Savings S(i, j) = d(0, i) + d(0, j) - d(i, j)
    # We modulate the heuristic by a factor derived from savings.
    
    # Get distances from depot (row 0)
    d_depot = distance_matrix[0, :]
    
    # We only modify entries where both i and j are customers (index > 0)
    for i in range(1, n):
        for j in range(i + 1, n):
            # Calculate savings
            # d(0, i) + d(0, j) - d(i, j)
            savings = d_depot[i] + d_depot[j] - distance_matrix[i, j]
            
            # Normalize savings to a factor between 0 and 1 (approx)
            # Base cost of separate trips
            base_cost = d_depot[i] + d_depot[j]
            
            if base_cost > 0:
                ratio = savings / base_cost
                # Ensure ratio is non-negative (should be for Euclidean, but safe)
                if ratio < 0:
                    ratio = 0.0
            else:
                ratio = 0.0
            
            # Modulation factor: 
            # If savings are high (ratio close to 1), factor is higher.
            # If savings are low (ratio close to 0), factor is lower.
            # We map ratio [0, 1] to a multiplier [0.5, 1.5] to avoid zeroing out edges completely
            # but still strongly prefer high-savings edges.
            # Alternatively, just use (1 + ratio) which maps [0,1] to [1,2].
            # Let's use (1 + ratio) to boost good merges by up to 2x.
            factor = 1.0 + ratio
            
            # Apply to symmetric entries
            H[i, j] *= factor
            H[j, i] *= factor
            
    # For edges involving the depot, we keep the pure distance-based heuristic.
    # This ensures that starting/ending a route is still guided by proximity to the depot.
    
    # Ensure all values are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
