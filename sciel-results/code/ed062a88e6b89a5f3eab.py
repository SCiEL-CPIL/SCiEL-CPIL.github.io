"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO using Delaunay triangulation.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n, n)
    coordinates : np.ndarray
        Coordinates of shape (n, 2)
    demands : np.ndarray
        Demands of shape (n,)
    capacity : float
        Vehicle capacity
    
    Returns:
    --------
    np.ndarray
        Heuristic matrix of same shape as distance_matrix
    """
    n = len(demands)
    
    # Initialize heuristic with a small positive value for non-edges
    # This value should be significantly smaller than the min(1/d) of edges
    # but large enough to be positive.
    heuristic = np.full((n, n), 1e-5)
    np.fill_diagonal(heuristic, 0.0)
    
    # Get customer indices (excluding depot at index 0)
    customer_indices = np.arange(1, n)
    customer_coords = coordinates[customer_indices]
    
    try:
        # Perform Delaunay triangulation on customer coordinates
        tri = Delaunay(customer_coords)
        
        # Create a boolean mask for edges
        edge_mask = np.zeros((n, n), dtype=bool)
        
        # Process simplices to find edges
        for simplex in tri.simplices:
            # Map indices from customer space (0..n-2) to global space (1..n-1)
            i = simplex[0] + 1
            j = simplex[1] + 1
            k = simplex[2] + 1
            
            edge_mask[i, j] = True
            edge_mask[j, i] = True
            edge_mask[i, k] = True
            edge_mask[k, i] = True
            edge_mask[j, k] = True
            edge_mask[k, j] = True
        
        # Connect depot (0) to nearest customers to ensure connectivity
        depot_distances = distance_matrix[0, 1:n]
        k_nearest = min(10, n - 1)
        nearest_indices = np.argpartition(depot_distances, k_nearest)[:k_nearest] + 1
        for idx in nearest_indices:
            edge_mask[0, idx] = True
            edge_mask[idx, 0] = True
        
        # Calculate inverse distances for all pairs
        # Avoid division by zero
        safe_dist = np.where(distance_matrix > 0, distance_matrix, 1.0)
        inv_dist = 1.0 / safe_dist
        
        # Apply mask: use inv_dist for edges, keep 1e-5 for non-edges
        heuristic = np.where(edge_mask, inv_dist, heuristic)
        
        # Ensure diagonal is 0
        np.fill_diagonal(heuristic, 0.0)
        
    except Exception:
        # Fallback: use inverse distance for all edges if triangulation fails
        safe_dist = np.where(distance_matrix > 0, distance_matrix, 1.0)
        heuristic = 1.0 / safe_dist
        np.fill_diagonal(heuristic, 0.0)
        
        # Ensure positivity for non-diagonal entries
        heuristic[~np.eye(n, dtype=bool)] = np.maximum(heuristic[~np.eye(n, dtype=bool)], 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
