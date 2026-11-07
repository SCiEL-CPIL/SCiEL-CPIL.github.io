"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses a smooth k-Nearest Neighbors filtered inverse distance weighting.
    Instead of a hard binary mask, it uses a Gaussian decay based on the local
    k-th nearest neighbor distance, acting as a smooth surrogate for the 
    non-smooth k-NN selection.
    """
    N = distance_matrix.shape[0]
    
    # Work with a copy to avoid modifying input
    dist = distance_matrix.astype(np.float64).copy()
    
    # Ensure diagonal is 0 for distance calculations, but we handle it later
    np.fill_diagonal(dist, 0.0)
    
    # Define k for nearest neighbors
    # Using k=8 provides a slightly larger neighborhood than k=6, 
    # reducing the risk of excluding optimal tour edges.
    k = 8
    
    # Find the distance to the k-th nearest neighbor for each node
    # We need the k-th smallest value in each row.
    # argpartition is fast, but we need the value, not just index.
    # We can sort or use partition.
    # dist_sorted = np.sort(dist, axis=1)
    # D_k = dist_sorted[:, k]
    
    # More efficient: use argpartition to get indices, then pick the k-th one
    # Note: argpartition places the k-th smallest element at index k, 
    # but the elements before it are not necessarily sorted.
    # However, we just need the value at the k-th position in the partitioned array
    # which corresponds to the k-th smallest value.
    
    # To be safe and simple for N=50, sorting is fast enough and less error-prone with edge cases
    dist_sorted = np.sort(dist, axis=1)
    
    # The k-th nearest neighbor is at index k (0-indexed, so k=1 is 1st NN, k=8 is 8th NN)
    # If k > N-1, clamp it.
    k_idx = min(k, N - 1)
    D_k = dist_sorted[:, k_idx]
    
    # Ensure D_k is not zero to avoid division by zero in the exponent
    # If points are identical, D_k might be 0.
    D_k_safe = np.where(D_k > 1e-10, D_k, 1.0)
    
    # Construct the symmetric local scale matrix Sigma
    # Sigma[i, j] = max(D_k[i], D_k[j])
    # This ensures that if either node has a sparse neighborhood, the scale is larger,
    # allowing for longer edges to be considered without being penalized too harshly.
    Sigma = np.maximum(D_k_safe[:, np.newaxis], D_k_safe[np.newaxis, :])
    
    # Calculate the heuristic
    # Base heuristic: 1 / distance
    # Smooth filter: exp(-(d / Sigma)^2)
    
    # Avoid division by zero on diagonal
    safe_dist = np.where(dist > 1e-10, dist, 1.0)
    
    # The exponential term
    # We use squared distance in the exponent for a sharper falloff, 
    # but scaled by the local neighborhood size.
    ratio = dist / Sigma
    smooth_filter = np.exp(-ratio ** 2)
    
    # Combine
    heuristic = (1.0 / safe_dist) * smooth_filter
    
    # Set diagonal to 0 (or very small) as ants don't move to themselves
    np.fill_diagonal(heuristic, 0.0)
    
    # The task says "Heuristic entries that are not positive become 1e-9".
    # Our heuristic is always positive for off-diagonal entries.
    # However, if d is very large, exp(-ratio^2) might underflow to 0.
    # Let's ensure all off-diagonal entries are at least 1e-9.
    # This allows the ACO to technically visit any node, but with very low probability
    # for distant nodes, effectively acting as a soft constraint.
    
    # Mask for off-diagonal
    mask = ~np.eye(N, dtype=bool)
    heuristic[mask] = np.maximum(heuristic[mask], 1e-9)
    
    # Final cleanup for any non-finite values
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
