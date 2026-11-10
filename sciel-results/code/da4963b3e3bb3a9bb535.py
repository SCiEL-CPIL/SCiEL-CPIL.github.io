# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a repulsion-based optimization to find a configuration close to the optimal spherical code.
    """
    n = 14
    d = 3
    
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Initialize points on the surface of a unit sphere
    # Using random normal distribution and normalizing
    points = np.random.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Optimization parameters
    steps = 2000
    alpha = 0.01
    
    for _ in range(steps):
        # Calculate pairwise distances
        # points shape (n, d)
        # diff shape (n, n, d)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        # Squared distances
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        # Replace 0 with a small number to avoid inf
        np.fill_diagonal(dist_sq, 1.0) # Diagonal is 0, we don't want self-interaction
        # Actually, we need to compute forces excluding self
        # Force F_i = sum_{j != i} (P_i - P_j) / ||P_i - P_j||^2
        # Note: diff[i, j] = P_i - P_j
        # So we need dist_sq[i, j] in the denominator
        
        # Create a mask for i != j
        mask = ~np.eye(n, dtype=bool)
        
        # Compute inverse squared distances
        # We want 1/dist_sq, but 0 on diagonal
        inv_dist_sq = np.zeros_like(dist_sq)
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        
        # Forces: sum over j of (P_i - P_j) * inv_dist_sq[i, j]
        # diff is (n, n, d). inv_dist_sq is (n, n).
        # We need to multiply each vector diff[i, j] by inv_dist_sq[i, j]
        # Then sum over j (axis=1)
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Update points
        points += alpha * forces
        
        # Center the points to prevent drift
        points -= np.mean(points, axis=0, keepdims=True)
        
        # Normalize to unit diameter (max distance = 1)
        # Calculate max distance
        # dist_sq is already computed but points changed, so recompute or just scale based on current spread
        # A cheaper way: scale by current max norm or max dist.
        # Let's compute max dist properly to maintain the constraint strictly.
        # But this is O(N^2) every step. N=14 is small, so it's fine.
        
        # Recompute distances for scaling
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        max_dist = np.sqrt(np.max(dist_sq))
        
        if max_dist > 0:
            points = points / max_dist
    
    return points

# EVOLVE-BLOCK-END
