# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses Fibonacci sphere initialization and repulsion-based optimization on the unit sphere.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0  # 600 seconds limit, keep margin
    
    # 1. Initialize points on a unit sphere using Fibonacci sphere distribution
    # This provides a nearly uniform distribution, a good starting point.
    points = np.zeros((n, d))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        # Spherical coordinates
        # theta (azimuth)
        theta = 2 * np.pi * i / golden_ratio
        # phi (polar)
        cos_phi = 1 - (2 * i + 1) / n
        sin_phi = np.sqrt(1 - cos_phi**2)
        
        x = sin_phi * np.cos(theta)
        y = sin_phi * np.sin(theta)
        z = cos_phi
        
        points[i] = [x, y, z]
    
    # Ensure unit length (they should be, but just in case)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # 2. Optimize using repulsion forces (Thomson problem approach)
    # We minimize the potential energy U = sum_{i<j} 1/r_ij^2
    # Gradient of U wrt p_i is -2 * sum_{j!=i} (p_i - p_j) / r_ij^4
    # We want to minimize U, so we move in direction -grad U = 2 * sum_{j!=i} (p_i - p_j) / r_ij^4
    # This force pushes points apart.
    
    max_iters = 5000
    step_size = 0.05
    
    for it in range(max_iters):
        # Check time
        if time.time() > deadline - 10.0:
            break
            
        forces = np.zeros_like(points)
        
        for i in range(n):
            # Calculate repulsion force on point i
            # Force from j on i is proportional to (p_i - p_j) / ||p_i - p_j||^3 (for 1/r potential)
            # For 1/r^2 potential, force is (p_i - p_j) / ||p_i - p_j||^4 * 2
            # Let's use the gradient of 1/r^2 which is -2(r_ij^-4)(p_i-p_j)
            # So we add 2(p_i-p_j)/r_ij^4 to the force vector for minimization descent.
            
            diff = points[i] - points  # (n, 3)
            dists = np.linalg.norm(diff, axis=1)  # (n,)
            dists[i] = 1.0  # Avoid division by zero, we skip i anyway
            
            # Weight: 2 / dist^4
            weights = np.zeros(n)
            mask = dists > 1e-10
            weights[mask] = 2.0 / (dists[mask] ** 4)
            weights[i] = 0.0
            
            # Force vector
            force_i = np.sum(diff * weights[:, np.newaxis], axis=0)
            
            # Project force onto tangent plane at p_i
            # Tangent plane normal is p_i (unit vector)
            # proj = force - (force . p_i) * p_i
            dot = np.dot(force_i, points[i])
            force_i = force_i - dot * points[i]
            
            forces[i] = force_i
        
        # Update points
        points = points + step_size * forces
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Adaptive step size reduction to converge
        if it % 1000 == 0 and it > 0:
            step_size *= 0.5
            
    # Final check and cleanup
    # Ensure no NaNs or Infs
    if not np.all(np.isfinite(points)):
        # Fallback to random if something went wrong (should not happen)
        np.random.seed(42)
        points = np.random.randn(n, d)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

    return points

# EVOLVE-BLOCK-END
