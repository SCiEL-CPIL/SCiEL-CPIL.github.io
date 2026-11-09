# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _repulsive_forces(points, power=10):
    """
    Calculate repulsive forces between all pairs.
    Force magnitude is proportional to 1/d^power.
    This strongly pushes apart points that are too close, 
    which directly maximizes d_min.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Compute pairwise distances
    dists = pdist(points)
    
    # Reshape to matrix for easier indexing
    # pdist returns condensed form. We need to map back to (n, n) or just iterate.
    # For n=14, iterating is fine, but vectorized is better.
    # Let's use a vectorized approach with broadcasting for clarity and speed (n=14 is tiny)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dists_matrix[~np.isfinite(dists_matrix)] = 1e-6
    dists_matrix[np.diag_indices_from(dists_matrix)] = 1.0 # Ignore self
    
    # Force vector from j to i is (p_i - p_j) / ||p_i - p_j||^3 * strength
    # We want to push them apart.
    # F_i = sum_j (p_i - p_j) / ||p_i - p_j||^(power+1)
    
    # Normalize diff to get direction, then scale by 1/d^power
    # diff[i,j] = p_i - p_j
    # norm = dists_matrix[i,j]
    # dir = diff / norm
    # force = dir / norm^power = diff / norm^(power+1)
    
    norm_p = dists_matrix ** (power + 1)
    # Avoid division by zero again
    norm_p[norm_p == 0] = 1e-10
    
    force_mat = diff / norm_p[:, :, np.newaxis]
    
    # Sum forces over j (axis 1)
    forces = np.sum(force_mat, axis=1)
    
    return forces

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Strategy: Repulsive force optimization on the unit sphere.
    # Since the metric is scale-invariant, we can constrain points to the unit sphere.
    # We use a repulsive force law that is very steep (high power) to maximize d_min.
    
    n = 14
    
    # Initial guess: Fibonacci sphere for good initial spread
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    # We will run multiple iterations of this optimization from different starts
    # to avoid local minima.
    
    num_starts = 10
    max_iters = 5000
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Different initializations
        if start_idx == 0:
            points = fibonacci_sphere(n)
        elif start_idx == 1:
            # Random points on sphere
            points = np.random.randn(n, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Perturbed fibonacci sphere
            points = fibonacci_sphere(n)
            points += np.random.randn(n, 3) * 0.1
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimization loop
        current_points = points.copy()
        lr = 0.01 # Learning rate
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Calculate forces
            forces = _repulsive_forces(current_points, power=10)
            
            # Update points
            current_points += lr * forces
            
            # Project back to unit sphere (constrain to surface)
            norms = np.linalg.norm(current_points, axis=1, keepdims=True)
            current_points /= norms
            
            # Also center them (remove mean) to keep them balanced
            current_points -= np.mean(current_points, axis=0, keepdims=True)
            norms = np.linalg.norm(current_points, axis=1, keepdims=True)
            current_points /= norms
            
            # Adaptive learning rate: decrease if oscillating or to converge
            if i % 100 == 0:
                lr *= 0.95
                if lr < 1e-5:
                    lr = 1e-5
        
        # Evaluate final score
        s = _score(current_points)
        if s > best_score:
            best_score = s
            best_points = current_points.copy()
            
            # If we find a very good score, we can try to polish it with more iterations
            # but for now, just keep the best.
            
    # Final polish on the best found
    if best_points is not None:
        # Run a few more iterations with very small step size to settle
        lr = 1e-5
        for i in range(1000):
            if time.time() > deadline - 1:
                break
            forces = _repulsive_forces(best_points, power=10)
            best_points += lr * forces
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            best_points /= norms
            best_points -= np.mean(best_points, axis=0, keepdims=True)
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            best_points /= norms
        
        # Update score
        s = _score(best_points)
        if s > best_score:
            best_score = s

    if best_points is None:
        # Fallback
        best_points = fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
