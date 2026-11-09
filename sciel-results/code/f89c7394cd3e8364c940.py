# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    deadline = time.time() + 600.0
    
    # 1. Initialize points on a Fibonacci sphere for good initial spread
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * np.arange(n)
    
    points = np.column_stack((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ))
    
    # 2. Optimize using Coulomb repulsion / Tammes problem approach
    # We want to maximize the minimum pairwise distance.
    # This is equivalent to minimizing the sum of inverse powers of distances (Coulomb energy)
    # or using a smooth approximation of the min distance.
    
    def compute_scores(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0, d_min, d_max
        return (d_min / d_max) ** 2, d_min, d_max
    
    current_score, _, _ = compute_scores(points)
    best_points = points.copy()
    best_score = current_score
    
    # Gradient ascent on the smooth surrogate: log(d_min) - log(d_max)
    # Or simply maximize d_min / d_max.
    # Let's use a soft-min/soft-max approach for differentiability.
    # f = logsumexp(-dist^2 / T) approximates -log(d_min)
    # g = logsumexp(dist^2 / T) approximates log(d_max)
    # Maximize f - g ? No, maximize d_min/d_max is maximize log(d_min) - log(d_max).
    # log(d_min) = -logsumexp(-dist^2/T) as T->0
    # log(d_max) = logsumexp(dist^2/T) as T->0
    
    T = 0.1  # Temperature for soft min/max
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(points)
    
    def objective(pts):
        dists = pdist(pts)
        # Soft min: -logsumexp(-dists^2/T)
        soft_min = -np.log(np.sum(np.exp(-dists**2 / T)))
        # Soft max: logsumexp(dists^2/T)
        soft_max = np.log(np.sum(np.exp(dists**2 / T)))
        # We want to maximize soft_min - soft_max
        # But soft_min and soft_max are not directly log(d_min) and log(d_max).
        # Actually, log(d_min) = 0.5 * log(d_min^2).
        # Let's just maximize the ratio directly using gradients if possible,
        # or use a simple heuristic like repulsive forces.
        
        # Repulsive force: each point is pushed away from others.
        # Force on point i is sum over j != i of (p_i - p_j) / ||p_i - p_j||^3
        # This maximizes the minimum distance on a sphere (if we project back to sphere).
        
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < 1e-10:
                    continue
                dist = np.sqrt(dist_sq)
                # Coulomb repulsion
                forces[i] += diff / (dist_sq * dist)
        
        return forces, soft_min - soft_max
    
    # Iterative optimization
    for iteration in range(10000):
        if time.time() > deadline - 10:
            break
            
        forces, _ = objective(points)
        
        # Update points
        v = momentum * v + lr * forces
        points = points + v
        
        # Project back to sphere (radius 1) to maintain scale invariance
        # The ratio d_min/d_max is scale invariant, so we can fix the points on a sphere.
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        points = points / norms
        
        # Check score
        score, d_min, d_max = compute_scores(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr *= 0.95
            
    # Final polish: local search with smaller steps
    lr = 0.001
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
            
        forces, _ = objective(best_points)
        best_points = best_points + lr * forces
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        best_points = best_points / norms
        
        score, _, _ = compute_scores(best_points)
        if score < best_score:
            # Revert if score decreased
            best_points = best_points - lr * forces
            norms = np.linalg.norm(best_points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1e-10
            best_points = best_points / norms
        else:
            best_score = score
            
    return best_points


# EVOLVE-BLOCK-END
