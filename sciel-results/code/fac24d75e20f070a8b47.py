"""Pack 300 equal circles in a unit disk using barrier method optimization."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _golden_spiral_init(n, radius=1.0):
    """
    Generate n points using golden angle spiral (sunflower model).
    This provides a rotationally symmetric, non-lattice starting point.
    """
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n, dtype=np.float64)
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _compute_min_radius(points):
    """
    Compute the maximum possible radius for the given points.
    """
    if len(points) == 0:
        return 0.0
    
    dists_to_center = np.linalg.norm(points, axis=1)
    boundary_min = np.min(1.0 - dists_to_center)
    
    if boundary_min < 0:
        return 0.0
    
    # Full pairwise distance matrix for N=300
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    
    pair_min = min_pair_dist / 2.0
    
    return min(boundary_min, pair_min)

def _barrier_optimize(points, max_iters=2000, time_limit=100.0):
    """
    Optimize using a logarithmic barrier method.
    Objective: maximize sum of log(distances to nearest neighbor and boundary)
    This encourages uniform spacing and keeps points feasible.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    # Adaptive step size
    lr = 0.01
    best_r = _compute_min_radius(points)
    best_points = points.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute distances
        dists_to_center = np.linalg.norm(points, axis=1)
        
        # Boundary barrier: log(1 - |c_i|) for each point
        # Clip to avoid log(0) or log(negative)
        boundary_margin = 1.0 - dists_to_center
        boundary_margin = np.maximum(boundary_margin, 1e-10)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # For each point, find its nearest neighbor distance
        # Use log of nearest neighbor distance as barrier
        # To make it more global, use sum of log of all pairwise distances
        # But that's O(n^2) per iteration - acceptable for n=300
        
        # Clip distances to avoid log(0)
        dist_matrix_clipped = np.maximum(dist_matrix, 1e-10)
        
        # Gradient of sum of log(distances)
        # d/dx_i sum_j log(|x_i - x_j|) = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
        # = sum_{j!=i} (x_i - x_j) / d_ij^2
        
        # Compute gradient efficiently
        grad = np.zeros_like(points)
        
        # For each pair, add contribution to gradient
        # (x_i - x_j) / d_ij^2 for i, and (x_j - x_i) / d_ij^2 for j
        for i in range(n):
            for j in range(i+1, n):
                d_vec = points[i] - points[j]
                d_sq = np.dot(d_vec, d_vec)
                if d_sq > 1e-20:
                    # Contribution to grad[i]: d_vec / d_sq
                    # Contribution to grad[j]: -d_vec / d_sq
                    coeff = 1.0 / d_sq
                    grad[i] += d_vec * coeff
                    grad[j] -= d_vec * coeff
        
        # Boundary gradient: d/dx_i log(1 - |x_i|) = -x_i / (|x_i| * (1 - |x_i|))
        for i in range(n):
            if dists_to_center[i] > 1e-10:
                grad[i] -= points[i] / (dists_to_center[i] * boundary_margin[i])
        
        # Normalize gradient to have consistent scale
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Adaptive step: reduce if we violate constraints
        new_points = points + lr * grad
        
        # Project back to unit disk
        new_dists = np.linalg.norm(new_points, axis=1)
        outside_mask = new_dists > 1.0
        if np.any(outside_mask):
            new_points[outside_mask] *= (1.0 / new_dists[outside_mask, np.newaxis])
        
        # Check if this improved the minimum radius
        new_r = _compute_min_radius(new_points)
        
        if new_r >= best_r - 1e-12:
            # Accept the step
            points = new_points
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
            # Increase learning rate slightly
            lr = min(lr * 1.05, 0.05)
        else:
            # Reject and reduce learning rate
            lr *= 0.8
            if lr < 0.001:
                break
        
        # Early termination if progress is negligible
        if it > 100 and (best_r - _compute_min_radius(points)) < 1e-8:
            break
    
    return best_points, best_r

def _simulated_annealing(points, max_iters=5000, time_limit=50.0):
    """
    Simulated annealing to escape local minima.
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    start_time = time.time()
    
    current_r = _compute_min_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Propose a random perturbation
        perturbation = np.random.normal(0, T, size=(n, 2))
        new_points = points + perturbation
        
        # Project to unit disk
        new_dists = np.linalg.norm(new_points, axis=1)
        outside_mask = new_dists > 1.0
        if np.any(outside_mask):
            new_points[outside_mask] *= (1.0 / new_dists[outside_mask, np.newaxis])
        
        new_r = _compute_min_radius(new_points)
        
        # Accept or reject
        if new_r >= current_r:
            points = new_points
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            # Boltzmann acceptance
            delta = current_r - new_r
            if delta < 1e-15:
                points = new_points
                current_r = new_r
            else:
                accept_prob = np.exp(-delta / T)
                if np.random.random() < accept_prob:
                    points = new_points
                    current_r = new_r
        
        T = max(T * cooling, T_min)
        
        # Early termination
        if T <= T_min * 1.1:
            break
    
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Initialize with golden spiral
    points = _golden_spiral_init(N)
    current_r = _compute_min_radius(points)
    
    best_points = points.copy()
    best_r = current_r
    
    # 2. Phase 1: Barrier optimization
    if time.time() < deadline - 50.0:
        opt_points, opt_r = _barrier_optimize(points, max_iters=3000, time_limit=40.0)
        if opt_r > best_r:
            best_points = opt_points
            best_r = opt_r
    
    # 3. Phase 2: Simulated annealing to escape local minima
    if time.time() < deadline - 30.0:
        sa_points, sa_r = _simulated_annealing(best_points, max_iters=10000, time_limit=30.0)
        if sa_r > best_r:
            best_points = sa_points
            best_r = sa_r
    
    # 4. Phase 3: Final barrier refinement
    if time.time() < deadline - 5.0:
        final_points, final_r = _barrier_optimize(best_points, max_iters=2000, time_limit=10.0)
        if final_r > best_r:
            best_points = final_points
            best_r = final_r
    
    return best_points
# EVOLVE-BLOCK-END
