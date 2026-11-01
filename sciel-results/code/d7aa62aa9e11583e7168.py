"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # We only want upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Clamp to avoid log of very small numbers
    d = np.maximum(d, 1e-15)
    # Energy is sum of log(1/d) = -sum log(d)
    # Use math.fsum for precision as per task description
    log_vals = np.log(d)
    return -math.fsum(list(log_vals))

def _gradient_log_energy(points):
    """Compute gradient of the logarithmic energy with respect to points.
    
    Energy E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Vectorized computation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient term: (x_i - x_j) / |x_i - x_j|^2
    grad_terms = diff / dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    grad = np.sum(grad_terms, axis=1)
    return grad

def _exponential_map(points, velocities, step_size):
    """Move points on the sphere using the exponential map.
    x_new = cos(step*|v|) * x + sin(step*|v|) * (v/|v|)
    """
    n = len(points)
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    # Avoid division by zero
    v_norms_safe = np.where(v_norms < 1e-12, 1e-12, v_norms)
    
    cos_term = np.cos(step_size * v_norms_safe)
    sin_term = np.sin(step_size * v_norms_safe) / v_norms_safe
    
    new_points = cos_term * points + sin_term * velocities
    return new_points

def _optimize_single(points, max_iters=5000, tol=1e-12):
    """Optimize a single configuration using gradient descent with backtracking line search."""
    current_energy = _log_energy(points)
    lr = 0.05
    momentum = 0.9
    velocity = np.zeros_like(points)
    
    for it in range(max_iters):
        grad = _gradient_log_energy(points)
        
        # Check convergence based on gradient norm
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad
        
        # Backtracking line search
        alpha = 1.0
        improved = False
        for _ in range(20):
            new_points = _exponential_map(points, velocity * alpha, 1.0)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_energy = _log_energy(new_points)
            
            if new_energy < current_energy:
                points = new_points
                current_energy = new_energy
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            # If we couldn't improve even with very small steps, break
            break
            
        # Decay learning rate slowly
        lr *= 0.999
        if lr < 1e-8:
            break
            
    return points, current_energy

def construct():
    # Generate initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    base_points /= np.linalg.norm(base_points, axis=1, keepdims=True)
    
    best_points = None
    best_energy = float('inf')
    
    # Try multiple starting points to avoid local minima
    # 1. The base Fibonacci spiral
    # 2. A few random perturbations of the Fibonacci spiral
    
    starts = [base_points.copy()]
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Add small random noise
        noise = rng.normal(0, 0.01, base_points.shape)
        perturbed = base_points + noise
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        starts.append(perturbed)
        
    for start in starts:
        opt_points, energy = _optimize_single(start)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points
            
    # Final polish with very small steps
    if best_points is not None:
        best_points, best_energy = _optimize_single(best_points, max_iters=1000, tol=1e-14)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        
        # Check for close points
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_points[i] - best_points[j])
                if d < min_dist:
                    min_dist = d
                    
        if min_dist < 1e-9:
            # Perturb if necessary
            points = best_points + rng.normal(0, 1e-6, best_points.shape)
            best_points, best_energy = _optimize_single(points, max_iters=1000)
            best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
            
    return best_points
# EVOLVE-BLOCK-END
