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
    v_norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    # Avoid division by zero
    v_norms_safe = np.where(v_norms < 1e-12, 1e-12, v_norms)
    
    cos_term = np.cos(step_size * v_norms_safe)
    sin_term = np.sin(step_size * v_norms_safe) / v_norms_safe
    
    new_points = cos_term * points + sin_term * velocities
    return new_points

def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _random_points(n, rng):
    """Generate random points on sphere."""
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _optimize_single(points, max_iters=3000, initial_lr=0.02, min_lr=1e-8, time_limit=110.0):
    """Optimize a single configuration with adaptive gradient descent."""
    start_time = time.time()
    current_energy = _log_energy(points)
    
    lr = initial_lr
    momentum = 0.85
    velocity = np.zeros_like(points)
    
    best_points = points.copy()
    best_energy = current_energy
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        grad = _gradient_log_energy(points)
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad
        
        # Move points
        new_points = _exponential_map(points, velocity, 1.0)
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_energy = _log_energy(new_points)
        
        # If energy increases, reject and reduce step size
        if new_energy > current_energy:
            lr *= 0.5
            if lr < min_lr:
                break
            continue
        else:
            points = new_points
            current_energy = new_energy
            
            # Track best
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
            
            # Decay learning rate
            lr *= 0.998
            
            # Check convergence
            if lr < min_lr:
                break
    
    # Final refinement
    lr = 1e-7
    for it in range(50):
        if time.time() - start_time > time_limit:
            break
        grad = _gradient_log_energy(points)
        new_points = points - lr * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        new_energy = _log_energy(new_points)
        if new_energy < current_energy:
            points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = points.copy()
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return best_points, best_energy

def construct():
    # Try multiple initial configurations and keep the best
    best_points = None
    best_energy = float('inf')
    
    # Initial configurations to try
    configs = []
    
    # Fibonacci spiral
    configs.append(_fibonacci_spiral(N))
    
    # Random configurations with different seeds
    rng = np.random.default_rng(42)
    for seed in range(5):
        configs.append(_random_points(N, rng))
    
    # Slightly perturbed Fibonacci
    rng2 = np.random.default_rng(123)
    fib = _fibonacci_spiral(N)
    fib_perturbed = fib + rng2.normal(0, 0.01, fib.shape)
    fib_perturbed /= np.linalg.norm(fib_perturbed, axis=1, keepdims=True)
    configs.append(fib_perturbed)
    
    # Optimize each configuration
    for idx, init_points in enumerate(configs):
        points, energy = _optimize_single(init_points)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    # Final normalization
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
        rng = np.random.default_rng(42)
        best_points = best_points + rng.normal(0, 1e-6, best_points.shape)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
