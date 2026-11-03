"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Only take upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Filter out any zero distances (shouldn't happen, but safe)
    d = d[d > 1e-15]
    if len(d) == 0:
        return 0.0
    # Use math.fsum for accuracy as specified
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())


def _log_energy_fast(pts):
    """Fast logarithmic energy computation for optimization"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Set diagonal to 1
    np.fill_diagonal(dist_sq, 1.0)
    # Only take upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    # Filter out any very small distances
    mask = d > 1e-15
    d = d[mask]
    if len(d) == 0:
        return 0.0
    return np.sum(np.log(1.0 / d))


def _project_to_sphere(pts):
    """Project points to unit sphere"""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere"""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _spherical_gradient_descent(pts, lr=0.01, max_iter=2000, tol=1e-8, time_limit=10.0):
    """Vectorized Gradient descent on the sphere for logarithmic energy minimization"""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    prev_energy = _log_energy_fast(pts)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute gradient of energy
        # E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
        # dE/dxi = sum_{j!=i} (xi - xj) / |xi - xj|^2
        
        # Vectorized gradient computation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        # grad[i] = sum_j (pts[i] - pts[j]) / |pts[i] - pts[j]|^2
        # This is equivalent to sum_j diff[i,j] / dist_sq[i,j]
        # We need to mask the diagonal where dist_sq is 1.0 (which is not 0, so it's fine, but the term should be 0)
        # The term for j=i is 0 because diff[i,i] is 0.
        
        # inv_dist_sq = 1.0 / dist_sq
        # grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # To be safe and fast:
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update
        pts = pts - lr * grad_tangent
        pts = _project_to_sphere(pts)
        
        energy = _log_energy_fast(pts)
        
        if abs(prev_energy - energy) < tol:
            break
        prev_energy = energy
    
    return pts, _log_energy_fast(pts)


def _local_search(pts, n_restarts=10, time_limit=100):
    """Local search with multiple restarts"""
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Try different starting points
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Estimate remaining time for this restart
        # Leave some buffer for the next restart and final checks
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break
            
        # Allocate time for this restart, leaving a small buffer
        t_alloc = max(1.0, remaining - 1.0)
        
        # Generate different starting configurations
        if restart == 0:
            start_pts = pts.copy()
        elif restart == 1:
            start_pts = _fibonacci_sphere(len(pts), seed=restart)
        elif restart == 2:
            # Random uniform on sphere
            rng = np.random.RandomState(restart * 100)
            start_pts = rng.randn(len(pts), 3)
            start_pts = start_pts / np.linalg.norm(start_pts, axis=1, keepdims=True)
        else:
            # Perturbed version of best
            rng = np.random.RandomState(restart * 200)
            perturbation = rng.randn(len(pts), 3) * 0.1
            start_pts = best_pts + perturbation
            start_pts = _project_to_sphere(start_pts)
        
        # Run gradient descent with time limit
        result_pts, result_energy = _spherical_gradient_descent(start_pts, lr=0.01, max_iter=1000, time_limit=t_alloc)
        
        if result_energy < best_energy:
            best_energy = result_energy
            best_pts = result_pts
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin (evaluator kills at 120s/150s)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _log_energy_fast(pts)
    
    # Run local search
    # Use a time limit that leaves 10s for final processing
    local_time_limit = deadline - 10.0
    best_pts, best_energy = _local_search(pts, n_restarts=10, time_limit=local_time_limit)
    
    # Check time
    if time.time() - start_time > deadline:
        return _project_to_sphere(best_pts)
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Validate: check for very close points
    # Vectorized min distance check
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Points too close, use Fibonacci instead
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
