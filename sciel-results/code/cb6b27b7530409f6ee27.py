"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/d_ij)"""
    n = len(pts)
    # Compute pairwise distances
    # Using broadcasting for efficiency
    # pts shape (n, 3)
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dist = np.linalg.norm(diff, axis=2)
    # But this is O(n^2) memory, for n=150 it's fine (150*150 = 22500)
    
    # More memory efficient: compute upper triangle only
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            energy += math.log(1.0 / d)
    return energy


def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    return _normalize(pts)


def _grad_log_energy(pts):
    """Compute gradient of log energy w.r.t. points on sphere (tangent space)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-15:
                continue
            # d/dx_i log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|^2
            # = -(x_i - x_j) / |x_i - x_j|^3
            grad[i] -= d / (dist ** 3)
    # Project gradient to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin of 5 seconds
    
    # Start with Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy = _log_energy(best_pts)
    
    # Try multiple random initializations and optimize
    n_restarts = 8
    max_iters = 500
    lr_initial = 0.01
    lr_decay = 0.995
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=0)
        else:
            pts = _random_sphere(N, seed=restart * 42 + 7)
        
        # Gradient descent with adaptive learning rate
        lr = lr_initial
        prev_energy = _log_energy(pts)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            grad = _grad_log_energy(pts)
            
            # Update points
            pts_new = pts - lr * grad
            pts_new = _project_to_sphere(pts_new)
            
            new_energy = _log_energy(pts_new)
            
            if new_energy < prev_energy:
                pts = pts_new
                prev_energy = new_energy
                # Slowly decay learning rate
                lr *= lr_decay
            else:
                # Reduce learning rate
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        # Check if this is better
        final_energy = _log_energy(pts)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = pts.copy()
    
    # Final cleanup: ensure points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci sphere
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
