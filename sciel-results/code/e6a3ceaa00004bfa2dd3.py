"""
Minimize logarithmic energy of 100 points on the unit sphere.

Strategy:
1. Start from a Fibonacci spiral (good initial spread).
2. Use gradient-based optimization with multiple restarts.
3. The energy is E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
4. Gradient of E w.r.t. point i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
   (derived from d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j)/|x_i - x_j|^2)
5. Constrain points to the sphere via projection after each step.
6. Use adaptive learning rate and multiple restarts with perturbations.
"""

import numpy as np
import time
import math

N = 100

def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    # points: (N, 3)
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    # Avoid log(0) on diagonal
    np.fill_diagonal(dist, 1.0)
    # Sum over i < j
    triu_idx = np.triu_indices(N, k=1)
    d = dist[triu_idx]
    # Use math.fsum for precision as specified
    terms = [-math.log(d[k]) for k in range(len(d))]
    return math.fsum(terms)

def _gradient(points):
    """
    Compute gradient of energy E = -sum_{i<j} log(|x_i - x_j|) w.r.t. each point.
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    N = len(points)
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-12:
                continue
            grad[i] += d / (dist * dist)
    return grad

def _gradient_fast(points):
    """Vectorized gradient computation."""
    N = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dist, 1.0)
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n=N):
    """Generate random points on the unit sphere."""
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize_single(points, max_iter=5000, lr_init=0.001, lr_decay=0.999, 
                     min_lr=1e-6, tol=1e-12, deadline=None):
    """
    Gradient descent with projection to sphere.
    Returns (points, energy).
    """
    points = _project_to_sphere(points.copy())
    E = _energy(points)
    lr = lr_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_fast(points)
        
        # Adaptive step: scale by norm of gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        
        # Take step
        points_new = points - lr * grad
        
        # Project to sphere
        points_new = _project_to_sphere(points_new)
        
        # Compute new energy
        E_new = _energy(points_new)
        
        if E_new < E:
            points = points_new
            E = E_new
            # Slightly increase lr if making good progress
            lr = min(lr * 1.02, 0.01)
        else:
            # Decrease learning rate
            lr *= lr_decay
        
        if lr < min_lr:
            break
        
        # Check convergence
        if it > 0 and abs(E_prev - E) < tol:
            break
        E_prev = E
    
    return points, E

def _polish(points, max_iter=10000, lr_init=0.0001, lr_decay=0.9999,
            min_lr=1e-8, tol=1e-14, deadline=None):
    """High-precision polish with very small steps."""
    points = _project_to_sphere(points.copy())
    E = _energy(points)
    lr = lr_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_fast(points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-16:
            break
        
        points_new = points - lr * grad
        points_new = _project_to_sphere(points_new)
        E_new = _energy(points_new)
        
        if E_new < E:
            points = points_new
            E = E_new
            lr = min(lr * 1.005, 0.001)
        else:
            lr *= lr_decay
        
        if lr < min_lr:
            break
    
    return points, E

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    best_points = _fibonacci_spiral(N)
    best_energy = _energy(best_points)
    
    # Phase 1: Gradient descent from Fibonacci spiral
    pts, E = _optimize_single(best_points, max_iter=3000, lr_init=0.002, 
                              lr_decay=0.999, min_lr=1e-6, deadline=deadline)
    if E < best_energy:
        best_energy = E
        best_points = pts
    
    # Phase 2: Try several random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate perturbed starting points
        if restart == 0:
            start_pts = _fibonacci_spiral(N)
        elif restart % 3 == 1:
            # Small perturbation of current best
            noise = np.random.randn(N, 3) * 0.1
            start_pts = _project_to_sphere(best_points + noise)
        elif restart % 3 == 2:
            # Larger perturbation
            noise = np.random.randn(N, 3) * 0.3
            start_pts = _project_to_sphere(best_points + noise)
        else:
            # Random
            start_pts = _random_points(N)
        
        pts, E = _optimize_single(start_pts, max_iter=2000, lr_init=0.001,
                                  lr_decay=0.999, min_lr=1e-6, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Phase 3: Polish the best result
    if time.time() < deadline - 5.0:
        pts, E = _polish(best_points, max_iter=5000, lr_init=0.0001,
                         lr_decay=0.9999, min_lr=1e-8, deadline=deadline)
        if E < best_energy:
            best_energy = E
            best_points = pts
    
    # Final verification: ensure all points are on unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Separate them
                best_points[j] = best_points[i] + np.random.randn(3) * 0.01
                best_points = _project_to_sphere(best_points)
    
    # Final energy check
    final_E = _energy(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
