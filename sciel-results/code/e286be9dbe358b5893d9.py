"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def fibonacci_spiral(n):
    """Generate points on unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def energy_and_grad(points):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Compute pairwise distances
    # points: (n, 3)
    # dist[i,j] = |x_i - x_j|
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Note: d/dx_i log(1/|x_i-x_j|) = d/dx_i (-log|x_i-x_j|) = -(x_i-x_j)/|x_i-x_j|^2
    # So gradient of energy w.r.t. x_i is -sum_{j!=i} (x_i-x_j)/|x_i-x_j|^2
    # But we want to minimize energy, so we need the negative gradient for descent
    # Actually, let's be careful: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j!=i, j>i} d(log d_ij)/dx_i - sum_{j<i} d(log d_ij)/dx_i
    # d(log d_ij)/dx_i = (x_i - x_j) / |x_i - x_j|^2
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    
    # Compute gradient
    # For each i, sum over j != i of (x_i - x_j) / dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    # grad_i = -sum_j (x_i - x_j) / dist_sq[i,j] for j != i
    inv_dist_sq = 1.0 / dist_sq  # (n, n), diagonal is 1.0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = -sum_j inv_dist_sq[i,j] * diff[i,j]
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)  # (n, 3)
    
    return energy, grad

def tangent_project(points, grad):
    """Project gradient onto tangent space of sphere at each point."""
    # For each point x_i, tangent space is orthogonal to x_i
    # proj(v) = v - (v . x_i) x_i
    dot = np.sum(grad * points, axis=1, keepdims=True)  # (n, 1)
    proj_grad = grad - dot * points  # (n, 3)
    return proj_grad

def optimize_from_start(points, time_limit, max_iter=5000):
    """
    Optimize points on sphere using L-BFGS-B in tangent space.
    We use a parameterization where we move in tangent directions.
    """
    n = len(points)
    start_time = time.time()
    
    # Flatten points for optimizer
    x0 = points.flatten()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        # Normalize to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x = x / norms
        
        e, g = energy_and_grad(x)
        # Project gradient to tangent space
        pg = tangent_project(x, g)
        return e, pg.flatten()
    
    # Use L-BFGS-B with analytical gradient
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-15}
        )
    except Exception:
        return points, None
    
    # Extract optimized points
    x_opt = result.x.reshape(n, 3)
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x_opt = x_opt / norms
    
    elapsed = time.time() - start_time
    return x_opt, result

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with Fibonacci spiral
    best_points = fibonacci_spiral(N)
    best_energy, _ = energy_and_grad(best_points)
    
    # Try multiple starts with different perturbations
    rng = np.random.RandomState(42)
    
    # Define perturbation strategies
    num_starts = 8
    start_configs = []
    
    # Start 0: original Fibonacci
    start_configs.append(best_points.copy())
    
    # Starts 1-7: various perturbations
    for k in range(1, num_starts):
        if time.time() > deadline:
            break
        p = best_points.copy()
        # Add random perturbation in tangent space
        for i in range(N):
            # Random tangent vector
            v = rng.randn(3)
            v = v - np.dot(v, p[i]) * p[i]
            v = v / (np.linalg.norm(v) + 1e-12)
            # Small perturbation
            p[i] = p[i] + 0.05 * v
            # Renormalize
            p[i] = p[i] / np.linalg.norm(p[i])
        start_configs.append(p)
    
    # Optimize each start
    for idx, start_pts in enumerate(start_configs):
        if time.time() > deadline:
            break
        
        # Reduce max_iter for later starts to ensure we finish in time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        max_iter = min(5000, int(remaining * 100))
        if max_iter < 100:
            break
        
        try:
            opt_pts, result = optimize_from_start(start_pts, deadline, max_iter=max_iter)
            if opt_pts is not None:
                e, _ = energy_and_grad(opt_pts)
                if e < best_energy:
                    best_energy = e
                    best_points = opt_pts.copy()
        except Exception:
            continue
    
    # Final high-precision polish if time remains
    if time.time() < deadline - 10:
        try:
            opt_pts, result = optimize_from_start(best_points, deadline, max_iter=10000)
            if opt_pts is not None:
                e, _ = energy_and_grad(opt_pts)
                if e < best_energy:
                    best_energy = e
                    best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Ensure output is valid
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / np.maximum(norms, 1e-12)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[j] += 1e-8 * best_points[i]
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
