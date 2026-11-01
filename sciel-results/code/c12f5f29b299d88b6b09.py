"""
Stronger algorithm for 100-point logarithmic energy on the unit sphere.
Uses Riemannian gradient descent with momentum, L-BFGS-B polishing, 
and random restarts within a strict time budget.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # Vectorized distance calculation
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    return math.fsum(np.log(1.0 / d).tolist())

def _energy_grad(points):
    """Compute energy and gradient of logarithmic energy.
    
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero on diagonal
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Gradient: -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    d = np.sqrt(d_sq)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _tangent_project(g, p):
    """Project gradient g to tangent space of sphere at p."""
    # g_tan = g - (g . p) p
    return g - (g * p).sum(axis=1, keepdims=True) * p

def _riemannian_flow(points, time_budget, max_iter=20000):
    """Riemannian gradient descent with adaptive step size and momentum."""
    start_time = time.time()
    p = points.copy()
    best_energy = _energy(p)
    best_p = p.copy()
    
    e, g = _energy_grad(p)
    g_tan = _tangent_project(g, p)
    
    step = 0.01
    momentum = np.zeros_like(p)
    beta = 0.9
    failed_steps = 0
    
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        # Momentum update
        momentum = beta * momentum + (1 - beta) * g_tan
        
        new_p = p + step * momentum
        new_p = _project_to_sphere(new_p)
        
        new_e, new_g = _energy_grad(new_p)
        new_g_tan = _tangent_project(new_g, new_p)
        
        if new_e < best_energy:
            best_energy = new_e
            best_p = new_p.copy()
            p = new_p
            g_tan = new_g_tan
            step = min(step * 1.02, 0.04)
            failed_steps = 0
        else:
            step *= 0.5
            failed_steps += 1
            if step < 1e-12 or failed_steps > 10:
                break
            momentum *= 0.5
    
    return best_p

def _polish_lbfgs(points, time_budget, maxiter=5000):
    """Polish with L-BFGS-B using tangent space gradients."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    def _obj_and_grad(x):
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, g = _energy_grad(p)
        g_tan = _tangent_project(g, p)
        return e, g_tan.flatten()
    
    try:
        res = minimize(_obj_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 30})
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points

def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(points)

def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, 3)
    return _project_to_sphere(points)

def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Phase 1: Initial Riemannian flow from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.2:
        optimized = _riemannian_flow(points, TIME_LIMIT * 0.15, max_iter=15000)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: L-BFGS-B polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.35:
        polished = _polish_lbfgs(best_points, TIME_LIMIT * 0.12, maxiter=5000)
        e = _energy(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Phase 3: Multiple random restarts
    # We try several different random starts to escape local minima
    seeds = [123, 456, 789, 101, 202, 303, 404, 505]
    np.random.seed(42)
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.6:
            break
        
        random_points = _random_points(N, seed=seed)
        remaining_time = TIME_LIMIT * 0.6 - elapsed
        budget = min(TIME_LIMIT * 0.02, remaining_time * 0.7)
        if budget < 0.5:
            break
        
        # Quick Riemannian flow
        optimized = _riemannian_flow(random_points, budget, max_iter=5000)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
            # Short polish after improvement
            elapsed = time.time() - start_time
            if elapsed < TIME_LIMIT * 0.55:
                polished = _polish_lbfgs(best_points, min(2.0, (TIME_LIMIT * 0.55 - elapsed) * 0.8), maxiter=2000)
                e2 = _energy(polished)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = polished
    
    # Phase 4: Final L-BFGS-B polish with tight tolerances
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.9:
        remaining = TIME_LIMIT * 0.9 - elapsed
        if remaining > 1.0:
            final_polish = _polish_lbfgs(best_points, remaining * 0.85, maxiter=15000)
            e = _energy(final_polish)
            if e < best_energy:
                best_energy = e
                best_points = final_polish
    
    # Phase 5: Final Riemannian flow for high-precision convergence
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.96:
        remaining = TIME_LIMIT * 0.96 - elapsed
        if remaining > 1.0:
            final_flow = _riemannian_flow(best_points, remaining * 0.9, max_iter=30000)
            e = _energy(final_flow)
            if e < best_energy:
                best_energy = e
                best_points = final_flow
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Perturb point i slightly
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
