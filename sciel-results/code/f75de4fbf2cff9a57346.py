"""Stronger algorithm for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _energy_grad_fast(points):
    """Compute energy and gradient efficiently using dist_sq directly."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    return energy, grad


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = np.sqrt(np.maximum(dist_sq[mask], 1e-30))
    return math.fsum(np.log(1.0 / d).tolist())


def _energy_grad(points):
    """Compute energy and gradient of logarithmic energy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    energy = math.fsum(np.log(1.0 / d).tolist())
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient in ambient R^(3N) space with projection."""
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _optimize_single(points, time_budget, maxiter=3000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from a single starting point."""
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _steepest_descent_sphere(points, time_budget, max_iter=5000):
    """Steepest descent on the sphere using Riemannian gradient."""
    start_time = time.time()
    p = points.copy()
    best_energy = _energy(p)
    best_p = p.copy()
    
    e, g = _energy_grad_fast(p)
    g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
    
    step = 0.01
    for _ in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        new_p = p + step * g_tan
        new_p = _project_to_sphere(new_p)
        
        new_e = _energy(new_p)
        if new_e < best_energy:
            best_energy = new_e
            best_p = new_p.copy()
            _, g = _energy_grad_fast(new_p)
            g_tan = g - (g * new_p).sum(axis=1, keepdims=True) * new_p
            p = new_p
            step = min(step * 1.05, 0.1)
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    return best_p


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


def _simulated_annealing_refine(points, time_budget):
    """Refine points using simulated annealing with rotation and swap moves."""
    start_time = time.time()
    best_points = points.copy()
    best_energy = _energy(best_points)
    T = 0.04
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        move_type = np.random.random()
        
        if move_type < 0.65:
            # Rotation move
            n_moves = np.random.choice([1, 1, 1, 2, 2])
            indices = np.random.choice(N, size=n_moves, replace=False)
            
            new_points = best_points.copy()
            for idx in indices:
                angle = np.random.uniform(0, 2 * np.pi) * T
                axis = np.random.randn(3)
                axis /= np.linalg.norm(axis)
                
                c = math.cos(angle)
                s = math.sin(angle)
                k = axis
                k_outer_k = np.outer(k, k)
                K = np.array([[0, -k[2], k[1]],
                              [k[2], 0, -k[0]],
                              [-k[1], k[0], 0]])
                R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
                
                new_points[idx] = R @ new_points[idx]
        else:
            # Swap move with small rotation
            i, j = np.random.choice(N, size=2, replace=False)
            new_points = best_points.copy()
            
            angle = np.random.uniform(0, 2 * np.pi) * T * 0.5
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            
            temp_i = new_points[i].copy()
            new_points[i] = R @ new_points[j]
            new_points[j] = R @ temp_i
        
        new_points = _project_to_sphere(new_points)
        
        new_energy = _energy(new_points)
        delta_e = new_energy - best_energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            best_points = new_points
            best_energy = new_energy
        
        T *= 0.9997
        if T < 1e-6:
            T = 0.02
    
    return best_points


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Phase 1: L-BFGS-B from Fibonacci spiral (generous budget)
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.35:
        budget = min(TIME_LIMIT * 0.30, TIME_LIMIT * 0.35 - elapsed)
        optimized = _optimize_single(points, budget, maxiter=5000, ftol=1e-15, gtol=1e-11)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 2: Multiple random restarts with L-BFGS-B
    np.random.seed(42)
    restart_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT * 0.65:
            break
        restart_count += 1
        
        seed = 1000 + restart_count * 7
        random_points = _random_points(N, seed=seed)
        
        budget_per_restart = min(TIME_LIMIT * 0.04, (TIME_LIMIT * 0.65 - elapsed) * 0.7)
        if budget_per_restart < 1.5:
            break
            
        optimized = _optimize_single(random_points, budget_per_restart, maxiter=2000, ftol=1e-13, gtol=1e-10)
        e = _energy(optimized)
        if e < best_energy:
            best_energy = e
            best_points = optimized
    
    # Phase 3: Fine-tune the best solution
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.80:
        remaining = TIME_LIMIT * 0.80 - elapsed
        if remaining > 2.0:
            refined = _optimize_single(best_points, remaining * 0.9, maxiter=6000, ftol=1e-15, gtol=1e-12)
            e = _energy(refined)
            if e < best_energy:
                best_energy = e
                best_points = refined
    
    # Phase 4: Simulated annealing with swaps
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.90:
        remaining = TIME_LIMIT * 0.90 - elapsed
        if remaining > 1.5:
            np.random.seed(789)
            sa_points = _simulated_annealing_refine(best_points, remaining * 0.85)
            e = _energy(sa_points)
            if e < best_energy:
                best_energy = e
                best_points = sa_points
    
    # Phase 5: Final high-precision L-BFGS polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.97:
        remaining = TIME_LIMIT * 0.97 - elapsed
        if remaining > 1.0:
            final_opt = _optimize_single(best_points, remaining * 0.9, maxiter=3000, ftol=1e-16, gtol=1e-13)
            e = _energy(final_opt)
            if e < best_energy:
                best_energy = e
                best_points = final_opt
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                perturb = np.random.randn(3) * 1e-5
                best_points[i] = best_points[i] + perturb
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final energy check
    final_energy = _energy(best_points)
    if not np.isfinite(final_energy):
        best_points = _fibonacci_spiral(N)
    
    return best_points
