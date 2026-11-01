"""Stronger optimizer for 100-point logarithmic energy on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 138.0  # safety margin before 150s kill


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * math.fsum(np.log(d_sq).tolist())


def _energy_grad_fast(points):
    """Compute energy and gradient of logarithmic energy using vectorized ops.
    
    E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    # inv_dist_sq[i,j] = 1/||x_i - x_j||^2
    inv_dist_sq = 1.0 / dist_sq_safe
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = -sum_j inv_dist_sq[i,j] * diff[i,j]
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy: sum over i<j of -0.5 * log(dist_sq_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * math.fsum(np.log(d_sq).tolist())
    
    return energy, grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _sphere_fun_grad(x):
    """Objective and gradient for scipy minimize in flat R^(3N) space.
    
    We use the ambient gradient from the sphere-embedded problem.
    The projection to sphere happens implicitly; we return the tangent-space
    gradient mapped back to ambient coordinates.
    """
    p = _project_to_sphere(x.reshape(-1, 3))
    e, g = _energy_grad_fast(p)
    # Project gradient to tangent space: g_tangent = g - (g·p)p
    g_tangent = g - (g * p).sum(axis=1, keepdims=True) * p
    return e, g_tangent.flatten()


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _steepest_descent_sphere(points, max_iter=2000, tol=1e-10, time_budget=10.0):
    """Steepest descent on the sphere with backtracking line search."""
    start_time = time.time()
    p = points.copy()
    e, g = _energy_grad_fast(p)
    best_e = e
    best_p = p.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Project gradient to tangent space
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        
        # Check convergence
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
        
        # Backtracking line search with initial step based on curvature estimate
        step_size = 0.5 / (grad_norm + 1e-8)
        step_size = min(step_size, 0.5)
        for _ in range(40):
            # Move in negative gradient direction and project back to sphere
            p_new = p - step_size * g_tan
            p_new = _project_to_sphere(p_new)
            
            e_new, g_new = _energy_grad_fast(p_new)
            if e_new < e - 1e-4 * step_size * grad_norm * grad_norm:
                break
            step_size *= 0.5
        
        else:
            # If no step accepted, try smaller
            step_size = 1e-8
            p_new = _project_to_sphere(p - step_size * g_tan)
            e_new, g_new = _energy_grad_fast(p_new)
        
        p = p_new
        e = e_new
        g = g_new
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
    
    return best_p, best_e


def _lbfgs_optimize(points, max_iter=5000, time_budget=20.0):
    """L-BFGS-B optimization from given initial points."""
    start_time = time.time()
    x0 = points.flatten().copy()
    
    try:
        res = minimize(_sphere_fun_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
        
        if hasattr(res, 'x'):
            new_points = _project_to_sphere(res.x.reshape(-1, 3))
            new_energy = _energy(new_points)
            return new_points, new_energy
    except Exception:
        pass
    
    return points, _energy(points)


def _simulated_annealing_refine(points, time_budget=15.0):
    """Simulated annealing refinement for logarithmic energy."""
    start_time = time.time()
    p = points.copy()
    e = _energy(p)
    best_p = p.copy()
    best_e = e
    
    # Initial temperature
    T = 0.1
    T_min = 1e-8
    cooling = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        original = p[idx].copy()
        
        # Random perturbation
        delta = np.random.randn(3) * 0.05
        p[idx] = original + delta
        p[idx] = p[idx] / np.linalg.norm(p[idx])
        
        e_new = _energy(p)
        delta_e = e_new - e
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
            e = e_new
            if e < best_e:
                best_e = e
                best_p = p.copy()
        else:
            p[idx] = original
        
        T *= cooling
    
    return best_p, best_e


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral()
    best_points = points.copy()
    best_energy = _energy(best_points)
    
    # Phase 1: L-BFGS from Fibonacci spiral
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.3:
        t_budget = min(TIME_LIMIT * 0.3 - elapsed, 30.0)
        pts, e = _lbfgs_optimize(best_points, max_iter=3000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 2: Steepest descent refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.5:
        t_budget = min(TIME_LIMIT * 0.5 - elapsed, 15.0)
        pts, e = _steepest_descent_sphere(best_points, max_iter=3000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 3: Multiple random restarts with L-BFGS
    np.random.seed(12345)
    elapsed = time.time() - start_time
    restart_count = 0
    while elapsed < TIME_LIMIT * 0.75:
        restart_count += 1
        random_pts = _random_points(seed=1000 + restart_count)
        
        # Quick L-BFGS from random start
        t_budget = min(TIME_LIMIT * 0.75 - elapsed, 8.0)
        pts, e = _lbfgs_optimize(random_pts, max_iter=2000, time_budget=t_budget)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
            # Refine with steepest descent
            elapsed = time.time() - start_time
            if elapsed < TIME_LIMIT * 0.75:
                t_budget2 = min(TIME_LIMIT * 0.75 - elapsed, 5.0)
                pts2, e2 = _steepest_descent_sphere(best_points, max_iter=2000, time_budget=t_budget2)
                if e2 < best_energy:
                    best_energy = e2
                    best_points = pts2
        
        elapsed = time.time() - start_time
    
    # Phase 4: Simulated annealing refinement
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.9:
        t_budget = min(TIME_LIMIT * 0.9 - elapsed, 10.0)
        pts, e = _simulated_annealing_refine(best_points, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 5: Final L-BFGS polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.92:
        t_budget = min(TIME_LIMIT * 0.92 - elapsed, 5.0)
        pts, e = _lbfgs_optimize(best_points, max_iter=2000, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Phase 6: High-precision steepest descent polish with tight tolerance
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT * 0.97:
        t_budget = min(TIME_LIMIT * 0.97 - elapsed, 3.0)
        pts, e = _steepest_descent_sphere(best_points, max_iter=5000, tol=1e-12, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # Final projection and validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        # Shouldn't happen, but safety
        pass
    
    return best_points
