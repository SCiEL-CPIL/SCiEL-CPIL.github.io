"""
Optimized 100-point logarithmic energy minimization on the unit sphere
using Riemannian Conjugate Gradient.
"""
import time
import math
import numpy as np

N = 100
TIME_LIMIT = 135.0  # Safety margin for 150s limit

def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    d_sq = np.maximum(d_sq, 1e-30)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * math.fsum(np.log(d_sq).tolist())

def _energy_grad(points):
    """
    Compute energy and ambient gradient.
    E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    dist_sq_safe = dist_sq.copy()
    np.fill_diagonal(dist_sq_safe, 1.0)
    
    inv_dist_sq = 1.0 / dist_sq_safe
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    grad = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    # Energy
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

def _tangent_project(points, grad):
    """Project gradient to tangent space: g_t = g - (g.p)p."""
    return grad - (grad * points).sum(axis=1, keepdims=True) * points

def _retract(points, v):
    """Retraction: move along tangent vector v and project back to sphere."""
    return _project_to_sphere(points + v)

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)

def _riemannian_cg(points, max_iter=2000, time_budget=10.0, tol=1e-12):
    """
    Riemannian Conjugate Gradient method for minimizing energy on the sphere.
    """
    start_time = time.time()
    p = points.copy()
    e, g_amb = _energy_grad(p)
    g = _tangent_project(p, g_amb)
    
    best_e = e
    best_p = p.copy()
    
    # Initial search direction
    d = -g
    
    for k in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Check convergence
        grad_norm = np.linalg.norm(g)
        if grad_norm < tol:
            break
        
        # Line search in tangent space
        # We want to find alpha such that f(p + alpha*d) is minimized
        # Using backtracking line search
        
        alpha = 1.0
        # Initial step size estimate
        # alpha = 1.0 / (np.linalg.norm(g) + 1e-8)
        
        for _ in range(20):
            p_new = _retract(p, alpha * d)
            e_new, g_amb_new = _energy_grad(p_new)
            
            # Armijo condition
            if e_new < e + 1e-4 * alpha * np.sum(g * d):
                break
            alpha *= 0.5
        else:
            # If no step accepted, break
            break
        
        # Update
        p = p_new
        e = e_new
        g_amb = g_amb_new
        g_new = _tangent_project(p, g_amb)
        
        if e < best_e:
            best_e = e
            best_p = p.copy()
            
        # Compute beta (Fletcher-Reeves)
        # beta = (g_new . g_new) / (g . g)
        # In Riemannian setting, we need to parallel transport d to p_new
        # Approximation: use the projected d
        d_prev = d
        # Transport d to p_new: simple approximation is to project d onto tangent space at p_new
        # Or use the formula beta = ||g_new||^2 / ||g||^2 and d_new = -g_new + beta * d
        # However, d must be in tangent space.
        
        norm_g_sq = np.sum(g * g)
        norm_g_new_sq = np.sum(g_new * g_new)
        
        if norm_g_sq < 1e-30:
            beta = 0.0
        else:
            beta = norm_g_new_sq / norm_g_sq
        
        # Restart if beta is negative or too large
        if beta < 0 or beta > 1.0:
            beta = 0.0
            
        # New direction
        d = -g_new + beta * d
        
        # Ensure d is in tangent space (it should be if g_new is and d was)
        # d = _tangent_project(p, d) # This might be unnecessary if we started tangent
        # But numerical errors accumulate. Let's project.
        d = _tangent_project(p, d)
        
        g = g_new
        
    return best_p, best_e

def _polish(points, time_budget=2.0):
    """
    High-precision polish using steepest descent with very small steps.
    """
    start_time = time.time()
    p = points.copy()
    e, g_amb = _energy_grad(p)
    g = _tangent_project(p, g_amb)
    best_e = e
    best_p = p.copy()
    
    while time.time() - start_time < time_budget:
        grad_norm = np.linalg.norm(g)
        if grad_norm < 1e-14:
            break
        
        # Small step size
        alpha = 0.01 / (grad_norm + 1e-8)
        
        p_new = _retract(p, -alpha * g)
        e_new, g_amb_new = _energy_grad(p_new)
        
        if e_new < e:
            p = p_new
            e = e_new
            g_amb = g_amb_new
            g = _tangent_project(p, g_amb)
            if e < best_e:
                best_e = e
                best_p = p.copy()
        else:
            # If no improvement, reduce step size or break
            # For polish, we just break if we can't improve
            break
            
    return best_p, best_e

def construct():
    start_time = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Fibonacci Spiral
    inits.append(_fibonacci_spiral())
    
    # 2. Random starts
    for i in range(5):
        inits.append(_random_points(seed=42 + i))
        
    # 3. Icosahedron-based (approximate by scaling and rotating random points? 
    # Actually, let's just use more randoms with different seeds)
    for i in range(5):
        inits.append(_random_points(seed=100 + i))

    best_points = inits[0].copy()
    best_energy = _energy(best_points)
    
    # Optimize each initialization
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        # Allocate time based on number of inits remaining
        remaining_inits = len(inits) - i
        time_per_init = (TIME_LIMIT - elapsed - 10.0) / remaining_inits  # Reserve 10s for final polish
        if time_per_init < 1.0:
            time_per_init = 1.0
            
        # Run Riemannian CG
        pts, e = _riemannian_cg(init_pts, max_iter=1000, time_budget=time_per_init)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
            
    # Final Polish
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5.0:
        t_budget = min(TIME_LIMIT - 5.0 - elapsed, 5.0)
        pts, e = _polish(best_points, time_budget=t_budget)
        if e < best_energy:
            best_energy = e
            best_points = pts
            
    # Final Projection and Validation
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                rng = np.random.RandomState(i * 1000 + j)
                best_points[i] += rng.randn(3) * 1e-6
                best_points[i] = best_points[i] / np.linalg.norm(best_points[i])
                
    return best_points
