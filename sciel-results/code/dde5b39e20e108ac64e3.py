import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and Riemannian gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad) where grad is the Riemannian gradient (tangent space).
    """
    n = len(pts)
    # Compute pairwise distances squared
    # dist_sq[i,j] = ||x_i - x_j||^2 = 2 - 2 x_i.x_j
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)  # Avoid log(0) and self-interaction
    
    # Check for very small distances (invalid)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf'), np.zeros_like(pts)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Note: log(1/d) = -log(d). Sum over i<j is half of sum over i!=j.
    # E = -0.5 * sum_{i!=j} log(sqrt(dist_sq)) = -0.25 * sum_{i!=j} log(dist_sq)
    # Wait, the task says log(1/distance) where distance is chordal distance.
    # d = sqrt(dist_sq).
    # log(1/d) = -log(d) = -0.5 log(dist_sq).
    # Sum_{i<j} log(1/d_ij) = -0.5 * Sum_{i<j} log(dist_sq_ij).
    
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    
    # Gradient w.r.t x_i in R^3 (Euclidean gradient before projection)
    # E = -0.5 sum_{j!=i} log(||x_i - x_j||)
    # dE/dx_i = -0.5 sum_{j!=i} (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j||
    # dE/dx_i = -0.5 sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # dE/dx_i = -0.5 sum_{j!=i} (x_i - x_j) / dist_sq_ij
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    # weights[i,j] = -0.5 / dist_sq[i,j]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    # grad_eucl[i] = sum_j weights[i,j] * diff[i,j]
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space: grad_riem = grad_eucl - (grad_eucl . x_i) x_i
    proj_factor = np.sum(grad_eucl * pts, axis=1, keepdims=True)
    grad_riem = grad_eucl - proj_factor * pts
    
    return float(energy), grad_riem

def _riemannian_cg(pts, max_iter=100, tol=1e-8):
    """
    Riemannian Conjugate Gradient method.
    pts: initial points (N, 3).
    """
    x = pts.copy()
    e, g = _log_energy_and_grad(x)
    if not np.isfinite(e):
        return x, e
    
    # Initial direction: negative gradient
    d = -g
    # Normalize d to have unit norm in tangent space? Not strictly necessary, but helps.
    
    for k in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        # Check convergence
        if np.linalg.norm(g) < tol:
            break
        
        # Line search: minimize f(x + t * Exp_x(d)) for t in [0, 1]
        # Exp_x(v) = x cos(||v||) + v sin(||v||)/||v||
        # We use backtracking line search.
        t = 1.0
        for _ in range(20):
            # Compute new point
            norm_d = np.linalg.norm(d, axis=1, keepdims=True)
            norm_d = np.maximum(norm_d, 1e-15)
            # v = t * d
            v = t * d
            norm_v = np.linalg.norm(v, axis=1, keepdims=True)
            norm_v = np.maximum(norm_v, 1e-15)
            
            # x_new = x cos(theta) + v sin(theta)/||v|| * ||v|| ? No.
            # Exp_x(v) = x * cos(||v||) + (v/||v||) * sin(||v||)
            # Let's implement carefully.
            cos_t = np.cos(norm_v)
            sin_t_over_norm_v = np.sin(norm_v) / norm_v
            x_new = x * cos_t + v * sin_t_over_norm_v
            
            # Normalize x_new to ensure it's on the sphere (numerical stability)
            x_new_norms = np.linalg.norm(x_new, axis=1, keepdims=True)
            x_new = x_new / np.maximum(x_new_norms, 1e-15)
            
            e_new, g_new = _log_energy_and_grad(x_new)
            if not np.isfinite(e_new):
                t *= 0.5
                continue
            
            # Armijo condition: f(x_new) <= f(x) + c * t * <g, d>
            # <g, d> is the Riemannian inner product
            inner_gd = np.sum(g * d)
            if e_new <= e + 1e-4 * t * inner_gd:
                break
            t *= 0.5
        
        if t < 1e-10:
            break
            
        # Update x
        norm_d = np.linalg.norm(d, axis=1, keepdims=True)
        norm_d = np.maximum(norm_d, 1e-15)
        v = t * d
        norm_v = np.linalg.norm(v, axis=1, keepdims=True)
        norm_v = np.maximum(norm_v, 1e-15)
        cos_t = np.cos(norm_v)
        sin_t_over_norm_v = np.sin(norm_v) / norm_v
        x = x * cos_t + v * sin_t_over_norm_v
        x_norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(x_norms, 1e-15)
        
        e = e_new
        g = g_new
        
        # Update direction (Polak-Ribiere)
        g_old = g_old if k > 0 else None
        if k > 0:
            inner_gg_old = np.sum(g_old * g)
            inner_gg_new = np.sum(g * g)
            beta = max(0, inner_gg_old / (inner_gg_new + 1e-12))
            # Parallel transport of d to x_new: d_pt = d - (d . x_old) x_old + (d . x_new) x_new ?
            # Simpler: just use d = -g + beta * d_pt
            # For simplicity and robustness, we can restart if beta is large or just use Fletcher-Reeves.
            # Let's use Fletcher-Reeves: beta = ||g_new||^2 / ||g_old||^2
            beta = inner_gg_old / (inner_gg_new + 1e-12)
            if beta > 1.0:
                beta = 1.0
            d = -g + beta * d
        else:
            d = -g
            
        g_old = g
        
    return x, e

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _rotate_sphere(pts, angle, axis):
    """Rotate points around a given axis by a given angle."""
    # Normalize axis
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return pts @ R.T

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # 1. Fibonacci sphere
    fib_pts = _fibonacci_sphere(N)
    
    # 2. Rotated Fibonacci spheres for diversity
    rotations = [
        (np.pi/2, [1, 0, 0]),
        (np.pi/3, [0, 1, 0]),
        (np.pi/4, [1, 1, 1]),
        (np.pi/6, [1, 2, 3]),
    ]
    
    candidates = [fib_pts]
    for angle, axis in rotations:
        rot_pts = _rotate_sphere(fib_pts, angle, axis)
        candidates.append(rot_pts)
        
    # 3. Random uniform points (a couple of seeds)
    for seed in [42, 123]:
        np.random.seed(seed)
        theta = np.random.uniform(0, 2 * np.pi, N)
        phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        rand_pts = np.column_stack([x, y, z])
        candidates.append(rand_pts)
        
    # Optimize each candidate
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
            
        try:
            # Run Riemannian CG
            opt_pts, e = _riemannian_cg(init_pts, max_iter=50, tol=1e-8)
            
            # Verify energy
            e_check, _ = _log_energy_and_grad(opt_pts)
            if e_check < best_energy and np.isfinite(e_check):
                best_energy = e_check
                best_pts = opt_pts
        except Exception:
            pass
            
    # If we have time, do a final polish on the best result with more iterations
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        try:
            opt_pts, e = _riemannian_cg(best_pts, max_iter=100, tol=1e-10)
            e_check, _ = _log_energy_and_grad(opt_pts)
            if e_check < best_energy and np.isfinite(e_check):
                best_energy = e_check
                best_pts = opt_pts
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = fib_pts

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
