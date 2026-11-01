"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    energy = np.sum(np.log(1.0 / dists[iu]))
    return energy

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. each point.
    dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    # grad[i] = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2 = -sum_{j != i} diff[i,j] / dist_sq[i,j]
    # Using broadcasting: for each i, sum over j of -diff[i,j]/dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _tangent_project(pts, vec):
    """Project vector to tangent space at pts: vec - (vec . x) x"""
    return vec - np.sum(vec * pts, axis=1, keepdims=True) * pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Riemannian Conjugate Gradient
    def _riemannian_cg(pts_init, max_iter=200, tol=1e-10):
        x = _project_to_sphere(pts_init.copy())
        f_prev = _log_energy(x)
        g = _log_energy_grad(x)
        p = _tangent_project(x, -g)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            if np.linalg.norm(p) < tol:
                break
            
            # Line search (backtracking)
            alpha = 0.5
            f_curr = f_prev
            for _ in range(30):
                x_new = _project_to_sphere(x + alpha * p)
                f_new = _log_energy(x_new)
                if f_new < f_curr - 1e-4 * alpha * np.sum(p * _tangent_project(x, -g)):
                    break
                alpha *= 0.5
            else:
                break
            
            g_new = _log_energy_grad(x_new)
            f_new = _log_energy(x_new)
            
            if np.linalg.norm(_tangent_project(x_new, g_new)) < tol:
                x = x_new
                f_prev = f_new
                break
            
            beta = np.sum(_tangent_project(x_new, g_new) * _tangent_project(x_new, g_new)) / \
                   max(np.sum(_tangent_project(x, g) * _tangent_project(x, g)), 1e-15)
            beta = max(0, beta)
            
            p = _tangent_project(x_new, -g_new + beta * p)
            x = x_new
            f_prev = f_new
            g = g_new
        
        return x, f_prev
    
    # Try Riemannian CG
    try:
        pts_cg, e_cg = _riemannian_cg(pts, max_iter=300)
        if e_cg < best_energy:
            best_pts = pts_cg
            best_energy = e_cg
    except Exception:
        pass
    
    # Try L-BFGS-B if time remains
    if time.time() < deadline - 15:
        from scipy.optimize import minimize
        
        def fun_and_grad(x):
            pts_3d = _project_to_sphere(x.reshape(N, 3))
            e = _log_energy(pts_3d)
            g = _log_energy_grad(pts_3d)
            g_tangent = _tangent_project(pts_3d, g)
            return e, g_tangent.ravel()
        
        try:
            res = minimize(fun_and_grad, best_pts.ravel(), method='L-BFGS-B', 
                          jac=True, options={'maxiter': 1000, 'maxfun': 10000,
                                            'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_energy:
                best_pts = _project_to_sphere(res.x.reshape(N, 3))
                best_energy = res.fun
        except Exception:
            pass
    
    # Final polish with very tight tolerance
    if time.time() < deadline - 5:
        try:
            pts_polish, e_polish = _riemannian_cg(best_pts, max_iter=500, tol=1e-12)
            if e_polish < best_energy:
                best_pts = pts_polish
                best_energy = e_polish
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
