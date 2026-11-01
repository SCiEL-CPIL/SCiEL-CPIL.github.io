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
    dists_sq = np.sum(diff * diff, axis=2)
    # Avoid log(0) by setting diagonal to 1
    np.fill_diagonal(dists_sq, 1.0)
    # Only sum upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))
    return energy

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. each point (vectorized).
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists_sq = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero
    dists_sq_safe = np.where(dists_sq < 1e-30, 1e-30, dists_sq)
    # weight[i,j] = 1 / dists_sq[i,j] for i != j, 0 for i == j
    weight = 1.0 / dists_sq_safe
    np.fill_diagonal(weight, 0.0)
    # grad[i] = sum_j weight[i,j] * (pts[i] - pts[j])
    # = sum_j weight[i,j] * diff[i,j]
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    from scipy.optimize import minimize
    
    # Generate multiple initial configurations
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def objective_and_grad(flat_pts):
        pts_3d = flat_pts.reshape(N, 3)
        pts_3d = _project_to_sphere(pts_3d)
        e = _log_energy(pts_3d)
        g = _log_energy_grad(pts_3d)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * pts_3d, axis=1, keepdims=True) * pts_3d
        return e, g_tangent.ravel()
    
    def fun_and_grad(x):
        e, g = objective_and_grad(x)
        return e, g
    
    best_pts = None
    best_energy = np.inf
    
    # Try multiple initializations
    inits = [
        fibonacci_sphere(N, 0.5),
        fibonacci_sphere(N, 0.3),
        fibonacci_sphere(N, 0.7),
    ]
    
    # Add a random initialization
    rng = np.random.default_rng(42)
    inits.append(rng.normal(0, 1, size=(N, 3)))
    inits[-1] = _project_to_sphere(inits[-1])
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 5:
            break
        
        pts = _project_to_sphere(init_pts)
        e_init = _log_energy(pts)
        
        # L-BFGS-B optimization
        try:
            res = minimize(fun_and_grad, pts.ravel(), method='L-BFGS-B', 
                          jac=True, options={'maxiter': 3000, 'maxfun': 50000,
                                            'ftol': 1e-14, 'gtol': 1e-12,
                                            'maxls': 50})
            
            res_pts = _project_to_sphere(res.x.reshape(N, 3))
            res_energy = _log_energy(res_pts)
            
            if res_energy < best_energy:
                best_pts = res_pts
                best_energy = res_energy
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _project_to_sphere(fibonacci_sphere(N, 0.5))
        best_energy = _log_energy(best_pts)
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        try:
            res = minimize(fun_and_grad, best_pts.ravel(), method='L-BFGS-B', 
                          jac=True, options={'maxiter': 5000, 'maxfun': 100000,
                                            'ftol': 1e-16, 'gtol': 1e-14,
                                            'maxls': 50})
            res_pts = _project_to_sphere(res.x.reshape(N, 3))
            res_energy = _log_energy(res_pts)
            if res_energy < best_energy:
                best_pts = res_pts
                best_energy = res_energy
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _project_to_sphere(fibonacci_sphere(N, 0.5))
    
    return best_pts
# EVOLVE-BLOCK-END
