"""Optimized sphere packing: 50 equal spheres in unit cube."""
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _score(c):
    """Compute the max radius for given centers (vectorized)."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    # Distance to walls
    wall = 0.5 - np.abs(c)
    min_wall = wall.min()
    return min(min_pair, 2.0 * min_wall) / 2.0

def _soft_min(a, beta):
    """Soft minimum via log-sum-exp."""
    m = a.min()
    return m - np.log(np.sum(np.exp(beta * (a - m)))) / beta

def _smooth_objective(c, beta):
    """Negative smooth approximation of the radius (to maximize)."""
    c = c.reshape(-1, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    # Soft min over pairwise distances
    pair_vals = dist
    pair_sm = _soft_min(pair_vals.ravel(), beta)
    # Wall distances
    wall = 0.5 - np.abs(c)
    wall_sm = _soft_min(wall.ravel(), beta)
    # Combined soft min
    combined = _soft_min(np.array([pair_sm, 2.0 * wall_sm]), beta)
    return -combined / 2.0

def _grad_smooth(c, beta):
    """Gradient of smooth objective (approximate via finite differences for speed)."""
    c = c.reshape(-1, 3)
    eps = 1e-6
    grad = np.zeros_like(c)
    for i in range(c.shape[0]):
        for j in range(3):
            cp = c.copy()
            cm = c.copy()
            cp[i, j] += eps
            cm[i, j] -= eps
            grad[i, j] = (_smooth_objective(cp, beta) - _smooth_objective(cm, beta)) / (2 * eps)
    return grad

def construct():
    import time
    t0 = time.time()
    
    # Start from a 5x5x2 grid structure
    # 50 = 5 * 5 * 2
    nx, ny, nz = 5, 5, 2
    # Create grid positions
    xs = np.linspace(-0.5 + 1.0/(nx+1), 0.5 - 1.0/(nx+1), nx)
    ys = np.linspace(-0.5 + 1.0/(ny+1), 0.5 - 1.0/(ny+1), ny)
    zs = np.linspace(-0.5 + 1.0/(nz+1), 0.5 - 1.0/(nz+1), nz)
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    best_c = centers.copy()
    best_r = _score(best_c)
    
    # Try multiple beta values for soft-min, starting coarse
    betas = [50, 100, 200, 500, 1000, 2000]
    
    for beta in betas:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        c_flat = best_c.ravel()
        
        try:
            res = minimize(
                _smooth_objective,
                c_flat,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            c_new = res.x.reshape(-1, 3)
            r_new = _score(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new
        except Exception:
            pass
    
    # Final polish with exact objective using Nelder-Mead or BFGS
    if time.time() - t0 < DEADLINE - 5:
        try:
            bounds = [(-0.5, 0.5)] * (N * 3)
            c_flat = best_c.ravel()
            
            def exact_obj(c):
                return -_score(c.reshape(-1, 3))
            
            res = minimize(
                exact_obj,
                c_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_new = res.x.reshape(-1, 3)
            r_new = _score(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
