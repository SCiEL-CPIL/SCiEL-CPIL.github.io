import numpy as np
from scipy.optimize import minimize
import time
import random

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before the 150s kill

def _compute_radius(c):
    """
    Compute the true radius and check validity.
    c: (N, 2) array
    Returns 0.0 if invalid.
    """
    if c is None or not isinstance(c, np.ndarray):
        return 0.0
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    
    # Pairwise distances
    # Using broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    min_pair_dist = np.min(dists[mask])
    
    # Boundary distances
    boundary_dists = np.min(0.5 - np.abs(c), axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if r < 0:
        return 0.0
        
    return r

def _soft_min_obj_and_grad(c_flat, beta):
    """
    Objective: -S(c) where S(c) is the soft-min approximation of the min distance.
    c_flat: (2*N,) array
    """
    c = c_flat.reshape(N, 2)
    
    # 1. Compute distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    eps = 1e-12
    dists_safe = np.maximum(dists, eps)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Pairwise part
    # U = -1/beta * log(sum(exp(-beta * d_ij)))
    exponents_p = -beta * dists
    exponents_p[~mask] = -np.inf # Ignore diagonal
    
    max_exp_p = np.max(exponents_p)
    if np.isinf(max_exp_p):
        return 1e10, np.zeros_like(c_flat)
        
    sum_exp_p = np.sum(np.exp(exponents_p - max_exp_p))
    if sum_exp_p == 0:
        return 1e10, np.zeros_like(c_flat)
        
    U = -1.0/beta * (np.log(sum_exp_p) + max_exp_p)
    
    weights_p = np.exp(exponents_p - max_exp_p) / sum_exp_p
    
    # Boundary part
    # b_ik = 0.5 - |c_ik|
    b_ik = 0.5 - np.abs(c)
    
    exponents_b = -beta * b_ik
    max_exp_b = np.max(exponents_b)
    if np.isinf(max_exp_b):
        return 1e10, np.zeros_like(c_flat)
        
    sum_exp_b = np.sum(np.exp(exponents_b - max_exp_b))
    if sum_exp_b == 0:
        return 1e10, np.zeros_like(c_flat)
        
    V = -1.0/beta * (np.log(sum_exp_b) + max_exp_b)
    
    weights_b = np.exp(exponents_b - max_exp_b) / sum_exp_b
    
    # Combine U and V into S
    # S = -1/(2*beta) * log(exp(-beta*U) + exp(-beta*V))
    exp_U = np.exp(-beta * U)
    exp_V = np.exp(-beta * V)
    
    denom = exp_U + exp_V
    if denom == 0:
        return 1e10, np.zeros_like(c_flat)
        
    S = -1.0/(2.0*beta) * np.log(denom)
    
    # Gradient
    grad_U = np.zeros_like(c)
    grad_V = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        grad_dists = diff / dists_safe[:, :, np.newaxis]
        
    # Sum weighted gradients
    grad_U = np.sum(weights_p[:, :, np.newaxis] * grad_dists, axis=1)
    
    sign_c = np.sign(c)
    grad_b = -sign_c
    grad_V = np.sum(weights_b[:, :, np.newaxis] * grad_b, axis=1)
    
    grad_S = 0.5 * (exp_U * grad_U + exp_V * grad_V) / denom
    
    obj = -S
    grad = -grad_S.flatten()
    
    return obj, grad

def _construct_grid():
    # Create a grid that is slightly denser than needed, then pick best N
    # For N=50, 7x7=49, 8x8=64.
    # Let's try 7x7 plus some extra points or an 8x8 subset.
    # A 7x7 grid is 49 points. We need 50.
    # Let's use 8x8 grid and remove the worst 14 points? Or just take 50?
    # Taking 50 from 8x8 might be unbalanced.
    # Let's create a 7x7 grid and add 1 point in the center of a cell?
    # Or just use a 5x10 grid?
    # 7x7 grid spacing 1/7 ~ 0.142, r ~ 0.071. This is close to the target!
    # The target is 0.0713.
    # Let's generate a 7x7 grid and one extra point.
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Add a point at the center? Or shift the grid?
    # Actually, a 7x7 grid has 49 points.
    # Let's just return the 7x7 grid plus one point at (0,0) or somewhere?
    # Better: return 7x7 grid and let the optimizer move it.
    # Or return 8x8 grid and trim to 50?
    # Let's return a 7x7 grid with the center point removed and two points added?
    # Simplest: 7x7 grid + 1 random point near center.
    extra = np.random.uniform(-0.05, 0.05, (1, 2))
    grid = np.vstack([grid, extra])
    return grid[:N]

def _construct_hex():
    # Hexagonal packing
    pts = []
    # Generate a hex grid in the unit square
    # Spacing based on target radius r ~ 0.071
    # Side length 2r ~ 0.142
    # Horizontal dist: 2r
    # Vertical dist: r * sqrt(3) ~ 0.123
    # Let's just generate a dense hex grid and pick 50 points
    # Or generate a specific hex pattern.
    # For 50 circles, a hex grid with about 8 columns and 7 rows might work.
    # Let's generate points and filter to those inside the square.
    # Then pick the 50 that maximize min distance? That's expensive.
    # Just pick first 50 from a sorted list.
    
    # Generate hex points
    r_target = 0.0714
    dx = 2 * r_target
    dy = r_target * np.sqrt(3)
    
    # Range to cover the square
    x_start = -0.5
    y_start = -0.5
    x_end = 0.5
    y_end = 0.5
    
    row = 0
    while y_start + row * dy < y_end + dy:
        y = y_start + row * dy
        x_offset = (row % 2) * dx / 2
        col = 0
        while x_start + col * dx + x_offset < x_end + dx:
            x = x_start + col * dx + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
            col += 1
        row += 1
        
    pts = np.array(pts)
    if len(pts) < N:
        # Fallback to grid
        return _construct_grid()
    
    # Shuffle to avoid bias? Or just take first N?
    # Taking first N from a top-left to bottom-right scan might be biased.
    # Let's randomly sample N points from the valid hex points.
    indices = np.random.choice(len(pts), N, replace=False)
    return pts[indices]

def _construct_random():
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _construct_smart():
    # Start with a grid, then add random perturbations
    grid = _construct_grid()
    return grid + np.random.normal(0, 0.02, (N, 2))

def construct():
    best_c = _construct_grid()
    best_r = _compute_radius(best_c)
    
    initializers = [
        _construct_grid(),
        _construct_hex(),
        _construct_random(),
        _construct_random(),
        _construct_smart(),
    ]
    
    # Beta schedule
    betas = [50, 100, 200, 500, 1000, 2000]
    
    for init_c in initializers:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        current_c = init_c.copy()
        
        for beta in betas:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            c_flat = current_c.flatten()
            bounds = [(-0.5, 0.5)] * (2*N)
            
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    c_flat,
                    method='L-BFGS-B',
                    jac=True,
                    args=(beta,),
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                new_c = res.x.reshape(N, 2)
                new_r = _compute_radius(new_c)
                
                if new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
                    
                current_c = new_c
                
            except Exception:
                pass
                
    # Final polish with very high beta on the best candidate found
    if time.time() - START_TIME < TIME_LIMIT - 2:
        c_flat = best_c.flatten()
        bounds = [(-0.5, 0.5)] * (2*N)
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                c_flat,
                method='L-BFGS-B',
                jac=True,
                args=(5000,),
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_c = res.x.reshape(N, 2)
            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_c = new_c
        except:
            pass
            
    return best_c

if __name__ == "__main__":
    c = construct()
    r = _compute_radius(c)
    print(f"Radius: {r}")
