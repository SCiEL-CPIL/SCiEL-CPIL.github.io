"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 110.0  # Safety margin for the 120s limit

# EVOLVE-BLOCK-START

def _radius(pts):
    """Compute the largest common radius for the given centres."""
    # Wall distance: min over i,k of (0.5 - |c_ik|)
    # c_ik is coordinate k of point i
    # dist to wall is 0.5 - max(|x|, |y|)
    wall_dists = 0.5 - np.max(np.abs(pts), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # Using broadcasting
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_wall, min_pair)

def _penalty_smooth(pts_flat, r_target):
    """
    Objective function for L-BFGS-B.
    Minimizes the squared violation of the constraints for radius r_target.
    Constraints:
    1. d_ij >= 2 * r_target
    2. Points inside [-0.5 + r, 0.5 - r]^2
    """
    pts = pts_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Violations for pairs
    # We want d >= 2*r_target
    viol_pairs = np.maximum(0, 2 * r_target - d)
    
    # Use triu_indices for efficiency in summing unique pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    viol_pairs_unique = viol_pairs[i_idx, j_idx]
    E_pairs = np.sum(viol_pairs_unique**2)
    
    # Wall distances
    # Instead of max(|x|, |y|), we use linear constraints:
    # x_i <= 0.5 - r, x_i >= -0.5 + r
    # y_i <= 0.5 - r, y_i >= -0.5 + r
    # This is smoother for optimization.
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Violations for walls
    # Right wall: x > 0.5 - r
    viol_x_right = np.maximum(0, x - (0.5 - r_target))
    # Left wall: x < -0.5 + r
    viol_x_left = np.maximum(0, (-0.5 + r_target) - x)
    # Top wall: y > 0.5 - r
    viol_y_top = np.maximum(0, y - (0.5 - r_target))
    # Bottom wall: y < -0.5 + r
    viol_y_bottom = np.maximum(0, (-0.5 + r_target) - y)
    
    E_walls = np.sum(viol_x_right**2) + np.sum(viol_x_left**2) + \
              np.sum(viol_y_top**2) + np.sum(viol_y_bottom**2)
    
    return E_pairs + E_walls

def _grad_penalty_smooth(pts_flat, r_target):
    """
    Gradient of the smooth penalty function.
    """
    pts = pts_flat.reshape(N, 2)
    grad = np.zeros_like(pts)
    
    # Pairwise gradient
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-8, d, 1.0)
    
    # viol = max(0, 2r - d)
    # d(viol^2)/d(pts_i) = 2 * viol * d(viol)/d(pts_i)
    # d(viol)/d(pts_i) = - (pts_i - pts_j) / d_ij  if viol > 0
    # So d(E)/d(pts_i) = sum_j 2 * viol_ij * (-(pts_i - pts_j) / d_ij)
    
    # We only care about viol > 0
    mask = (2 * r_target - d) > 0
    # Set diagonal to False just in case
    np.fill_diagonal(mask, False)
    
    # Weight matrix W_ij = 2 * viol_ij / d_ij
    W = np.zeros((N, N))
    W[mask] = 2 * (2 * r_target - d[mask]) / d_safe[mask]
    
    # grad_i = - sum_j W_ij (pts_i - pts_j) = - (sum_j W_ij) pts_i + sum_j W_ij pts_j
    row_sums = W.sum(axis=1)
    grad_pairs = -row_sums[:, np.newaxis] * pts + W.dot(pts)
    
    # Wall gradient
    # E_walls = sum (max(0, x - (0.5-r))^2 + max(0, (-0.5+r) - x)^2 + ...)
    # d/dx (max(0, x - a)^2) = 2 * max(0, x - a)
    # d/dx (max(0, b - x)^2) = -2 * max(0, b - x)
    
    x = pts[:, 0]
    y = pts[:, 1]
    
    viol_x_right = np.maximum(0, x - (0.5 - r_target))
    viol_x_left = np.maximum(0, (-0.5 + r_target) - x)
    viol_y_top = np.maximum(0, y - (0.5 - r_target))
    viol_y_bottom = np.maximum(0, (-0.5 + r_target) - y)
    
    grad_walls = np.zeros_like(pts)
    grad_walls[:, 0] = 2 * viol_x_right - 2 * viol_x_left
    grad_walls[:, 1] = 2 * viol_y_top - 2 * viol_y_bottom
    
    grad = grad_pairs + grad_walls
    return grad.flatten()

def _init_grid(jitter=0.0):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    if jitter > 0:
        rng = np.random.default_rng(42)
        grid = grid + rng.normal(0, jitter, grid.shape)
        grid = np.clip(grid, -0.49, 0.49)
    return grid

def _pbd_solve(pts, r_target, max_iter=2000, tol=1e-8, deadline=None):
    """
    Position-Based Dynamics solver for circle packing.
    Projects points onto feasible set defined by:
    1. Pairwise distance >= 2*r_target
    2. Points inside [-0.5+r_target, 0.5-r_target]^2
    
    Returns (pts, converged) where converged is True if max violation < tol.
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        max_viol = 0.0
        changed = False
        
        # Pairwise constraint projection
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                min_d = 2.0 * r_target
                if d < min_d:
                    if d < 1e-12:
                        # Coincident points: push apart randomly
                        diff = np.array([1.0, 0.0])
                        d = 1.0
                    else:
                        diff = diff / d
                    # Move each point by half the violation
                    correction = 0.5 * (min_d - d) * diff
                    pts[i] += correction
                    pts[j] -= correction
                    viol = min_d - d
                    if viol > max_viol:
                        max_viol = viol
                    changed = True
        
        # Wall constraint projection
        lo = -0.5 + r_target
        hi = 0.5 - r_target
        for i in range(n):
            for k in range(2):
                if pts[i, k] < lo:
                    pts[i, k] = lo
                    viol = lo - pts[i, k] + (hi - lo)  # crude
                    if (lo - pts[i, k]) > max_viol:
                        max_viol = lo - pts[i, k]
                    changed = True
                elif pts[i, k] > hi:
                    pts[i, k] = hi
                    if (pts[i, k] - hi) > max_viol:
                        max_viol = pts[i, k] - hi
                    changed = True
        
        if not changed:
            break
        if max_viol < tol:
            break
    
    # Compute final max violation
    max_viol_final = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            viol = 2.0 * r_target - d
            if viol > max_viol_final:
                max_viol_final = viol
    for i in range(n):
        for k in range(2):
            if pts[i, k] < lo:
                max_viol_final = max(max_viol_final, lo - pts[i, k])
            elif pts[i, k] > hi:
                max_viol_final = max(max_viol_final, pts[i, k] - hi)
    
    converged = max_viol_final < tol
    return pts, converged

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT - 10  # Safety margin
    
    # Initial guess: Grid
    best_pts = _init_grid()
    best_r = _radius(best_pts)
    
    # Binary search for radius
    r_low = best_r
    r_high = 0.075  # Conservative upper bound (known best is ~0.0714)
    
    rng = np.random.default_rng(42)
    
    max_steps = 12
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid using PBD
        feasible = False
        best_pts_for_r = None
        best_r_for_r = 0
        
        # Multiple restarts with different initializations
        n_restarts = 3
        for k in range(n_restarts):
            if time.time() > deadline:
                break
                
            if k == 0:
                init_pts = _init_grid(jitter=0.002)
            elif k == 1:
                init_pts = _init_grid(jitter=0.008)
            else:
                # Jittered previous best if available
                if best_pts_for_r is not None:
                    init_pts = best_pts_for_r + rng.normal(0, 0.003, best_pts_for_r.shape)
                else:
                    init_pts = _init_grid(jitter=0.015)
            
            init_pts = np.clip(init_pts, -0.49, 0.49)
            
            try:
                cand_pts, converged = _pbd_solve(init_pts, r_mid, max_iter=3000, tol=1e-9, deadline=deadline)
                
                if converged:
                    r_cand = _radius(cand_pts)
                    if r_cand >= r_mid - 1e-7:
                        feasible = True
                        if r_cand > best_r_for_r:
                            best_r_for_r = r_cand
                            best_pts_for_r = cand_pts
            except Exception:
                pass
        
        if feasible and best_pts_for_r is not None:
            r_low = r_mid
            best_pts = best_pts_for_r
            best_r = best_r_for_r
        else:
            r_high = r_mid
            
        # If we've converged enough
        if r_high - r_low < 1e-7:
            break

    # Final polish with L-BFGS-B to squeeze out more precision
    if time.time() < deadline - 5:
        current_r = _radius(best_pts)
        # Try to improve with penalty-based optimizer starting from PBD solution
        for _ in range(2):
            if time.time() > deadline:
                break
            r_try = current_r + 1e-5
            init_pts = best_pts + rng.normal(0, 0.0005, best_pts.shape)
            try:
                res = minimize(
                    _penalty_smooth,
                    init_pts.flatten(),
                    args=(r_try,),
                    jac=_grad_penalty_smooth,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                if res.fun < 1e-8:
                    cand_pts = res.x.reshape(N, 2)
                    r_cand = _radius(cand_pts)
                    if r_cand > current_r:
                        best_pts = cand_pts
                        current_r = r_cand
                    else:
                        break
                else:
                    break
            except Exception:
                break
        best_r = _radius(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
