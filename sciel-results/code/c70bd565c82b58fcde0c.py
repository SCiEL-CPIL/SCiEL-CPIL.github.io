"""Combined program: smooth-surrogate + exact-polish for 50-sphere packing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s/150s limit

def _compute_radius(pts):
    """Compute the maximum valid radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _neg_radius(x):
    """Objective: negative radius (to be minimized)."""
    pts = x.reshape(N, 3)
    return -_compute_radius(pts)

def _soft_min_radius(x, beta=50.0):
    """
    Smooth surrogate for negative radius using log-sum-exp soft-min.
    Returns a differentiable approximation of -min(constraints).
    """
    pts = x.reshape(N, 3)
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    # Upper triangle indices for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_flat = pair_dists[iu]
    
    # Wall distances
    wall_dists = (0.5 - np.abs(pts)).flatten()
    
    all_dists = np.concatenate([pair_flat, wall_dists])
    
    # Stable soft-min: min(d) ≈ d_min - (1/beta)*log(sum(exp(-beta*(d-d_min))))
    d_min = np.min(all_dists)
    shifted = -beta * (all_dists - d_min)
    # Clip to prevent overflow (shifted <= 0)
    shifted = np.clip(shifted, -700, 0)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = d_min - log_sum / beta
    
    return -soft_min_val

def _init_grid():
    """5x5x2 grid with small jitter."""
    xs = (np.arange(5) + 0.5) / 5.0 - 0.5
    ys = (np.arange(5) + 0.5) / 5.0 - 0.5
    zs = (np.arange(2) + 0.5) / 2.0 - 0.5
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers += np.random.normal(0, 0.001, size=(N, 3))
    return np.clip(centers, -0.49, 0.49)

def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / 3.0
    pts = []
    basis = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
    for i in range(-1, 4):
        for j in range(-1, 4):
            for k in range(-1, 4):
                for b in basis:
                    p = a * np.array([i, j, k]) + a * np.array(b)
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) < N:
        return _init_grid()
    # Greedy selection for spread
    selected = []
    remaining = pts.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        # Pick point farthest from already selected
        if not selected:
            idx = 0
        else:
            sel = np.array(selected)
            d = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    return np.array(selected)

def _init_random():
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _refine_smooth(pts):
    """Refine using smooth surrogate with annealing beta, then exact polish."""
    x = pts.flatten().copy()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Phase 1: Smooth surrogate with increasing beta
    betas = [15.0, 40.0, 100.0, 250.0]
    for beta in betas:
        if time.time() - START_TIME > TIME_LIMIT - 15.0:
            break
        try:
            def obj(x):
                return _soft_min_radius(x, beta)
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
            if np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass
    
    # Phase 2: Exact Nelder-Mead polish
    x_2d = x.reshape(N, 3)
    x_2d = np.clip(x_2d, -0.499, 0.499)
    try:
        res = minimize(_neg_radius, x_2d, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if np.all(np.isfinite(res.x)):
            x_2d = res.x
    except Exception:
        pass
    
    return np.clip(x_2d, -0.499, 0.499)

def _refine_exact(pts):
    """Refine using exact Nelder-Mead only."""
    x = pts.flatten().copy()
    try:
        res = minimize(_neg_radius, x, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if np.all(np.isfinite(res.x)):
            x = res.x
    except Exception:
        pass
    return np.clip(x.reshape(N, 3), -0.499, 0.499)

def construct():
    best_pts = None
    best_r = -np.inf
    
    inits = [
        _init_grid(),
        _init_fcc(),
        _init_random(),
        _init_random(),
    ]
    
    # Add more random inits if time permits
    while time.time() - START_TIME < TIME_LIMIT - 40.0:
        inits.append(_init_random())
    
    for i, init_pts in enumerate(inits):
        if time.time() - START_TIME > TIME_LIMIT - 20.0:
            break
        
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        if i % 2 == 0:
            refined = _refine_smooth(init_pts)
        else:
            refined = _refine_exact(init_pts)
        
        refined = np.clip(refined, -0.499, 0.499)
        r = _compute_radius(refined)
        
        if r > best_r:
            best_r = r
            best_pts = refined.copy()
    
    # Final polish on best solution if time remains
    if best_pts is not None and time.time() - START_TIME < TIME_LIMIT - 5.0:
        try:
            res = minimize(_neg_radius, best_pts, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                r2 = _compute_radius(res.x)
                if r2 > best_r:
                    best_r = r2
                    best_pts = np.clip(res.x, -0.499, 0.499)
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _init_grid()
    
    return best_pts

# EVOLVE-BLOCK-END
