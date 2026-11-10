"""Improved packing of 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

_DEADLINE = 115.0  # safety margin before 150s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _radius(pts):
    """Compute the feasible radius for a set of centres in [-0.5, 0.5]^2."""
    # Boundary constraint
    bound = 0.5 - np.max(np.abs(pts), axis=1)
    r_bound = np.min(bound)
    if r_bound <= 0:
        return 0.0
    # Pairwise distances
    d = pdist(pts)
    r_pair = np.min(d) / 2.0
    return min(r_bound, r_pair)


def _neg_radius(pts_flat):
    """Objective: negative radius (to maximize)."""
    pts = pts_flat.reshape(-1, 2)
    return -_radius(pts)


def _jittered_hexagonal(n):
    """Create a jittered hexagonal lattice initialization for n points."""
    # Approximate hexagonal arrangement
    # For 50 points, use a grid with hexagonal offset
    rows = 7
    cols = 8
    # Spacing to fit in unit square
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= n:
                break
            x = -0.5 + (j + 0.5) * dx + (0.5 if i % 2 == 0 else 0.0) * dx * 0.5
            y = -0.5 + (i + 0.5) * dy
            # Clamp to boundary
            x = np.clip(x, -0.45, 0.45)
            y = np.clip(y, -0.45, 0.45)
            pts.append([x, y])
    
    # If we don't have enough, add random points
    while len(pts) < n:
        pts.append([np.random.uniform(-0.45, 0.45), np.random.uniform(-0.45, 0.45)])
    
    pts = np.array(pts[:n])
    
    # Add small jitter
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.01, size=pts.shape)
    pts = pts + jitter
    pts = np.clip(pts, -0.45, 0.45)
    
    return pts


def _jittered_random(n, seed):
    """Create a random initialization with jitter."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.45, 0.45, size=(n, 2))
    return pts


def _jittered_grid(n, seed):
    """Create a jittered square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:n].copy()
    
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, 0.02, size=pts.shape)
    pts = pts + jitter
    pts = np.clip(pts, -0.45, 0.45)
    return pts


def construct():
    global _start_time
    _start_time = time.time()
    
    best_pts = None
    best_r = 0.0
    
    # Multiple initialization strategies
    inits = [
        _jittered_hexagonal(N),
        _jittered_grid(N, 1),
        _jittered_grid(N, 2),
        _jittered_random(N, 3),
        _jittered_random(N, 4),
    ]
    
    for init_idx, init_pts in enumerate(inits):
        if _elapsed() > _DEADLINE - 5:
            break
        
        init_flat = init_pts.flatten()
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            result = minimize(
                _neg_radius,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True,
                }
            )
            opt_pts = result.x.reshape(-1, 2)
            # Clamp to valid region
            opt_pts = np.clip(opt_pts, -0.499, 0.499)
            
            r = _radius(opt_pts)
            if r > best_r:
                best_r = r
                best_pts = opt_pts.copy()
        except Exception:
            continue
        
        # Also try a second pass with smaller jitter from the best result
        if best_pts is not None and _elapsed() < _DEADLINE - 10:
            try:
                rng = np.random.default_rng(init_idx + 100)
                fine_pts = best_pts + rng.normal(0, 0.005, size=best_pts.shape)
                fine_pts = np.clip(fine_pts, -0.499, 0.499)
                
                result2 = minimize(
                    _neg_radius,
                    fine_pts.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True,
                    }
                )
                opt_pts2 = result2.x.reshape(-1, 2)
                opt_pts2 = np.clip(opt_pts2, -0.499, 0.499)
                
                r2 = _radius(opt_pts2)
                if r2 > best_r:
                    best_r = r2
                    best_pts = opt_pts2.copy()
            except Exception:
                continue
    
    # Final refinement pass
    if best_pts is not None and _elapsed() < _DEADLINE - 3:
        try:
            result_final = minimize(
                _neg_radius,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            final_pts = result_final.x.reshape(-1, 2)
            final_pts = np.clip(final_pts, -0.499, 0.499)
            
            r_final = _radius(final_pts)
            if r_final > best_r:
                best_r = r_final
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Final validation: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    return best_pts
# EVOLVE-BLOCK-END
