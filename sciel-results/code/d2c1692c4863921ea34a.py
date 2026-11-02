"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import sys

def _score(pts):
    """Compute the packing radius for given centers."""
    # Wall distances
    wall = np.min(np.minimum(0.5 - np.abs(pts), np.abs(pts) + 0.5), axis=1)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_pair, np.min(wall))

def _objective(pts):
    """Negative radius (we minimize negative to maximize radius)."""
    r = _score(pts)
    return -r

def _construct_grid(n):
    """Create a grid of n points in [-0.5, 0.5]^3."""
    m = int(np.ceil(n ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n, seed=42):
    """Random initialization with some structure."""
    rng = np.random.RandomState(seed)
    # Start from a grid and add random perturbation
    pts = _construct_grid(n).copy()
    # Add small random perturbation to break symmetry
    pts += rng.uniform(-0.02, 0.02, size=pts.shape)
    # Clip to cube
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _local_optimize(pts, method='L-BFGS-B', maxiter=2000, tol=1e-12):
    """Run local optimization starting from pts."""
    bounds = [(-0.5, 0.5)] * 3 * len(pts)
    result = minimize(
        _objective,
        pts,
        method=method,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol, 'maxls': 200}
    )
    return result.x, -result.fun

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple initializations
    seeds = [42, 123, 789, 314, 271, 161, 999, 555, 137, 251]
    
    for i, seed in enumerate(seeds):
        if time.time() - start_time > deadline:
            break
        
        # Different initialization strategies
        if i == 0:
            pts = _construct_grid(N)
        elif i == 1:
            pts = _random_init(N, seed=seed)
        elif i == 2:
            # FCC-like lattice perturbation
            pts = _fcc_init(N, seed=seed)
        elif i == 3:
            # BCC-like lattice perturbation  
            pts = _bcc_init(N, seed=seed)
        else:
            # More random
            rng = np.random.RandomState(seed)
            pts = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        # Local optimization
        try:
            opt_pts, r = _local_optimize(pts, maxiter=1500)
            if r > best_r:
                best_r = r
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # If we have time, do a final polish with tighter tolerance
    if best_pts is not None and time.time() - start_time < deadline - 30:
        try:
            opt_pts, r = _local_optimize(best_pts, maxiter=3000, tol=1e-14)
            if r > best_r:
                best_r = r
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final verification and return
    if best_pts is None:
        # Fallback: return grid
        return _construct_grid(N)
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts

def _fcc_init(n, seed=42):
    """Initialize with FCC-like lattice structure."""
    rng = np.random.RandomState(seed)
    # FCC basis vectors
    a = 0.35  # Lattice parameter
    pts = []
    # Generate FCC points
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC positions
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    pts = np.array(pts)
    # Keep only points in cube
    mask = np.all(np.abs(pts) <= 0.48, axis=1)
    pts = pts[mask]
    
    # If we have too many, randomly sample; if too few, add random points
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
        pts = pts[idx]
    else:
        while len(pts) < n:
            pt = rng.uniform(-0.45, 0.45, size=3)
            pts = np.vstack([pts, pt])
    
    # Add small perturbation
    pts += rng.uniform(-0.01, 0.01, size=pts.shape)
    return pts[:n]

def _bcc_init(n, seed=42):
    """Initialize with BCC-like lattice structure."""
    rng = np.random.RandomState(seed)
    a = 0.35
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    pts = np.array(pts)
    mask = np.all(np.abs(pts) <= 0.48, axis=1)
    pts = pts[mask]
    
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
        pts = pts[idx]
    else:
        while len(pts) < n:
            pt = rng.uniform(-0.45, 0.45, size=3)
            pts = np.vstack([pts, pt])
    
    pts += rng.uniform(-0.01, 0.01, size=pts.shape)
    return pts[:n]
# EVOLVE-BLOCK-END
