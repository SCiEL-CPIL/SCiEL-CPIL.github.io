"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist
import itertools

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    # For 50 points, 1225 pairs - manageable
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _objective(x, bounds):
    """Objective function to minimize (negative radius)."""
    centers = x.reshape(-1, 3)
    # Check if centers are within bounds (with small tolerance)
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 1e10  # Penalty for out-of-bounds
    
    r = _compute_radius(centers)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: 5x5x2 grid
    # x, y: 5 points in [-0.4, 0.4], step 0.2
    # z: 2 points in [-0.25, 0.25], step 0.5
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = x0.copy()
    best_score = _compute_radius(centers)
    
    # Try a few different initializations and optimization strategies
    strategies = [
        ("L-BFGS-B", {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-12}),
        ("Nelder-Mead", {"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-10}),
    ]
    
    # Also try FCC-based initialization
    # FCC lattice points
    def _fcc_init():
        """Generate FCC-like initialization."""
        pts = []
        # Face-centered cubic arrangement
        # Use a subset of FCC points
        a = 0.3  # lattice parameter
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    # FCC: (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
                    for dx, dy, dz in [(0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)]:
                        x = i*a + dx
                        y = j*a + dy
                        z = k*a + dz
                        if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                            pts.append((x, y, z))
        pts = np.array(pts)
        if len(pts) >= N:
            # Select N points with maximum minimum distance (greedy)
            selected = [pts[0]]
            for _ in range(N - 1):
                best_pt = None
                best_min_dist = -1
                for p in pts:
                    if p in selected:
                        continue
                    min_d = np.min([np.linalg.norm(p - s) for s in selected])
                    if min_d > best_min_dist:
                        best_min_dist = min_d
                        best_pt = p
                selected.append(best_pt)
            return np.array(selected)
        return None
    
    # Run optimizations
    for method, options in strategies:
        if time.time() > deadline:
            break
        
        # Try with grid initialization
        try:
            result = minimize(_objective, x0, method=method, bounds=bounds, options=options)
            if result.fun < -best_score:  # Negative objective, so lower is better
                best_x = result.x.copy()
                best_score = -result.fun
        except Exception:
            pass
        
        # Try with FCC initialization
        fcc_pts = _fcc_init()
        if fcc_pts is not None and time.time() < deadline:
            x0_fcc = fcc_pts.flatten()
            try:
                result = minimize(_objective, x0_fcc, method=method, bounds=bounds, options=options)
                if result.fun < -best_score:
                    best_x = result.x.copy()
                    best_score = -result.fun
            except Exception:
                pass
    
    # Local refinement with more iterations if time allows
    if time.time() < deadline - 5:
        try:
            result = minimize(_objective, best_x, method="L-BFGS-B", bounds=bounds, 
                            options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-15})
            if result.fun < -best_score:
                best_x = result.x.copy()
                best_score = -result.fun
        except Exception:
            pass
    
    centers = best_x.reshape(-1, 3)
    return centers

# EVOLVE-BLOCK-END
