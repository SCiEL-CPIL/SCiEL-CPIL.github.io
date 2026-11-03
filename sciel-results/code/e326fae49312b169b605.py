"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _distance_matrix(pts):
    """Compute pairwise distances between points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _compute_radius(pts):
    """Compute the maximum radius for given centers."""
    # Minimum pairwise distance
    dm = _distance_matrix(pts)
    # Exclude diagonal
    np.fill_diagonal(dm, np.inf)
    min_pair = np.min(dm)
    
    # Distance to boundary: for square [-0.5, 0.5]^2
    # For each point, min distance to any of the 4 sides
    dist_to_bounds = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_bound = np.min(dist_to_bounds)
    
    return min(min_pair, 2 * min_bound) / 2

def _objective(pts, N=N):
    """Objective to minimize: negative of the radius."""
    # Check validity
    if not np.all(np.isfinite(pts)):
        return 1e6
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        # Points outside square - heavy penalty
        outside = (np.abs(pts[:, 0]) > 0.5) | (np.abs(pts[:, 1]) > 0.5)
        penalty = np.sum(np.maximum(0, np.abs(pts[:, 0]) - 0.5 + 1e-4)) + \
                  np.sum(np.maximum(0, np.abs(pts[:, 1]) - 0.5 + 1e-4))
        return 1e6 + penalty
    
    r = _compute_radius(pts)
    return -r

def _hexagonal_initialization(n_rows=None):
    """
    Create a hexagonal lattice initialization.
    If n_rows is None, estimate based on N.
    """
    if n_rows is None:
        # Approximate number of rows for a hexagonal packing of N circles in a square
        # Area of square = 1. Area of one circle approx = pi*r^2.
        # But we are packing centers.
        # Hexagonal density is ~0.9069.
        # Let's just try a few reasonable row counts.
        # For N=200, sqrt(200) is ~14.
        # Hexagonal rows alternate in length.
        # Let's aim for roughly 14-16 rows.
        n_rows = 14

    # Generate a hexagonal grid
    # Row i has either k or k+1 points
    # Let's generate a large enough hexagonal grid and pick the best subset or scale it.
    
    # Alternative: Generate points in a hexagonal pattern scaled to fit [-0.5, 0.5]
    # Let's assume a horizontal hexagonal packing.
    # Spacing dx = 1, dy = sqrt(3)/2.
    # We need to fit N points.
    
    # Strategy: Generate a hexagonal grid with a specific 'unit' size, 
    # then scale the whole thing to fit the square.
    
    # Let's try to construct a grid that is roughly square in extent.
    # Number of points in a hexagonal triangle of side n is n(n+1)/2.
    # For 200, n is around 20 (20*21/2 = 210).
    # But a triangle doesn't fit a square well.
    
    # Let's use a rectangular patch of hexagonal lattice.
    # Rows: n_rows
    # Cols: n_cols
    # Total points approx n_rows * n_cols.
    n_cols = int(np.ceil(N / n_rows))
    
    # Generate points
    pts = []
    # Hexagonal lattice vectors
    # a1 = (1, 0)
    # a2 = (0.5, sqrt(3)/2)
    # We want to fit this into a square.
    # Let's generate a grid of size n_rows x n_cols
    # and then scale it.
    
    # To make it fit better, we can offset every other row.
    for i in range(n_rows):
        for j in range(n_cols):
            if len(pts) >= N:
                break
            # Standard hexagonal coordinates
            x = j + 0.5 * (i % 2)
            y = i * np.sqrt(3) / 2
            pts.append([x, y])
            
    pts = np.array(pts[:N])
    
    # Center the points
    center = np.mean(pts, axis=0)
    pts = pts - center
    
    # Scale to fit the square [-0.5, 0.5]
    # Find the max extent
    max_ext = np.max(np.abs(pts))
    if max_ext > 0:
        # We want max extent to be 0.5 * factor.
        # If we scale by 0.5 / max_ext, the furthest point is at 0.5.
        # But we need to ensure it fits.
        scale = 0.5 / max_ext
        pts = pts * scale
        
    # Clip to ensure validity (though scaling should mostly do it)
    pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def _random_initialization():
    """Random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _jitter(pts, sigma=0.01):
    """Add small random jitter to points."""
    return pts + np.random.normal(0, sigma, size=pts.shape)

def _local_optimize(pts, max_iter=5000, tol=1e-8):
    """Perform local optimization using L-BFGS-B."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        pts_x = x.reshape(-1, 2)
        return _objective(pts_x)
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6}
        )
        if result.success or result.status == 1:
            best_pts = result.x.reshape(-1, 2)
            best_pts = np.clip(best_pts, -0.5, 0.5)
            return best_pts, -_objective(best_pts)
        else:
            return pts, _compute_radius(pts)
    except Exception:
        return pts, _compute_radius(pts)

def _hill_climbing(pts, n_iterations=100, step_size=0.005):
    """Simple hill climbing with random moves."""
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    
    for _ in range(n_iterations):
        idx = np.random.randint(0, N)
        move = np.random.normal(0, step_size, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] = np.clip(new_pts[idx] + move, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
    
    return current_pts, current_r

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Hexagonal initialization with different row counts
    # This is much better than a square grid for circle packing.
    for n_rows in [12, 13, 14, 15, 16, 17, 18]:
        if time.time() - start_time > time_limit - 60:
            break
        pts = _hexagonal_initialization(n_rows)
        if len(pts) != N:
            continue
        
        # Quick relaxation to fix any overlaps or boundary issues from scaling
        # A few iterations of gradient ascent on radius (descent on objective)
        # Or just use L-BFGS with low iterations first
        opt_pts, opt_r = _local_optimize(pts, max_iter=500, tol=1e-6)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Strategy 2: If we have a good candidate, do a deep optimization
    if best_pts is not None:
        if time.time() - start_time < time_limit - 30:
            # Add a small jitter to escape local minima
            jittered_pts = _jitter(best_pts, sigma=0.005)
            opt_pts, opt_r = _local_optimize(jittered_pts, max_iter=5000, tol=1e-8)
            
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
                
    # Strategy 3: Random restarts if time remains
    remaining_time = time_limit - (time.time() - start_time)
    n_restarts = int(remaining_time / 5) # Assume 5s per restart
    for i in range(min(n_restarts, 10)):
        if time.time() - start_time > time_limit - 10:
            break
        
        pts = _random_initialization()
        # Quick optimize
        opt_pts, opt_r = _local_optimize(pts, max_iter=1000, tol=1e-6)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Strategy 4: Hill climbing on the best solution
    if best_pts is not None and time.time() - start_time < time_limit - 5:
        hc_pts, hc_r = _hill_climbing(best_pts, n_iterations=200, step_size=0.002)
        if hc_r > best_r:
            best_r = hc_r
            best_pts = hc_pts.copy()
            
    # Final check
    if best_pts is None:
        best_pts = _hexagonal_initialization(14)
        
    best_pts = np.clip(best_pts, -0.5, 0.5)
    return best_pts

# EVOLVE-BLOCK-END
