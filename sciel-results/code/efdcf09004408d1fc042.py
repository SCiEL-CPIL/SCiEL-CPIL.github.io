"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time


def _compute_radius(pts):
    """Compute the minimum safe radius for a set of centers."""
    if len(pts) < 2:
        return 0.5
    tree = cKDTree(pts)
    # Find all pairs within a reasonable distance (2*0.12 = 0.24)
    pairs = tree.query_pairs(0.3)
    if len(pairs) == 0:
        min_dist = 1.0
    else:
        min_dist = float('inf')
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    # Boundary constraint
    boundary = np.min(0.5 - np.abs(pts))
    r = min(min_dist / 2.0, boundary)
    return r


def _generate_bcc_points():
    """Generate BCC (body-centered cubic) lattice points in [-0.5, 0.5]^3."""
    # BCC has 2 points per unit cell: corners + center
    # For a cube of side 1, we need to figure out lattice spacing
    # Let's use a grid approach: place points in a BCC arrangement
    # 
    # Strategy: Use an m x m x m grid of "cells", each cell has 8 corners (shared) and 1 center
    # Actually, let's think differently.
    #
    # BCC lattice: points at (i, j, k) and (i+0.5, j+0.5, k+0.5) for integer i,j,k
    # scaled to fit in [-0.5, 0.5]^3
    #
    # Let's try: generate a dense grid, then select N points with max min-distance
    #
    # Alternative: Start with a simple cubic grid and optimize
    #
    # Let's generate points on a body-centered cubic lattice
    # For a cube [0,1]^3, BCC points are at:
    # (i*L, j*L, k*L) and (i*L+L/2, j*L+L/2, k*L+L/2)
    # where L is the lattice constant
    #
    # Number of points in [0,1]^3: roughly 2/L^3
    # For N=100: 2/L^3 ≈ 100, so L ≈ 0.27
    
    # Let's just generate a fine grid and use it as initial points
    m = 5  # 5x5x5 = 125 points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add BCC-like interior points
    # For each 1x1x1 sub-cell, add center
    # Actually, let's just use the grid and optimize
    
    # Try different starting configurations
    # 1. Simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    
    return grid


def _local_optimize(pts, max_iters=5000, time_budget=None, start_time=None):
    """Local optimization using gradient-free perturbation."""
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    step_size = 0.01
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        # Time check
        if time_budget is not None and (time.time() - start_time) > time_budget:
            break
        
        # Try random perturbations
        improved = False
        for _ in range(50):
            if time_budget is not None and (time.time() - start_time) > time_budget:
                break
            
            trial_pts = best_pts.copy()
            idx = rng.integers(0, N)
            delta = rng.normal(0, step_size, size=3)
            trial_pts[idx] = np.clip(trial_pts[idx] + delta, -0.5, 0.5)
            
            r = _compute_radius(trial_pts)
            if r > best_r + 1e-12:
                best_pts = trial_pts
                best_r = r
                improved = True
        
        # Adapt step size
        if improved:
            step_size *= 1.1
        else:
            step_size *= 0.8
        
        if step_size < 1e-6:
            break
    
    return best_pts, best_r


def _simulated_annealing(pts, time_budget=100, start_time=None):
    """Simulated annealing optimization."""
    if start_time is None:
        start_time = time.time()
    
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    T = 0.005  # Initial temperature
    T_min = 1e-8
    cooling = 0.995
    
    rng = np.random.default_rng(123)
    
    while T > T_min:
        # Time check
        if (time.time() - start_time) > (time_budget - 5):
            break
        
        # Generate multiple trials per temperature
        n_trials = 200
        for _ in range(n_trials):
            if (time.time() - start_time) > (time_budget - 5):
                break
            
            trial_pts = current_pts.copy()
            # Perturb 1-3 points
            n_perturb = rng.integers(1, 4)
            idxs = rng.choice(N, size=n_perturb, replace=False)
            for idx in idxs:
                delta = rng.normal(0, T, size=3)
                trial_pts[idx] = np.clip(trial_pts[idx] + delta, -0.5, 0.5)
            
            trial_r = _compute_radius(trial_pts)
            delta_r = trial_r - current_r
            
            if delta_r > 0 or rng.random() < np.exp(delta_r / T):
                current_pts = trial_pts
                current_r = trial_r
                if current_r > best_r:
                    best_pts = current_pts.copy()
                    best_r = current_r
        
        T *= cooling
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin for safety
    
    # Generate initial points
    pts = _generate_bcc_points()
    
    # Phase 1: Local optimization
    pts, r = _local_optimize(pts, max_iters=2000, time_budget=20, start_time=start_time)
    
    # Phase 2: Simulated annealing with multiple restarts
    best_pts = pts.copy()
    best_r = r
    
    remaining_time = time_limit - (time.time() - start_time)
    
    # Try SA from current best
    if remaining_time > 30:
        sa_pts, sa_r = _simulated_annealing(pts, time_budget=min(remaining_time - 5, 60), start_time=start_time)
        if sa_r > best_r:
            best_pts = sa_pts
            best_r = sa_r
    
    # Try SA from different random starts
    remaining_time = time_limit - (time.time() - start_time)
    rng = np.random.default_rng(789)
    
    n_restarts = min(5, int(remaining_time // 15))
    for i in range(n_restarts):
        if (time.time() - start_time) > (time_limit - 15):
            break
        
        # Random initial configuration
        rand_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
        
        # Quick local optimization
        opt_pts, opt_r = _local_optimize(rand_pts, max_iters=500, time_budget=5, start_time=start_time)
        
        if opt_r > best_r:
            best_pts = opt_pts
            best_r = opt_r
        
        # SA from this start
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 20:
            sa_pts, sa_r = _simulated_annealing(opt_pts, time_budget=min(remaining_time - 10, 30), start_time=start_time)
            if sa_r > best_r:
                best_pts = sa_pts
                best_r = sa_r
    
    # Final local polish
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        best_pts, best_r = _local_optimize(best_pts, max_iters=1000, time_budget=min(remaining_time - 2, 10), start_time=start_time)
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    # Final validation
    r_final = _compute_radius(best_pts)
    if not np.isfinite(r_final) or r_final <= 0:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
