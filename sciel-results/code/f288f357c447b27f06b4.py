"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.spatial import cKDTree
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _compute_radius(pts):
    """Compute the maximum radius for given centers using cKDTree for efficiency."""
    if len(pts) == 0:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Distance to boundary: for square [-0.5, 0.5]^2
    dist_to_bounds = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_bound = np.min(dist_to_bounds)
    
    # Minimum pairwise distance using cKDTree
    if len(pts) > 1:
        tree = cKDTree(pts)
        # Query nearest neighbors (k=2 to get one neighbor + self)
        dists, _ = tree.query(pts, k=2)
        min_pair = np.min(dists[:, 1])
    else:
        min_pair = np.inf
    
    return min(min_pair, 2 * min_bound) / 2

def _repulsion_optimize(pts, n_iter=500, time_limit=5.0):
    """Iterative repulsion-based local optimization."""
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    start_time = time.time()
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces: push points apart
        tree = cKDTree(current_pts)
        dists, idxs = tree.query(current_pts, k=3)
        
        forces = np.zeros_like(current_pts)
        for i in range(N):
            for k in range(1, 3):  # Consider 2 nearest neighbors
                j = idxs[i, k]
                if j == i:
                    continue
                d = dists[i, k]
                if d < 1e-10:
                    # Coincident points - push randomly
                    forces[i] += np.random.normal(0, 0.001, size=2)
                    continue
                # Repulsion force proportional to 1/d^2
                force_mag = 0.001 / (d * d)
                direction = (current_pts[i] - current_pts[j]) / d
                forces[i] += force_mag * direction
        
        # Add boundary attraction
        for i in range(N):
            # Push back if too close to boundary
            for coord in range(2):
                if current_pts[i, coord] > 0.5 - current_r:
                    forces[i, coord] -= 0.001 * (current_pts[i, coord] - (0.5 - current_r))
                elif current_pts[i, coord] < -0.5 + current_r:
                    forces[i, coord] += 0.001 * ((-0.5 + current_r) - current_pts[i, coord])
        
        # Apply forces with adaptive step size
        step_size = 0.01 / (1 + it * 0.01)
        new_pts = current_pts + forces * step_size
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
        else:
            # Try smaller step
            new_pts = current_pts + forces * step_size * 0.1
            new_pts = np.clip(new_pts, -0.5, 0.5)
            new_r = _compute_radius(new_pts)
            if new_r > current_r:
                current_pts = new_pts
                current_r = new_r
    
    return current_pts, current_r

def _hexagonal_initialization():
    """Create a hexagonal packing initialization."""
    pts = []
    # Hexagonal grid: rows offset by half
    row_height = np.sqrt(3) / 2
    # Estimate number of rows
    n_rows = int(np.sqrt(N / 1.5)) + 1
    
    for i in range(n_rows):
        offset = (i % 2) * 0.5
        for j in range(int(np.sqrt(N)) + 1):
            if len(pts) >= N:
                break
            x = (j + offset) / (np.sqrt(N) + 1) - 0.5
            y = (i + 0.5) / (n_rows + 1) - 0.5
            pts.append([x, y])
    
    pts = np.array(pts[:N])
    return pts

def _grid_initialization(m, n):
    """Create a grid initialization with m columns and n rows."""
    pts = []
    for i in range(n):
        for j in range(m):
            if len(pts) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

def _random_initialization():
    """Random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _perturb(pts, sigma=0.02):
    """Add random perturbation to points."""
    return np.clip(pts + np.random.normal(0, sigma, size=pts.shape), -0.5, 0.5)

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 115  # Leave margin for safety
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Try multiple initializations
    initializations = []
    
    # Hexagonal
    hex_pts = _hexagonal_initialization()
    initializations.append(hex_pts)
    
    # Various grids
    for m, n in [(14, 15), (15, 14), (13, 16), (16, 13), (14, 14), (15, 15)]:
        if m * n >= N:
            grid_pts = _grid_initialization(m, n)
            initializations.append(grid_pts)
    
    # Random
    initializations.append(_random_initialization())
    
    # Evaluate and optimize each initialization
    for init_pts in initializations:
        if time.time() - start_time > time_limit - 10:
            break
        
        if len(init_pts) != N:
            continue
        
        init_r = _compute_radius(init_pts)
        if init_r > best_r:
            best_r = init_r
            best_pts = init_pts.copy()
        
        # Local optimization with time budget
        remaining_time = time_limit - (time.time() - start_time)
        opt_time = min(8.0, remaining_time * 0.6)
        
        if opt_time > 1.0:
            opt_pts, opt_r = _repulsion_optimize(init_pts, n_iter=300, time_limit=opt_time)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts.copy()
    
    # Strategy 2: Multi-start refinement from best solution
    if best_pts is not None and time.time() - start_time < time_limit - 15:
        for restart in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            
            perturbed = _perturb(best_pts, sigma=0.01 + restart * 0.005)
            remaining_time = time_limit - (time.time() - start_time)
            opt_time = min(6.0, remaining_time * 0.5)
            
            if opt_time > 0.5:
                opt_pts, opt_r = _repulsion_optimize(perturbed, n_iter=200, time_limit=opt_time)
                if opt_r > best_r:
                    best_r = opt_r
                    best_pts = opt_pts.copy()
    
    # Strategy 3: Final fine-tuning
    if best_pts is not None and time.time() - start_time < time_limit - 3:
        remaining_time = time_limit - (time.time() - start_time)
        opt_time = min(4.0, remaining_time * 0.8)
        
        if opt_time > 0.5:
            final_pts, final_r = _repulsion_optimize(best_pts, n_iter=100, time_limit=opt_time)
            if final_r > best_r:
                best_r = final_r
                best_pts = final_pts
    
    # Ensure valid output
    if best_pts is None:
        # Fallback: simple grid
        best_pts = _grid_initialization(14, 15)
    
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    # Ensure finite values
    best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=0.5, neginf=-0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
