import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(positions):
    """
    Compute the score (radius) for a given set of positions.
    positions: (N, 3) array
    """
    # Distance between all pairs
    # Using broadcasting for efficiency
    # diff: (N, N, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Set diagonal to infinity so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # positions are in [-0.5, 0.5]
    # Distance to nearest wall for each point
    # For each coordinate, distance to -0.5 or 0.5
    # min( x + 0.5, 0.5 - x )
    wall_dists = np.minimum(positions + 0.5, 0.5 - positions)
    min_wall_dist = np.min(wall_dists)
    
    # Radius is half of the minimum distance
    r = 0.5 * np.min([min_pair_dist, min_wall_dist])
    
    return r

def _neg_score(positions_flat):
    """
    Objective function for scipy.optimize.minimize.
    Takes flat array of positions, reshapes, and returns negative score.
    """
    positions = positions_flat.reshape((N, 3))
    # Check for invalid values
    if not np.all(np.isfinite(positions)):
        return 1e6
    return -_score(positions)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for safety
    
    # 1. Initialization
    # Use a jittered grid or Halton sequence
    # A 4x4x4 grid has 64 points. Let's use a 4x4x4 grid and take 50.
    # Or better, use a FCC-like initialization which is denser.
    
    # Let's try a simple random initialization with multiple restarts if time permits,
    # or a single high-quality structured init.
    
    # Halton sequence for low-discrepancy initialization
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def halton_sequence(count, dims):
        points = []
        for i in range(1, count + 1):
            points.append([halton(i, base) for base in [2, 3, 5]])
        return np.array(points)
    
    # Generate more points than needed and pick the best 50? 
    # Or just use 50 Halton points.
    # Halton points are well distributed but not optimal for packing.
    # Let's use a jittered 4x4x4 grid.
    
    grid_size = 4
    step = 1.0 / grid_size
    centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                x = -0.5 + (i + 0.5) * step
                y = -0.5 + (j + 0.5) * step
                z = -0.5 + (k + 0.5) * step
                centers.append([x, y, z])
                
    # We have 64 centers. We need 50.
    # Randomly select 50? Or take the first 50?
    # Taking the first 50 from a 4x4x4 grid might leave a "hole" at the end.
    # Let's shuffle and take 50.
    np.random.seed(42)
    idx = np.random.choice(len(centers), N, replace=False)
    init_positions = np.array(centers)[idx]
    
    # Add small jitter to break symmetry
    init_positions += np.random.uniform(-0.01, 0.01, size=init_positions.shape)
    
    # Clip to inside cube
    init_positions = np.clip(init_positions, -0.49, 0.49)
    
    x0 = init_positions.flatten()
    
    best_positions = init_positions
    best_score = _score(best_positions)
    
    # 2. Optimization with Nelder-Mead
    # Nelder-Mead parameters
    options = {
        'maxiter': 5000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'disp': False
    }
    
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead', options=options)
        
        if res.success or res.fun < -best_score:
            pos = res.x.reshape((N, 3))
            # Ensure validity
            if np.all(np.isfinite(pos)) and np.all(pos >= -0.5) and np.all(pos <= 0.5):
                s = _score(pos)
                if s > best_score:
                    best_score = s
                    best_positions = pos
    except Exception as e:
        pass
        
    if time.time() > deadline:
        return best_positions

    # 3. Polish phase
    # Run another optimization starting from the best found so far
    # Use tighter tolerances and maybe more iterations if time allows
    x0 = best_positions.flatten()
    
    options_polish = {
        'maxiter': 5000,
        'xatol': 1e-10,
        'fatol': 1e-12,
        'disp': False
    }
    
    try:
        res = minimize(_neg_score, x0, method='Nelder-Mead', options=options_polish)
        
        if res.success or res.fun < -best_score:
            pos = res.x.reshape((N, 3))
            if np.all(np.isfinite(pos)) and np.all(pos >= -0.5) and np.all(pos <= 0.5):
                s = _score(pos)
                if s > best_score:
                    best_score = s
                    best_positions = pos
    except Exception as e:
        pass

    return best_positions
