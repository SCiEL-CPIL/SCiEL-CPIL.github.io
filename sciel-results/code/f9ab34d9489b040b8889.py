"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _clearance_sum(pos):
    """
    Objective function for optimization.
    Minimizes negative sum of clearances.
    pos: shape (2*N,)
    """
    p = pos.reshape(N, 2)
    # Calculate pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    # Minimum distance to any other point
    min_dist_to_others = np.min(dist, axis=1)
    
    # Distance to box boundaries
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    # Clearance for each point
    # r_i <= 0.5 * dist(i, j) ensures r_i + r_j <= dist(i, j)
    clearances = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    
    return -np.sum(clearances)

def _get_valid_packing(pos):
    """
    Convert optimized positions to a valid packing with radii.
    Returns (positions (N,2), radii (N,))
    """
    p = pos.reshape(N, 2).copy()
    # Clamp to unit square
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    
    # Calculate radii based on clearances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    
    # Apply safety factor to ensure strict non-overlap
    safety = 0.9999
    radii *= safety
    
    # Re-clamp positions to respect radii (in case radii became larger than distance to edge due to floating point)
    # Actually, radii is derived from distance to edge, so p[i] >= radii[i] should hold.
    # But if p[i] was 0.0 and radii was 0.0, it's fine.
    # If p[i] was 0.0001 and radii was 0.0001 * 0.9999, it's fine.
    
    return p, radii

def _hex_initialization():
    """Generate a hexagonal grid initialization."""
    pts = []
    # Use a grid density that provides at least N points
    # 6x6 = 36 points
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            # Stagger rows
            if i % 2 == 0:
                y += 0.5 / 6.0 * 0.8
            # Ensure within bounds
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    # If we have more than N, take the first N (or center them better?)
    # For now, just take first N
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    return np.array(pts[:N]).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # List of initializations
    inits = [
        _hex_initialization(),
        np.random.rand(2 * N),
        np.random.rand(2 * N),
        np.random.rand(2 * N),
        np.random.rand(2 * N)
    ]
    
    for init_pos in inits:
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Use L-BFGS-B for local optimization
            res = minimize(_clearance_sum, init_pos, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12})
            
            # Evaluate the result
            if res.fun < best_score or best_result is None:
                pos, radii = _get_valid_packing(res.x)
                score = np.sum(radii)
                
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((pos, radii))
                    
        except Exception:
            continue
            
    # Fallback if optimization failed
    if best_result is None:
        pos, radii = _get_valid_packing(inits[0])
        best_result = np.column_stack((pos, radii))
        
    return best_result

# EVOLVE-BLOCK-END
