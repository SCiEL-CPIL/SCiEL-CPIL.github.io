import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Exclude self-distances
    iu = np.triu_indices(N, k=1)
    return dists[iu]

def objective(x):
    """(max_dist / min_dist)^2"""
    pts = x.reshape(N, D)
    d = pairwise_distances(pts)
    # Handle degenerate cases
    if np.any(d < 1e-12):
        return 1e10
    ratio = d.max() / d.min()
    return ratio**2

def objective_smooth(x, beta=100.0):
    """Smooth approximation using log-sum-exp."""
    pts = x.reshape(N, D)
    d = pairwise_distances(pts)
    if np.any(d < 1e-12):
        return 1e10
    # log-sum-exp for max
    max_d = np.log(np.sum(np.exp(beta * d))) / beta
    # For min, use negative log-sum-exp: min = -logsumexp(-beta*d)/beta
    min_d = -np.log(np.sum(np.exp(-beta * d))) / beta
    if min_d < 1e-12:
        return 1e10
    ratio = max_d / min_d
    return ratio**2

def make_seeds():
    """Generate multiple diverse initial configurations."""
    seeds = []
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid.flatten())
    
    # 2. Hexagonal lattice subset
    # Hex lattice: rows offset
    hex_pts = []
    for row in range(4):
        x_offset = (row % 2) * 0.5
        for col in range(4):
            hex_pts.append([col + x_offset, row * np.sqrt(3)/2])
    hex_pts = np.array(hex_pts[:16])
    seeds.append(hex_pts.flatten())
    
    # 3. Concentric circles: 1 center + 5 + 10
    pts = [np.array([0.0, 0.0])]
    # Inner ring: 5 points
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    # Outer ring: 10 points
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 4. Two concentric octagons
    pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 5. 1 center + 6 + 9 (hexagon + outer)
    pts = [np.array([0.0, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 6. 2 center + 6 + 8
    pts = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.8 * np.cos(angle), 2.8 * np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 7. Regular 16-gon (degenerate but a seed)
    pts = []
    for i in range(16):
        angle = 2 * np.pi * i / 16
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    seeds.append(np.array(pts).flatten())
    
    # 8. 4x4 grid with slight perturbation
    grid2 = np.array([(i + 0.1*(i%2), j + 0.1*(j%2)) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid2.flatten())
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    seeds = make_seeds()
    best_x = None
    best_val = np.inf
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        
        # Try Nelder-Mead
        try:
            res = minimize(
                objective,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Try Powell
        if time.time() > deadline:
            break
        try:
            res = minimize(
                objective,
                seed,
                method='Powell',
                options={'maxiter': 3000, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback to grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return best_x.reshape(N, D)
