import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def objective(coords):
    """
    Calculate the squared ratio of max to min pairwise distance.
    coords: (N, D) array
    """
    if coords.shape != (N, D):
        return 1e6
    
    # Compute all pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask out diagonal (self-distances)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    
    if len(dist_sq) == 0:
        return 1e6
        
    d_max_sq = np.max(dist_sq)
    d_min_sq = np.min(dist_sq)
    
    if d_min_sq < 1e-12:
        return 1e6
        
    ratio_sq = d_max_sq / d_min_sq
    return ratio_sq

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave margin for 120s/150s limit
    
    best_score = 1e6
    best_coords = None
    
    # Define initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    hex_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            x = q + 0.5 * r
            y = (np.sqrt(3) / 2) * r
            hex_points.append([x, y])
    hex_points = np.array(hex_points)
    hex_points -= np.mean(hex_points, axis=0)
    norm = np.max(np.linalg.norm(hex_points, axis=1))
    if norm > 0:
        hex_points /= norm
    # Take 16 points
    inits.append(hex_points[:16].copy())
    
    # Add jittered hex lattice
    np.random.seed(1)
    hex_jittered = hex_points[:16].copy()
    hex_jittered += np.random.normal(0, 0.05, hex_jittered.shape)
    inits.append(hex_jittered)
    
    # 2. Random points
    np.random.seed(42)
    for i in range(15):
        rand_points = np.random.rand(N, D) * 2 - 1
        rand_points -= np.mean(rand_points, axis=0)
        norm = np.max(np.linalg.norm(rand_points, axis=1))
        if norm > 0:
            rand_points /= norm
        inits.append(rand_points.copy())
        
    # 3. Square lattice
    side = int(np.ceil(np.sqrt(N)))
    grid = np.array(list(np.meshgrid(np.arange(side), np.arange(side)))).T.reshape(-1, 2)[:N]
    grid -= np.mean(grid, axis=0)
    norm = np.max(np.linalg.norm(grid, axis=1))
    if norm > 0:
        grid /= norm
    inits.append(grid.copy())
    
    # 4. Ring
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ring.copy())
    
    # Local search using scipy.optimize.minimize
    methods = ['Nelder-Mead', 'Powell']
    
    for init in inits:
        if time.time() > deadline:
            break
            
        for method in methods:
            if time.time() > deadline:
                break
                
            try:
                if method == 'Nelder-Mead':
                    opts = {'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7}
                else:
                    opts = {'maxiter': 200, 'maxfev': 2000}
                
                res = minimize(objective, init, method=method, options=opts)
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_coords = res.x
                    
            except Exception as e:
                pass
                
    # If no optimization succeeded, return the best initial
    if best_coords is None:
        best_coords = inits[0]
        best_score = objective(best_coords)
        
    return best_coords
