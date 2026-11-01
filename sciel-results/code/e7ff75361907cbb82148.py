import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    spacing = 0.25
    points = []
    max_k = int(1.0 / spacing) + 3
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append((x, y))
    
    if len(points) < N:
        # Fallback if spacing too large
        spacing = 0.2
        points = []
        max_k = int(1.0 / spacing) + 3
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    points.append((x, y))

    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii, kind='stable')
    x0 = points[indices[:N]]
    x_flat = x0.flatten()
    
    # Pre-compute indices for pairwise distances
    i_idx, j_idx = np.triu_indices(N, 1)
    
    def smooth_min_radius(x, beta):
        x = x.reshape(N, 2)
        # Pairwise distances
        diffs = x[i_idx] - x[j_idx]
        dists = np.linalg.norm(diffs, axis=1) / 2.0
        
        # Boundary distances
        radii = np.linalg.norm(x, axis=1)
        boundary_dists = 1.0 - radii
        
        all_vals = np.concatenate([dists, boundary_dists])
        
        s = np.min(all_vals)
        shifted = all_vals - s
        exp_terms = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            return -s
            
        soft_min = s - (1.0 / beta) * np.log(sum_exp)
        return -soft_min
        
    beta = 10.0
    current_x = x_flat.copy()
    best_radius = -np.inf
    best_x = current_x.copy()
    
    # Optimization Loop
    while time.time() < deadline:
        def obj(x):
            return smooth_min_radius(x, beta)
        
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            result = minimize(obj, current_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            if result.success:
                current_x = result.x
            else:
                # If it didn't converge, maybe we should break or continue?
                # L-BFGS-B usually improves even if not fully converged
                current_x = result.x
        except Exception:
            break
            
        # Calculate actual radius
        x_curr = current_x.reshape(N, 2)
        diffs = x_curr[i_idx] - x_curr[j_idx]
        dists = np.linalg.norm(diffs, axis=1) / 2.0
        radii = np.linalg.norm(x_curr, axis=1)
        boundary_dists = 1.0 - radii
        
        min_val = min(np.min(dists), np.min(boundary_dists))
        
        if min_val > best_radius:
            best_radius = min_val
            best_x = current_x.copy()
            
        # Increase beta
        beta *= 2.0
        if beta > 2000:
            break
            
    return best_x.reshape(N, 2)
