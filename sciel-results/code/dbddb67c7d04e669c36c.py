import numpy as np
from scipy.optimize import differential_evolution, minimize
import math
import time

N = 100

def construct():
    # Fixed point at North Pole
    fixed_point = np.array([0.0, 0.0, 1.0])
    n_opt = N - 1
    
    def get_points(params):
        thetas = params[:n_opt]
        phis = params[n_opt:]
        
        x = np.sin(thetas) * np.cos(phis)
        y = np.sin(thetas) * np.sin(phis)
        z = np.cos(thetas)
        
        opt_points = np.column_stack([x, y, z])
        return np.vstack([fixed_point, opt_points])

    def energy(params):
        points = get_points(params)
        # Pairwise distances using vectorized computation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        if np.any(pair_dists < 1e-12):
            return 1e10
            
        return -np.sum(np.log(pair_dists))

    def floyd_repulsion(n_iter=2000):
        # Initialize with Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Fix first point at North Pole
        p = pts.copy()
        p[0] = fixed_point
        
        for k in range(n_iter):
            # Calculate forces vectorized
            # Force on point i is sum over j != i of (p_j - p_i) / ||p_i - p_j||^2
            diff = p[np.newaxis, :, :] - p[:, np.newaxis, :]  # p_j - p_i for each i
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            # forces[i] = sum_j (p_j - p_i) / dist^2
            forces = np.sum(inv_dist_sq[:, np.newaxis] * diff, axis=1)
            
            # Don't move the fixed point
            forces[0] = 0.0
            
            # Update with annealing step
            step = 0.01 * (1.0 - k / n_iter) ** 0.5
            p[1:] += step * forces[1:]
            
            # Normalize all points
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            p[0] = fixed_point
            
        # Convert final p[1:] to params
        opt_pts = p[1:]
        thetas = np.arccos(np.clip(opt_pts[:, 2], -1.0, 1.0))
        phis = np.arctan2(opt_pts[:, 1], opt_pts[:, 0])
        phis = phis % (2 * np.pi)
        return np.concatenate([thetas, phis])

    # 1. Generate candidates
    candidates = []
    
    # Candidate 1: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    opt_pts = pts[1:]
    thetas = np.arccos(np.clip(opt_pts[:, 2], -1.0, 1.0))
    phis = np.arctan2(opt_pts[:, 1], opt_pts[:, 0])
    phis = phis % (2 * np.pi)
    candidates.append(np.concatenate([thetas, phis]))
    
    # Candidate 2: Floyd Repulsion (vectorized)
    try:
        candidates.append(floyd_repulsion(1000))
    except Exception:
        pass
        
    # Candidate 3: Random
    np.random.seed(42)
    thetas = np.random.uniform(0.05, np.pi - 0.05, n_opt)
    phis = np.random.uniform(0, 2*np.pi, n_opt)
    candidates.append(np.concatenate([thetas, phis]))

    # 2. Refine with L-BFGS-B
    bounds = [(0.001, np.pi - 0.001)] * n_opt + [(0.0, 2.0 * np.pi)] * n_opt
    
    best_params = None
    best_energy = float('inf')
    
    for c in candidates:
        try:
            res = minimize(energy, c, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 3. Run DE with better settings
    try:
        res_de = differential_evolution(
            energy, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-10,
            init='latinhypercube',
            polish=False
        )
        if res_de.fun < best_energy:
            best_energy = res_de.fun
            best_params = res_de.x
            # Polish DE result
            res_polish = minimize(energy, best_params, method='L-BFGS-B', bounds=bounds, 
                                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res_polish.fun < best_energy:
                best_energy = res_polish.fun
                best_params = res_polish.x
    except Exception:
        pass

    # 4. Final multi-start polishing
    for _ in range(3):
        try:
            # Add small perturbation and re-optimize
            perturbation = np.random.normal(0, 0.01, best_params.shape)
            perturbed = best_params + perturbation
            # Clip to valid range
            perturbed[:n_opt] = np.clip(perturbed[:n_opt], 0.001, np.pi - 0.001)
            perturbed[n_opt:] = perturbed[n_opt:] % (2 * np.pi)
            
            res = minimize(energy, perturbed, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    final_points = get_points(best_params)
    norms = np.linalg.norm(final_points, axis=1, keepdims=True)
    final_points = final_points / norms
    
    # Final check
    diff = final_points[:, np.newaxis, :] - final_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    if np.any(pair_dists < 1e-9) or not np.all(np.isfinite(final_points)):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return final_points
