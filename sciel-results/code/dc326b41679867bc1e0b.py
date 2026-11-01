"""Improved packing: smooth surrogate optimization with soft-min for 50 spheres in unit cube."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    
    n = len(centers)
    if n < 2:
        return r_wall
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def _soft_min(values, beta=100.0):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values)
    m = np.min(values)
    return m - np.log(np.sum(np.exp(-beta * (values - m)))) / beta

def _soft_score(centers, beta=100.0):
    """Smooth surrogate for the radius objective."""
    centers = np.asarray(centers)
    wall_vals = 0.5 - np.abs(centers)  # shape (N, 3)
    r_wall_soft = _soft_min(wall_vals.ravel(), beta)
    
    n = len(centers)
    if n < 2:
        return r_wall_soft
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair_soft = _soft_min(dists.ravel(), beta) / 2.0
    
    return min(r_wall_soft, r_pair_soft)

def _initial_configuration():
    """Create a well-spread initial configuration using Fibonacci sphere-like distribution."""
    # Use a body-centered cubic lattice approach for better initial spread
    # 50 spheres: use a 5x5x2 arrangement with some offset
    positions = []
    # Layer 1: 5x5 grid
    for i in range(5):
        for j in range(5):
            x = -0.4 + i * 0.2
            y = -0.4 + j * 0.2
            z = -0.25
            positions.append([x, y, z])
    # Layer 2: 5x5 grid offset
    for i in range(5):
        for j in range(5):
            x = -0.3 + i * 0.2
            y = -0.3 + j * 0.2
            z = 0.25
            positions.append([x, y, z])
    
    return np.array(positions[:N])

def _local_optimize_smooth(centers, t_start, deadline):
    """Optimize using smooth surrogate with decreasing beta."""
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Start with moderate smoothness, decrease over iterations
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline - 5:
            break
        
        def objective(x):
            c = x.reshape(-1, 3)
            # Boundary penalty
            out_of_bounds = np.maximum(0, np.abs(c) - 0.499)
            penalty = 100.0 * np.sum(out_of_bounds ** 2)
            s = _soft_score(c, beta)
            return -s + penalty
        
        try:
            res = minimize(objective, best_centers.ravel(), method='L-BFGS-B',
                          bounds=[(-0.5, 0.5)] * 3 * N,
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            
            new_centers = res.x.reshape(-1, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
        except Exception:
            pass
    
    return best_centers, best_score

def _refine_pairwise(centers, t_start, deadline):
    """Refine by pushing apart the closest pairs."""
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    for iteration in range(50):
        if time.time() > deadline - 3:
            break
        
        # Find closest pair
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance pair
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Push them apart slightly
        direction = best_centers[i] - best_centers[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        
        # Try different push amounts
        improved = False
        for push in [0.005, 0.01, 0.02, 0.003, 0.008]:
            new_c = best_centers.copy()
            new_c[i] += direction * push / 2
            new_c[j] -= direction * push / 2
            
            # Check bounds
            if np.all(np.abs(new_c) <= 0.5):
                s = _score(new_c)
                if s > best_score:
                    best_centers = new_c
                    best_score = s
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 145  # Safety margin before 150s limit
    
    # Initial configuration
    centers = _initial_configuration()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Smooth surrogate optimization with decreasing beta
    best_centers, best_score = _local_optimize_smooth(best_centers, t_start, deadline)
    
    # Check time
    if time.time() > deadline:
        return best_centers
    
    # Pairwise refinement
    best_centers, best_score = _refine_pairwise(best_centers, t_start, deadline)
    
    # Final high-precision polish with very tight tolerances
    if time.time() < deadline - 5:
        def final_objective(x):
            c = x.reshape(-1, 3)
            out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
            penalty = 1000.0 * np.sum(out_of_bounds ** 2)
            s = _score(c)
            return -s + penalty
        
        try:
            res = minimize(final_objective, best_centers.ravel(), method='L-BFGS-B',
                          bounds=[(-0.5, 0.5)] * 3 * N,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            
            new_centers = res.x.reshape(-1, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for coord in range(3):
                    for delta in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002, 
                                 0.0002, -0.0002, 0.0008, -0.0008]:
                        new_c = best_centers.copy()
                        new_c[i, coord] += delta
                        if np.all(np.abs(new_c) <= 0.5):
                            s = _score(new_c)
                            if s > best_score:
                                best_centers = new_c
                                best_score = s
                                improved = True
    
    return best_centers
