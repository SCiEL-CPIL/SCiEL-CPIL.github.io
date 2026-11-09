import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_score(centers):
    """Compute the radius for a given set of centers."""
    if centers.shape != (N, 3):
        return -np.inf
    if not np.all(np.isfinite(centers)):
        return -np.inf
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return -np.inf
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2
    boundary = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary)
    return min(min_pair, min_boundary)

def optimize_repulsion(centers, max_iter=1000, base_lr=0.05, decay=0.999):
    """Fast vectorized repulsion optimization."""
    centers = centers.copy().astype(float)
    lr = base_lr
    
    for it in range(max_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        boundary_dists = 0.5 - np.abs(centers)
        boundary_min = np.min(boundary_dists, axis=1)
        
        r_min = min(np.min(dists) / 2, np.min(boundary_min))
        if r_min <= 0:
            centers = np.clip(centers, -0.49, 0.49)
            continue
        
        # Pairwise forces (vectorized)
        push_mask = (dists < 2 * r_min) & ~np.eye(N, dtype=bool)
        mag_factor = np.zeros((N, N))
        mag_factor[push_mask] = (2 * r_min - dists[push_mask]) * 0.5 / dists[push_mask]
        forces = np.sum(mag_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        for i in range(N):
            for k in range(3):
                bd = boundary_dists[i, k]
                if bd < r_min:
                    sign = 1 if centers[i, k] > 0 else -1
                    forces[i, k] += sign * (r_min - bd)
        
        centers += lr * forces
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        lr *= decay
    
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_r = -np.inf
    best_centers = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Jittered 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].copy())
    
    # 2. Various random inits
    for seed in range(20):
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. Jittered grids with different jitter
    for jit in [0.02, 0.05, 0.1]:
        rng = np.random.RandomState(100 + int(jit*100))
        jittered = grid[:N].copy() + rng.uniform(-jit, jit, size=(N, 3))
        inits.append(np.clip(jittered, -0.49, 0.49))
    
    # Optimize each init
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        result = optimize_repulsion(init, max_iter=800, base_lr=0.05, decay=0.9995)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
        
        # If we have time, try scipy polish
        if time.time() < deadline - 20 and r > 0.13:
            try:
                def obj(x):
                    c = x.reshape(-1, 3)
                    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                    d = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(d, np.inf)
                    pw = d / 2
                    mask = ~np.eye(N, dtype=bool)
                    pw_flat = pw[mask]
                    bd = 0.5 - np.abs(c)
                    bd_flat = np.min(bd, axis=1)
                    all_c = np.concatenate([pw_flat, bd_flat])
                    m = np.min(all_c)
                    T = 20.0
                    return -(m - np.log(np.sum(np.exp(-T * (all_c - m)))) / T)
                
                res = minimize(obj, result.flatten(), method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-14})
                polished = res.x.reshape(-1, 3)
                r_polished = compute_score(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished.copy()
            except:
                pass
    
    return best_centers
