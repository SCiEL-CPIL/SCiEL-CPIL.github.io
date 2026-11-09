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

def soft_min_score(centers, beta=50.0):
    """Smooth approximation of the min-constraint score using log-sum-exp."""
    c = centers
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    # Pairwise distances / 2
    pw_dists = np.sqrt(dists_sq) / 2.0
    mask = ~np.eye(N, dtype=bool)
    pw_flat = pw_dists[mask]
    # Boundary distances
    bd = 0.5 - np.abs(c)
    bd_flat = np.min(bd, axis=1)
    all_constraints = np.concatenate([pw_flat, bd_flat])
    m = np.min(all_constraints)
    # Soft min: -log(sum(exp(-beta*(x - m))))/beta + m
    log_sum = np.log(np.sum(np.exp(-beta * (all_constraints - m))))
    soft = m - log_sum / beta
    return soft

def neg_soft_score(x, beta=50.0):
    """Negative soft score for minimization."""
    c = x.reshape(-1, 3)
    return -soft_min_score(c, beta)

def optimize_repulsion(centers, max_iter=1000, base_lr=0.05, decay=0.999, deadline=None):
    """Fast vectorized repulsion optimization."""
    centers = centers.copy().astype(float)
    lr = base_lr
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
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

def coordinate_descent_polish(centers, max_iter=500, deadline=None):
    """Coordinate descent to polish the solution."""
    centers = centers.copy().astype(float)
    best_r = compute_score(centers)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for k in range(3):
                # Try small perturbations in coordinate k for center i
                orig_val = centers[i, k]
                best_val = orig_val
                best_local_r = best_r
                
                # Adaptive step sizes
                for step in [0.001, 0.0005, 0.0001, 0.00005, -0.001, -0.0005, -0.0001, -0.00005]:
                    trial_val = orig_val + step
                    if trial_val < -0.5 or trial_val > 0.5:
                        continue
                    centers[i, k] = trial_val
                    r = compute_score(centers)
                    if r > best_local_r + 1e-15:
                        best_local_r = r
                        best_val = trial_val
                centers[i, k] = best_val
                if best_val != orig_val:
                    improved = True
                    best_r = best_local_r
        
        if not improved:
            break
    
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
    for seed in range(15):
        rng = np.random.RandomState(seed)
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # 3. Jittered grids with different jitter
    for jit in [0.02, 0.05, 0.1]:
        rng = np.random.RandomState(100 + int(jit*100))
        jittered = grid[:N].copy() + rng.uniform(-jit, jit, size=(N, 3))
        inits.append(np.clip(jittered, -0.49, 0.49))
    
    # Optimize each init
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Repulsion-based optimization
        result = optimize_repulsion(init, max_iter=600, base_lr=0.05, decay=0.9995, deadline=deadline - 25)
        r = compute_score(result)
        
        if r > best_r:
            best_r = r
            best_centers = result.copy()
        
        # Phase 2: Scipy polish with soft-min objective
        if time.time() < deadline - 20 and r > 0.12:
            try:
                res = minimize(neg_soft_score, result.flatten(), method='L-BFGS-B',
                              bounds=[(-0.5, 0.5)] * (N * 3),
                              options={'maxiter': 150, 'ftol': 1e-16, 'gtol': 1e-10})
                polished = res.x.reshape(-1, 3)
                # Ensure within bounds
                polished = np.clip(polished, -0.5 + 1e-12, 0.5 - 1e-12)
                r_polished = compute_score(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished.copy()
            except Exception:
                pass
        
        # Phase 3: Coordinate descent polish
        if time.time() < deadline - 10 and best_r > 0.12:
            try:
                cd_result = coordinate_descent_polish(best_centers, max_iter=200, deadline=deadline - 5)
                r_cd = compute_score(cd_result)
                if r_cd > best_r:
                    best_r = r_cd
                    best_centers = cd_result.copy()
            except Exception:
                pass
    
    # Final safety: ensure we have a valid result
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        # Fallback to simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
    
    return best_centers
