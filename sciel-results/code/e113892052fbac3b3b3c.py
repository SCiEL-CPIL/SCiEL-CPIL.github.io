"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: High-symmetry seeds (FCC, BCC) followed by fast custom gradient ascent
on a soft-min objective. This avoids the overhead of scipy.optimize.minimize
and allows for tighter control over the annealing schedule.
"""

import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

# Precompute indices for pairwise distances
_IU, _JU = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_IU)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_IU] - c[_JU]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp formulation.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    # Pair distances
    diff = c[_IU] - c[_JU]  # shape (M, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
    
    # Weights
    pair_weights = np.exp(-pair_vals / t)
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        m = min(np.min(pair_vals), np.min(wall_vals))
        return m, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Pair contributions
    pair_grad = np.zeros_like(c)
    np.add.at(pair_grad, _IU, 0.5 * pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(pair_grad, _JU, -0.5 * pair_weights[:, np.newaxis] * unit_vecs)
    
    grad_c += t * pair_grad
    
    # Wall contributions
    signs = np.sign(c)
    wall_grad = -t * wall_weights * signs
    
    grad_c += wall_grad
    
    # Normalize
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _fast_polish(c_init, deadline, t_start=0.05, t_end=0.001, max_steps=2000):
    """
    Custom gradient ascent with adaptive step size for the soft-min objective.
    Much faster than L-BFGS-B for this problem size.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Annealing schedule
    n_temps = 8
    temps = np.geomspace(t_start, t_end, n_temps)
    
    steps_per_temp = max_steps // n_temps
    
    for t in temps:
        if time.time() > deadline:
            break
            
        step_size = 0.01
        for step in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            r, grad = _softmin_radius_and_grad(c, t)
            
            # Normalize gradient to control step size
            norm = np.linalg.norm(grad)
            if norm < 1e-12:
                break
            grad = grad / norm
            
            # Try a step
            c_try = c + step_size * grad
            c_try = np.clip(c_try, -0.4999, 0.4999)
            
            r_try, _ = _softmin_radius_and_grad(c_try, t)
            
            # Accept if better
            if r_try > r:
                c = c_try
                step_size *= 1.05  # Increase step size slightly
                r_curr = r_try
                # Update global best based on hard radius
                hard_r = _compute_radius(c)
                if hard_r > best_r:
                    best_r = hard_r
                    best_c = c.copy()
            else:
                step_size *= 0.5  # Decrease step size
                if step_size < 1e-6:
                    break
    
    return best_c, best_r

def _repulsion_anneal(c_init, deadline, r_target=0.13):
    """
    Repulsion-based simulated annealing to spread points apart.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    r_curr = best_r
    
    temp = 0.05
    decay = 0.997
    max_iters = 3000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute repulsion forces (vectorized)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        np.fill_diagonal(dists, np.inf)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            force_scale = np.where(dists < 2*r_target, 1.0/(dists**2 + 1e-10), 0.0)
        force_scale[np.isnan(force_scale)] = 0.0
        
        forces = np.sum(force_scale[:, :, np.newaxis] * diff, axis=1)
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_excess = np.maximum(0.0, r_target - wall_dists)
        wall_force = np.sign(c) * wall_excess * 2.0
        
        total_force = forces + wall_force
        
        # Random perturbation
        noise = np.random.normal(0.0, temp, size=c.shape)
        
        c_new = c + 0.01 * total_force + noise
        c_new = np.clip(c_new, -0.499, 0.499)
        
        r_new = _compute_radius(c_new)
        
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / temp):
            c = c_new
            r_curr = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= decay
        if temp < 0.001:
            temp = 0.001
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Find binding constraint
        diff = c[_IU] - c[_JU]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.002, 0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.4999, 0.4999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            # Pair binding
            min_idx = np.argmin(dists)
            i = _IU[min_idx]
            j = _JU[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.4999, 0.4999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    return best_c, best_r

def _seed_fcc():
    """
    Generate an FCC-like initial configuration.
    """
    a = 0.25
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x, y, z = i*a - 0.5, j*a - 0.5, k*a - 0.5
                points.append((x, y, z))
                if i < 3:
                    points.append((x + a/2, y, z))
                if j < 3:
                    points.append((x, y + a/2, z))
                if k < 3:
                    points.append((x, y, z + a/2))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.48, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.015, size=(N, 3))
    else:
        pad = np.random.uniform(-0.35, 0.35, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_bcc():
    """
    Generate a BCC-like initial configuration.
    BCC has 2 atoms per unit cell: (0,0,0) and (0.5, 0.5, 0.5).
    We use a 4x4x4 grid of unit cells, giving 4*4*4*2 = 128 points.
    """
    a = 0.25
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x0, y0, z0 = i*a - 0.5, j*a - 0.5, k*a - 0.5
                points.append((x0, y0, z0))
                points.append((x0 + a/2, y0 + a/2, z0 + a/2))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.48, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.015, size=(N, 3))
    else:
        pad = np.random.uniform(-0.35, 0.35, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx] + np.random.normal(0, 0.01, size=(N, 3))

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.4, 0.4, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_bcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Allocate time to seeds
    # FCC and BCC are most promising
    time_budgets = [0.30, 0.30, 0.20, 0.20]
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 10.0:
            break
            
        seed_deadline = start_time + time_budgets[i] * (deadline - start_time)
        seed_deadline = min(seed_deadline, deadline - 10.0)
        
        # Stage 1: Repulsion annealing (short)
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=0.13)
        
        # Stage 2: Fast polish (long)
        c_polish, r_polish = _fast_polish(c_anneal, seed_deadline, t_start=0.05, t_end=0.001, max_steps=2000)
        
        # Stage 3: Local search (short)
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 5.0:
        final_deadline = deadline - 2.0
        c_final, r_final = _fast_polish(best_c, final_deadline, t_start=0.01, t_end=0.0001, max_steps=1000)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # Final local search
        if time.time() < final_deadline - 1.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
