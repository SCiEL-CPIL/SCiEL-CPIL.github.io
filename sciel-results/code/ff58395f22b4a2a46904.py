"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Smooth soft-min maximization via log-sum-exp with gradient-based
polishing, combined with repulsion-based simulated annealing for global search.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using upper triangle only
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
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
    # Use upper triangle for pairs
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Weights for pairs
    pair_weights = np.exp(-pair_vals / t)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        m = min(np.min(pair_vals), np.min(wall_vals))
        return m, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions: for each pair (i,j), gradient on i is -0.5*w*(ci-cj)/d, on j is +0.5*w*(ci-cj)/d
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # grad on i: -0.5 * w * unit_vec
    # grad on j: +0.5 * w * unit_vec
    grad_contrib = -0.5 * pair_weights[:, np.newaxis] * unit_vecs
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        grad_c[i] += grad_contrib[idx]
        grad_c[j] -= grad_contrib[idx]
    
    # Wall contributions
    signs = np.sign(c)
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, r_target=0.13):
    """
    Repulsion-based simulated annealing to spread points apart.
    Uses inverse-distance repulsion to push overlapping spheres apart.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temp = 0.05
    decay = 0.998
    max_iters = 3000
    
    iu = np.triu_indices(N, k=1)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute repulsion forces using upper triangle
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-10)
        
        # Repulsion force: stronger when closer
        with np.errstate(divide='ignore', invalid='ignore'):
            force_scale = np.where(dists < 2*r_target, 1.0/(dists**2 + 1e-10), 0.0)
        force_scale[np.isnan(force_scale)] = 0.0
        
        # Accumulate forces
        forces = np.zeros_like(c)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            f = force_scale[idx] * diff[idx]
            forces[i] += f
            forces[j] -= f
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_force = np.zeros_like(c)
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] < r_target:
                    wall_force[i, k] = np.sign(c[i, k]) * (r_target - wall_dists[i, k]) * 2.0
        
        total_force = forces + wall_force
        
        # Random perturbation
        noise = np.random.normal(0.0, temp, size=c.shape)
        
        c_new = c + 0.01 * total_force + noise
        c_new = np.clip(c_new, -0.499, 0.499)
        
        # Evaluate
        r_new = _compute_radius(c_new)
        r_curr = _compute_radius(c)
        
        # Accept if better or with probability
        if r_new > r_curr or np.random.random() < np.exp((r_new - r_curr) / temp):
            c = c_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        
        temp *= decay
        if temp < 0.001:
            temp = 0.001
    
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0005):
    """
    Polish the configuration using gradient ascent on the softmin radius
    with decreasing temperature for increasingly sharp approximation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Geometric progression of temperatures - more stages
    n_temps = 10
    temps = np.geomspace(t_start, t_end, n_temps)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
            
        x0 = c.flatten()
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 80}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _lloyd_step(c, iterations=50, clip_margin=0.001):
    """
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell,
    clipped to the cube boundary. This equalizes clearance around each sphere.
    """
    from scipy.spatial import Voronoi
    
    c = c.copy()
    for _ in range(iterations):
        try:
            vor = Voronoi(c)
        except Exception:
            break
        
        new_c = c.copy()
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region - nudge toward center
                new_c[i] = c[i] * 0.9
                continue
            
            # Get vertices of the Voronoi cell
            verts = vor.vertices[region]
            
            if len(verts) == 0:
                new_c[i] = c[i] * 0.9
                continue
            
            # Clip vertices to cube
            verts = np.clip(verts, -0.5 + clip_margin, 0.5 - clip_margin)
            
            # Compute centroid
            centroid = np.mean(verts, axis=0)
            
            # Move toward centroid with damping
            new_c[i] = 0.7 * c[i] + 0.3 * centroid
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        c = new_c
    
    return c

def _lloyd_optimize(c_init, deadline):
    """
    Run Lloyd's algorithm followed by smooth polishing.
    """
    c = _lloyd_step(c_init, iterations=30)
    c = np.clip(c, -0.4999, 0.4999)
    r = _compute_radius(c)
    
    # Polish after Lloyd
    c, r = _smooth_polish(c, deadline, t_start=0.02, t_end=0.001)
    
    return c, r

def _local_search(c_init, deadline, max_iters=300):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    iu = np.triu_indices(N, k=1)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        # Find the binding constraint using upper triangle
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        # If wall is binding, move the closest point away from the wall
        # If pair is binding, move the closest pair apart
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            # Direction away from wall
            sign = 1.0 if c[i, k] > 0 else -1.0
            # Try moving
            for step in (0.003, 0.006, 0.01, 0.02):
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
            i, j = iu[0][min_idx], iu[1][min_idx]
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.003, 0.006, 0.01, 0.02):
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
    # FCC lattice points in a cube
    a = 1.0 / 4.0  # lattice parameter
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Corner
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                # Face centers
                if i < 3:
                    points.append((i*a - 0.5 + a/2, j*a - 0.5, k*a - 0.5))
                if j < 3:
                    points.append((i*a - 0.5, j*a - 0.5 + a/2, k*a - 0.5))
                if k < 3:
                    points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5 + a/2))
    
    points = np.array(points)
    # Filter to cube
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
        # Pad with random points
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
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

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    c = np.zeros((N, 3))
    # Core: 8 points in inner region
    core = np.random.uniform(-0.15, 0.15, size=(8, 3))
    c[:8] = core
    # Shell: 42 points in outer region
    shell = np.random.uniform(-0.45, 0.45, size=(42, 3))
    c[8:] = shell
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_random())
    seeds.append(_seed_grid())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: give more time to final polish
    # Reserve 25% of time for final polish
    final_time = (deadline - start_time) * 0.25
    search_time = (deadline - start_time) - final_time
    time_per_seed = search_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - final_time - 5.0)
        
        if time.time() > seed_deadline:
            continue
            
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=0.13)
        
        if time.time() > seed_deadline:
            if r_anneal > best_r:
                best_r = r_anneal
                best_c = c_anneal.copy()
            continue
            
        # Stage 2: Lloyd's algorithm + smooth polish
        c_lloyd, r_lloyd = _lloyd_optimize(c_anneal, seed_deadline)
        
        if time.time() > seed_deadline:
            if r_lloyd > best_r:
                best_r = r_lloyd
                best_c = c_lloyd.copy()
            continue
            
        # Stage 3: Additional smooth polish with finer temps
        c_polish, r_polish = _smooth_polish(c_lloyd, seed_deadline, t_start=0.01, t_end=0.0003)
        
        if time.time() > seed_deadline:
            if r_polish > best_r:
                best_r = r_polish
                best_c = c_polish.copy()
            continue
            
        # Stage 4: Local search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    final_deadline = deadline - 5.0
    if time.time() < final_deadline:
        # Multi-stage final polish
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.005, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        # Ultra-fine final polish
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.001, t_end=0.00005)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
    return best_c

# EVOLVE-BLOCK-END
