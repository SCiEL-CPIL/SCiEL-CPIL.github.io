"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Physics-based Simulated Annealing with improved soft-min polishing.
"""

import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import Voronoi, Delaunay

N = 50

# EVOLVE-BLOCK-START

# Precompute upper triangle indices for pairwise distances
_PAIR_I, _PAIR_J = np.triu_indices(N, k=1)
_N_PAIRS = len(_PAIR_I)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_PAIR_I] - c[_PAIR_J]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp for stability.
    Returns (radius, gradient) where gradient is (N, 3).
    Only uses upper triangle for pairs to avoid double counting.
    """
    # Pair values (upper triangle only)
    diff = c[_PAIR_I] - c[_PAIR_J]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    # Wall values
    wall_vals = 0.5 - np.abs(c)
    
    # Log-sum-exp for numerical stability
    pair_log_w = -pair_vals / t
    wall_log_w = -wall_vals / t
    
    all_log_w = np.concatenate([pair_log_w, wall_log_w.ravel()])
    max_log_w = np.max(all_log_w)
    
    if max_log_w < -700:
        # Underflow case, return hard min
        m = np.min(np.concatenate([pair_vals, wall_vals.ravel()]))
        return m, np.zeros_like(c)
    
    exp_w = np.exp(all_log_w - max_log_w)
    sum_exp = np.sum(exp_w)
    
    if sum_exp < 1e-300:
        m = np.min(np.concatenate([pair_vals, wall_vals.ravel()]))
        return m, np.zeros_like(c)
    
    softmin_val = -t * (np.log(sum_exp) + max_log_w)
    
    # Weights (normalized)
    weights = exp_w / sum_exp
    
    # Split back into pair and wall weights
    pair_weights = weights[:_N_PAIRS]
    wall_weights = weights[_N_PAIRS:].reshape(N, 3)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions: for pair (i,j) with value d/2, 
    # d(softmin)/d(c_i) = -weight * (c_i - c_j)/d * 0.5 * ... 
    # Actually: softmin = -t * log(sum exp(-v_k/t))
    # d(softmin)/d(c_i) = sum_k weight_k * d(-v_k)/d(c_i)
    # For pair (i,j): v = |c_i - c_j|/2
    # d(v)/d(c_i) = (c_i - c_j)/|c_i - c_j|
    # So d(softmin)/d(c_i) = -weight * (c_i - c_j)/|c_i - c_j|
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # For each pair (i,j): contribution to grad[i] is -w * unit_vec
    #                         contribution to grad[j] is +w * unit_vec
    for k in range(_N_PAIRS):
        i, j = _PAIR_I[k], _PAIR_J[k]
        grad_c[i] -= pair_weights[k] * unit_vecs[k]
        grad_c[j] += pair_weights[k] * unit_vecs[k]
    
    # Wall contributions: v = 0.5 - |c_ik|
    # d(v)/d(c_ik) = -sign(c_ik)
    # d(softmin)/d(c_ik) = -weight * (-sign(c_ik)) = weight * sign(c_ik)
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    return softmin_val, grad_c

def _anneal_objective(x, r_target):
    """
    Objective for simulated annealing phase.
    """
    c = x.reshape(N, 3)
    energy = 0.0
    
    # Pairwise overlaps
    diff = c[_PAIR_I] - c[_PAIR_J]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    
    overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
    energy += np.sum(overlap_pair ** 2)
    
    # Wall overlaps
    wall_dists = 0.5 - np.abs(c)
    overlap_wall = np.maximum(0.0, r_target - wall_dists)
    energy += np.sum(overlap_wall ** 2)
    
    return energy

def _simulate_anneal(c_init, deadline, r_init=0.08, r_step=0.0003, max_iters=3000):
    """
    Perform simulated annealing to find a configuration that supports a growing radius.
    """
    c = c_init.copy()
    current_r = r_init
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.05
    decay = 0.998
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current energy
        E = _anneal_objective(c.flatten(), current_r)
        
        # Try to increase radius if feasible
        if E < 1e-8:
            current_r += r_step
            if it % 200 == 0:
                temp = max(0.005, temp * 1.05)
        else:
            temp *= decay
        
        # Perturbation - adaptive step size
        step_size = 0.005 + 0.01 * min(1.0, E)
        move = np.random.normal(0.0, step_size, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.495, 0.495)
        
        E_new = _anneal_objective(c_new.flatten(), current_r)
        
        # Accept if better or with probability exp(-dE/T)
        dE = E_new - E
        if dE < 0 or (temp > 1e-6 and np.random.random() < np.exp(-dE / temp)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.03, t_end=0.001, n_temps=12):
    """
    Polish the configuration using gradient descent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
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
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
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
    Lloyd's algorithm: move each center toward the centroid of its Voronoi cell.
    """
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
                center = np.array([0.0, 0.0, 0.0])
                direction = center - c[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
                new_c[i] = c[i] + 0.005 * direction
                continue
            
            vertices = vor.vertices[region]
            
            if len(vertices) < 3:
                continue
            
            centroid = np.mean(vertices, axis=0)
            centroid = np.clip(centroid, -0.5 + clip_margin, 0.5 - clip_margin)
            new_c[i] = 0.8 * c[i] + 0.2 * centroid
        
        c = new_c
    
    return c

def _lloyd_optimize(c_init, deadline, max_rounds=40):
    """
    Run Lloyd's algorithm with periodic radius checks.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        c = _lloyd_step(c, iterations=15)
        
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        else:
            if round_idx % 7 == 6:
                c = best_c.copy()
                perturb_idx = np.random.choice(N, size=4, replace=False)
                for idx in perturb_idx:
                    c[idx] += np.random.normal(0, 0.015, size=3)
                c = np.clip(c, -0.49, 0.49)
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=800):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r + 1e-15:
            c = best_c.copy()
            r = best_r
             
        # Find the binding constraint
        diff = c[_PAIR_I] - c[_PAIR_J]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_idx = np.argmin(dists)
        min_pair_dist = dists[min_pair_idx]
        i, j = _PAIR_I[min_pair_idx], _PAIR_J[min_pair_idx]
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_dist = wall_dists[min_wall_idx]
        wi, wk = np.unravel_index(min_wall_idx, wall_dists.shape)
        
        # If wall is binding, move the closest point away from the wall
        # If pair is binding, move the closest pair apart
        
        if min_wall_dist < min_pair_dist / 2.0:
            sign = 1.0 if c[wi, wk] > 0 else -1.0
            for step in (0.003, 0.006, 0.01, 0.015, 0.02):
                c_try = c.copy()
                c_try[wi, wk] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.003, 0.006, 0.01, 0.015, 0.02):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
    return best_c, best_r

def _nelder_mead_polish(c_init, deadline):
    """
    Final high-precision polish using Nelder-Mead on the exact radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def obj(x):
        return -_compute_radius(x.reshape(N, 3))
    
    x0 = c.flatten()
    bounds = [(-0.499999, 0.499999)] * (N * 3)
    
    # Use Nelder-Mead with tight tolerances
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        c_new = res.x.reshape(N, 3)
        c_new = np.clip(c_new, -0.499999, 0.499999)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
    except Exception:
        pass
    
    return best_c, best_r

def _seed_fcc():
    """
    Generate an FCC-like initial configuration.
    """
    # FCC lattice points in a cube
    a = 1.0  # cube side
    n = 4  # divisions per axis
    
    points = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = (i + 0.5) / n - 0.5
                y = (j + 0.5) / n - 0.5
                z = (k + 0.5) / n - 0.5
                points.append([x, y, z])
                # FCC offset
                points.append([x + 0.25/n, y + 0.25/n, z])
                points.append([x + 0.25/n, y, z + 0.25/n])
                points.append([x, y + 0.25/n, z + 0.25/n])
    
    points = np.array(points)
    # Filter to those inside cube
    mask = np.all(np.abs(points) < 0.5, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        # Randomly select N
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx].copy()
    else:
        # Pad with random
        remaining = N - len(points)
        extra = np.random.uniform(-0.45, 0.45, size=(remaining, 3))
        return np.vstack([points, extra])

def _seed_grid():
    """
    Generate a grid-like initial configuration.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx].copy()

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_shell_core():
    """
    Generate a shell-core configuration: some points on a shell, some in core.
    """
    n_shell = 30
    n_core = N - n_shell
    
    # Shell points on a sphere of radius ~0.35
    theta = np.random.uniform(0, 2*np.pi, n_shell)
    phi = np.random.uniform(0, np.pi, n_shell)
    r_shell = 0.35
    shell = np.column_stack([
        r_shell * np.sin(phi) * np.cos(theta),
        r_shell * np.sin(phi) * np.sin(theta),
        r_shell * np.cos(phi)
    ])
    
    # Core points
    core = np.random.uniform(-0.2, 0.2, size=(n_core, 3))
    
    return np.vstack([shell, core])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Initial configurations - diverse seeds
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_shell_core(),
        np.random.uniform(-0.4, 0.4, size=(N, 3)),
    ]
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time - 15.0) / n_seeds  # Reserve 15s for final polish
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        if time.time() > deadline - 20.0:
            break
        
        # Stage 1: Annealing
        c_anneal, r_anneal = _simulate_anneal(s, min(seed_deadline, deadline - 25.0), max_iters=2000)
        
        # Stage 2: Lloyd's algorithm
        if time.time() < min(seed_deadline, deadline - 25.0) - 5.0:
            c_lloyd, r_lloyd = _lloyd_optimize(c_anneal, min(seed_deadline, deadline - 25.0) - 3.0, max_rounds=25)
        else:
            c_lloyd, r_lloyd = c_anneal, r_anneal
        
        # Stage 3: Polish
        c_polish, r_polish = _polish(c_lloyd, min(seed_deadline, deadline - 25.0), t_start=0.02, t_end=0.001)
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, min(seed_deadline, deadline - 25.0), max_iters=400)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final refinement on the global best
    final_start = time.time()
    final_deadline = deadline - 3.0
    
    if final_start < final_deadline - 10.0:
        # Final Lloyd's pass
        c_lloyd_final, r_lloyd_final = _lloyd_optimize(best_c, final_deadline - 12.0, max_rounds=20)
        if r_lloyd_final > best_r:
            best_r = r_lloyd_final
            best_c = c_lloyd_final.copy()
    
    if time.time() < final_deadline - 8.0:
        # Final polish with tighter parameters
        c_final, r_final = _polish(best_c, final_deadline - 8.0, t_start=0.01, t_end=0.0005, n_temps=8)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
    
    if time.time() < final_deadline - 5.0:
        # Nelder-Mead polish
        c_nm, r_nm = _nelder_mead_polish(best_c, final_deadline - 5.0)
        if r_nm > best_r:
            best_r = r_nm
            best_c = c_nm.copy()
    
    if time.time() < final_deadline - 2.0:
        # One last local search
        c_ls, r_ls = _local_search(best_c, final_deadline - 2.0, max_iters=500)
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.499999, 0.499999)
    
    return best_c

# EVOLVE-BLOCK-END
