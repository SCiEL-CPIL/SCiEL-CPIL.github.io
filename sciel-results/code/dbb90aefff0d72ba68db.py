"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initial configurations (FCC-inspired, grid, random)
2. Repulsion-based simulated annealing for global search
3. Voronoi Lloyd's algorithm for balanced spacing
4. Softmin gradient polishing for fine-tuning
5. Local search on binding constraints
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# EVOLVE-BLOCK-START

def _seed_fcc():
    """
    Generate an FCC-inspired initial configuration for 50 spheres.
    FCC has 4 atoms per unit cell. We create a 4x4x4 grid of FCC positions
    (64 positions) and select 50 of them.
    """
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                # FCC basis vectors
                for dx, dy, dz in [(0, 0, 0),
                                    (0.5, 0.5, 0),
                                    (0.5, 0, 0.5),
                                    (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    
    points = np.array(points)
    if len(points) < N:
        # If we don't have enough, fill with random
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        # Randomly select N points
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_grid():
    """
    Generate a grid-based initial configuration.
    Use a 4x4x4 grid (64 points) and select 50.
    """
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_random():
    """
    Generate a random initial configuration.
    """
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    Place some points on a spherical shell and some in the core.
    """
    n_core = 20
    n_shell = N - n_core
    
    # Core: random points in a smaller cube
    core = np.random.uniform(-0.25, 0.25, size=(n_core, 3))
    
    # Shell: points on a sphere of radius ~0.35
    theta = np.random.uniform(0, 2 * np.pi, size=n_shell)
    phi = np.random.uniform(0, np.pi, size=n_shell)
    r = 0.35 + np.random.uniform(-0.05, 0.05, size=n_shell)
    shell = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    
    return np.vstack([core, shell])

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=50):
    """
    Repulsion-based simulated annealing.
    Objective: maximize the minimum of (pair distances/2, wall distances).
    We use a repulsive force model: each pair of points repels with force ~ 1/d^2,
    and walls repel with force ~ 1/wall_dist^2.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 15)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for it in range(iters_per_temp):
            if time.time() > deadline:
                break
            
            # Compute repulsive forces
            forces = np.zeros_like(c)
            
            # Pair repulsion
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, 1e10)
            
            # Force magnitude: k / d^2, direction: away from other point
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists * dists)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            unit_vecs = diff / dists[:, :, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, :, np.newaxis] * unit_vecs
            forces += np.sum(pair_forces, axis=1)
            
            # Wall repulsion
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / (wall_dists * wall_dists)
            wall_force_mag = np.where(wall_force_mag > 1e6, 1e6, wall_force_mag)
            wall_force_mag = np.where(np.isnan(wall_force_mag), 0, wall_force_mag)
            
            signs = np.sign(c)
            forces += wall_force_mag * signs
            
            # Add noise and update
            noise = np.random.normal(0, t * 0.1, size=c.shape)
            c_new = c + forces * 0.01 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            
            # Accept if better or with probability exp(-delta_r / t)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            delta_r = r_new - _compute_radius(c)
            if delta_r > 0 or np.random.random() < np.exp(delta_r / max(t, 1e-10)):
                c = c_new
    
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=100):
    """
    Polish using gradient descent on the softmin radius with very fine temperature
    for final precision extraction.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 12)
    
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
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
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

def _epigraph_polish(c_init, deadline, r_target=None, max_iters=200):
    """
    Epigraph formulation: maximize t such that all constraints hold.
    We optimize over (centers, t) where t is the radius lower bound.
    Objective: minimize -t + penalty for violated constraints.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    if r_target is None:
        r_target = best_r * 0.999
    
    # Precompute pair indices for efficiency
    n_pairs = N * (N - 1) // 2
    pair_idx = np.array([(i, j) for i in range(N) for j in range(i+1, N)])
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute current radius
        r_cur = _compute_radius(c)
        
        # Try to increase the target radius slightly
        r_target = max(r_target, r_cur * 0.9999)
        
        # Define objective: minimize sum of constraint violations + regularizer
        # Constraint: dist(i,j) >= 2*t for all pairs
        # Constraint: wall_dist(i,k) >= t for all i,k
        
        def obj(x):
            centers = x[:N*3].reshape(N, 3)
            t = x[N*3]
            
            # Pair constraints
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            pair_viol = np.maximum(0, 2*t - dists)
            
            # Wall constraints
            wall_dists = 0.5 - np.abs(centers)
            wall_viol = np.maximum(0, t - wall_dists)
            
            # Penalty: squared violations (smooth)
            penalty = 0.5 * (np.sum(pair_viol**2) + np.sum(wall_viol**2))
            
            # Regularizer to encourage larger t
            return -t + 100.0 * penalty
        
        def grad(x):
            centers = x[:N*3].reshape(N, 3)
            t = x[N*3]
            
            # Pair gradient
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            grad_centers = np.zeros_like(centers)
            grad_t = -1.0
            
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2*t and d > 1e-10:
                        # Violation: 2*t - d > 0
                        viol = 2*t - d
                        # dGrad of dist w.r.t. center i
                        vec = centers[i] - centers[j]
                        grad_d_i = vec / d
                        grad_d_j = -vec / d
                        
                        # Gradient of 0.5*(2*t - d)^2
                        factor = 100.0 * viol
                        grad_centers[i] += factor * grad_d_i
                        grad_centers[j] += factor * grad_d_j
                        grad_t += 100.0 * 2 * viol
            
            # Wall gradient
            wall_dists = 0.5 - np.abs(centers)
            for i in range(N):
                for k in range(3):
                    wd = wall_dists[i, k]
                    if wd < t:
                        viol = t - wd
                        sign = np.sign(centers[i, k])
                        if sign == 0:
                            sign = 1.0
                        grad_centers[i, k] += 100.0 * viol * sign
                        grad_t += 100.0 * viol
            
            # Clip centers to valid range
            centers = np.clip(centers, -0.499999, 0.499999)
            
            return np.concatenate([grad_centers.ravel(), [grad_t]])
        
        x0 = np.concatenate([c.flatten(), [r_target]])
        bounds = [(-0.5, 0.5)] * (N * 3) + [(r_target * 0.5, 0.5)]
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            c_new = res.x[:N*3].reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                r_target = min(r_new * 1.0001, 0.5)
            else:
                # No improvement, slightly reduce target
                r_target = max(r_target * 0.999, best_r * 0.999)
                
        except Exception:
            continue
    
    return best_c, best_r

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    m = np.min(all_vals)
    
    # Weights for pairs
    pair_weights = np.exp(-pair_vals / t)
    np.fill_diagonal(pair_weights, 0.0)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, :, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    grad_pair_contrib = -0.5 * pair_weights[:, :, np.newaxis] * unit_vecs
    grad_c += np.sum(grad_pair_contrib, axis=1)
    
    # Wall contributions
    signs = np.sign(c)
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _lloyd_step(c, step_size=0.5):
    """
    Perform one Lloyd's algorithm iteration: move each center toward the centroid
    of its Voronoi cell, clipped to the cube boundary.
    
    The Voronoi cell for point i is the set of points closer to i than to any other
    center. The centroid of this cell is a natural "balanced" position that
    equalizes the influence region, which helps maximize the minimum clearance.
    """
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    
    c_new = c.copy()
    
    for i in range(N):
        # Get the Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Unbounded region - keep original position or nudge toward center
            continue
            
        # Get the vertices of this Voronoi cell
        verts = vor.vertices[region]
        
        if len(verts) == 0:
            continue
            
        # Compute centroid of the Voronoi cell
        centroid = np.mean(verts, axis=0)
        
        # Clip to cube
        centroid = np.clip(centroid, -0.499999, 0.499999)
        
        # Move toward centroid with damping
        c_new[i] = c[i] + step_size * (centroid - c[i])
        c_new[i] = np.clip(c_new[i], -0.499999, 0.499999)
    
    return c_new

def _lloyd_optimize(c_init, deadline, max_iters=500, step_decay=0.995):
    """
    Run Lloyd's algorithm iterations with decreasing step size.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    step_size = 0.3
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        c_new = _lloyd_step(c, step_size)
        
        # Ensure no coincident points
        diff = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 1e-6:
            # Add small perturbation to separate coincident points
            c_new += np.random.normal(0, 1e-4, size=c_new.shape)
            c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            # If no improvement, reduce step size more aggressively
            step_size *= 0.95
        
        step_size *= step_decay
        step_size = max(step_size, 0.001)
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.001, iters_per_temp=100):
    """
    Polish the configuration using gradient descent on the softmin radius with analytical gradients.
    """
    return _smooth_polish(c_init, deadline, t_start=t_start, t_end=t_end, iters_per_temp=iters_per_temp)

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
            
        r = _compute_radius(c)
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        # Find the binding constraint
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
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
            for step in (0.005, 0.01, 0.02):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    break
        else:
            # Pair binding
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            # Direction apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.005, 0.01, 0.02):
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

def _nelder_mead_polish(c_init, deadline, maxiter=200):
    """
    Final high-precision polish using Nelder-Mead on the exact radius objective.
    This handles the non-smooth min objective directly.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def obj(x):
        centers = x.reshape(N, 3)
        centers = np.clip(centers, -0.499999, 0.499999)
        r = _compute_radius(centers)
        return -r
    
    x0 = c.flatten()
    
    try:
        res = minimize(
            obj,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. FCC-inspired
    seeds.append(_seed_fcc())
    
    # 2. Grid
    seeds.append(_seed_grid())
    
    # 3. Random
    seeds.append(_seed_random())
    
    # 4. Shell-core
    seeds.append(_seed_shell_core())
    
    # 5. Another random
    seeds.append(_seed_random())

    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    n_seeds = len(seeds)
    time_per_seed = (deadline - start_time) / n_seeds
    
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5.0:
            break
        
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - 5.0)
        
        # Stage 1: Repulsion annealing for global search
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, t_start=2.0, t_end=0.05, iters_per_temp=30)
        
        # Stage 2: Lloyd's algorithm (Voronoi-based)
        c_lloyd, r_lloyd = _lloyd_optimize(c_anneal, seed_deadline, max_iters=200)
        
        # Stage 3: Polish with softmin gradient
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.005)
        
        # Stage 4: Epigraph polish
        c_epi, r_epi = _epigraph_polish(c_polish, seed_deadline, max_iters=100)
        
        # Stage 5: Local Search
        c_ls, r_ls = _local_search(c_epi, seed_deadline, max_iters=100)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 10.0:
        final_deadline = deadline - 3.0
        
        # Try Lloyd's again on the best configuration
        c_lloyd2, r_lloyd2 = _lloyd_optimize(best_c, final_deadline, max_iters=150)
        if r_lloyd2 > best_r:
            best_r = r_lloyd2
            best_c = c_lloyd2.copy()
        
        # Very fine polish to extract maximum precision
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.03, t_end=0.0001, iters_per_temp=80)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # Epigraph polish on the best
        if time.time() < final_deadline - 5.0:
            c_epi_final, r_epi_final = _epigraph_polish(best_c, final_deadline, max_iters=150)
            if r_epi_final > best_r:
                best_r = r_epi_final
                best_c = c_epi_final.copy()
        
        # Nelder-Mead polish for final precision
        if time.time() < final_deadline - 3.0:
            c_nm, r_nm = _nelder_mead_polish(best_c, final_deadline, maxiter=100)
            if r_nm > best_r:
                best_r = r_nm
                best_c = c_nm.copy()
        
        # One last local search
        if time.time() < final_deadline - 1.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
