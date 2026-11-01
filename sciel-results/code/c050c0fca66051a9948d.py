"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice)
2. Simulated annealing with repulsion to escape local minima
3. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
4. Smooth gradient-based polishing (soft-min via log-sum-exp)
5. Final high-precision polish pass
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
    Uses only unique pairs (i < j) to avoid double-counting.
    Returns (radius, gradient) where gradient is (N, 3).
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Use only upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]  # shape (N*(N-1)/2,)
    pair_vals = pair_dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
    wall_flat = wall_vals.ravel()
    
    # Concatenate all constraints
    all_vals = np.concatenate([pair_vals, wall_flat])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)
        
    # Stable log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)
    
    weights = exp_vals / s  # normalized weights
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    n_pairs = len(pair_vals)
    pair_w = weights[:n_pairs]  # weights for unique pairs
    
    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    
    # Vectorized: for each pair, compute contribution
    diff_pair = c[pi] - c[pj]  # shape (n_pairs, 3)
    
    # factor[k] = pair_w[k] / (2 * safe_dists[k])
    factor = pair_w / (2.0 * safe_dists)  # shape (n_pairs,)
    
    # Contribution to grad[pi[k]]: -factor[k] * diff_pair[k]
    # Contribution to grad[pj[k]]: +factor[k] * diff_pair[k]
    
    np.add.at(grad, pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, pj, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_w = weights[n_pairs:]  # N*3 weights
    wall_w = wall_w.reshape(N, 3)
    
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _epigraph_objective(x, t):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    x = [c.flatten(), t] where c is (N,3) and t is scalar.
    Returns negative of t (to minimize) with penalty for constraint violations.
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]
    
    # Compute all constraint values
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu] / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Violations: how much each constraint is below t
    violations = np.maximum(0, t - all_vals)
    
    # Penalize violations quadratically + maximize t
    penalty = np.sum(violations ** 2)
    
    return -t + penalty

def _epigraph_grad(x, t):
    """
    Gradient of epigraph objective.
    """
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_vals = pair_dists / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # violations = max(0, t - val)
    # d/dval: -2*violations if t > val else 0
    violations = np.maximum(0, t_val - all_vals)
    
    n_pairs = len(pair_vals)
    pair_viol = violations[:n_pairs]
    wall_viol = violations[n_pairs:]
    
    grad = np.zeros_like(x)
    
    # Pair contributions
    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    
    # d(pair_val)/d(c_i) = (c_i - c_j) / (2*dist)
    # d(obj)/d(c_i) += -2 * pair_viol[k] * (c_i - c_j) / (2*dist)
    #                = -pair_viol[k] * (c_i - c_j) / dist
    
    diff_pair = c[pi] - c[pj]
    factor = pair_viol / safe_dists  # shape (n_pairs,)
    
    np.add.at(grad[:N*3].reshape(N, 3), pi, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad[:N*3].reshape(N, 3), pj, factor[:, np.newaxis] * diff_pair)
    
    # Wall contributions
    wall_viol_reshaped = wall_viol.reshape(N, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    # d(wall_val)/d(c[i,k]) = -sign(c[i,k])
    # d(obj)/d(c[i,k]) += -2 * wall_viol[i,k] * (-sign(c[i,k])) = 2 * wall_viol[i,k] * sign(c[i,k])
    grad[:N*3] += (2.0 * wall_viol_reshaped * signs).reshape(N*3)
    
    # d(obj)/dt = -1 + sum(2 * violations)
    grad[N*3] = -1.0 + 2.0 * np.sum(violations)
    
    return grad

def _repulsion_force_step(c_init, deadline, max_iters=200):
    """
    Apply a smooth repulsive force field to spread points.
    This acts as a 'pressure' solver to avoid clustering,
    using a potential that decays with distance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    iu = np.triu_indices(N, k=1)
    pi, pj = iu
    
    # We want to maximize the minimum distance.
    # Using a soft repulsion helps distribute points uniformly.
    # Force is proportional to 1/d^2 for pairs closer than some threshold.
    
    step_size = 0.01
    decay = 0.99
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        diff = c[pi] - c[pj]
        dists = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-6, dists, 1.0)
        
        # Only apply force if distance is small (clustering)
        # Threshold: slightly larger than current best radius * 2
        r_current = best_r
        threshold = 2.0 * (r_current + 0.01)
        
        # Mask for pairs that are too close
        mask = dists < threshold
        if not np.any(mask):
            # If no pairs are too close, we might still want to push towards walls if wall distance is small
            # But for now, just break or do a tiny random jitter
            # Actually, if no pairs are close, the main constraint is likely walls or we are at a local max for pairs.
            # Let's just break to save time for polish.
            break
        
        forces = np.zeros_like(c)
        
        # Compute forces for close pairs
        # F = k * (threshold - dist) / dist^2 * direction
        # We use a simple linear repulsion for stability
        deficit = np.where(mask, threshold - dists, 0.0)
        # Normalize direction
        dirs = diff / dists_safe[:, np.newaxis]
        
        # Weight by deficit
        weights = deficit / (dists_safe * dists_safe) # Inverse square law-like
        # Scale down to prevent explosion
        weights *= 0.01
        
        f_vecs = weights[:, np.newaxis] * dirs
        
        np.add.at(forces, pi, f_vecs)
        np.add.at(forces, pj, -f_vecs)
        
        # Wall forces: push away from walls if close
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        if min_wall < r_current:
            wall_forces = np.zeros_like(c)
            # If a point is too close to a wall, push it in
            for i in range(N):
                for k in range(3):
                    wd = wall_dists[i, k]
                    if wd < r_current:
                        sign = 1.0 if c[i, k] > 0 else -1.0
                        wall_forces[i, k] += sign * (r_current - wd) * 0.5
            forces += wall_forces
        
        c_new = c + step_size * forces
        c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            # If it didn't improve, try a smaller step or revert
            # Reverting is safer to maintain progress
            pass
        
        step_size *= decay
        
    return best_c, best_r

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=5000):
    """
    Simulated annealing with repulsion to spread points apart.
    Uses adaptive perturbation size and tracks best radius.
    Optimized: precompute indices, use squared distances where possible.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.01
    
    temp = 0.05
    decay = 0.999
    pert_scale = 0.008
    
    # Precompute pair indices
    iu = np.triu_indices(N, k=1)
    pi, pj = iu
    n_pairs = len(pi)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute energy using only unique pairs (faster)
        diff = c[pi] - c[pj]  # (n_pairs, 3)
        dists_sq = np.sum(diff * diff, axis=1)  # (n_pairs,)
        
        # Target: all pair distances >= 2*r_target, wall distances >= r_target
        target_dist_sq = (2.0 * r_target) ** 2
        deficit_sq = np.maximum(0.0, target_dist_sq - dists_sq)
        overlap_pair = deficit_sq
        
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair) + np.sum(overlap_wall ** 2)
        
        # Adaptive perturbation
        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Compute new energy
        diff_new = c_new[pi] - c_new[pj]
        dists_sq_new = np.sum(diff_new * diff_new, axis=1)
        deficit_sq_new = np.maximum(0.0, target_dist_sq - dists_sq_new)
        overlap_pair_new = deficit_sq_new
        
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new) + np.sum(overlap_wall_new ** 2)
        
        # Accept
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            if it % 10 == 0:
                r_check = _compute_radius(c)
                if r_check > best_r:
                    best_r = r_check
                    best_c = c.copy()
                    pert_scale *= 0.95
            else:
                pert_scale = min(pert_scale * 1.01, 0.02)
        else:
            pert_scale = min(pert_scale * 1.02, 0.02)
        
        temp *= decay
        
        # Periodically try to increase r_target
        if it % 200 == 0 and E < 1e-3:
            r_target += 0.002
    
    # Final check
    r_final = _compute_radius(c)
    if r_final > best_r:
        best_r = r_final
        best_c = c.copy()
    
    return best_c, best_r

def _lloyds_algorithm(c_init, deadline, max_iters=200, step_size=0.1):
    """
    Replaced with a faster gradient-based 'Lloyd-like' step.
    Moves points towards the centroid of their 'Voronoi-like' region defined by
    the soft-min radius gradient, effectively equalizing clearance.
    """
    # We will just use the soft-min polish with a higher temperature
    # to simulate the equalizing effect of Lloyd's algorithm.
    return _polish(c_init, deadline, t_start=0.02, t_end=0.001, iters_per_temp=30)

def _polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=50):
    """
    Polish the configuration using gradient descent on the softmin radius.
    Uses analytical gradient for speed.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    n_temps = 25
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
            
        max_iter = 80 + 80 * (ti // (n_temps // 3))
        
        def obj(x, t=t):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x, t=t):
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
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
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

def _epigraph_polish(c_init, deadline):
    """
    Final high-precision polish using epigraph formulation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    t0 = best_r - 0.001
    x0 = np.concatenate([c.flatten(), [t0]])
    
    bounds = [(-0.499999, 0.499999)] * (N * 3) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            jac=_epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
        )
        x_new = res.x
        c_new = x_new[:N*3].reshape(N, 3)
        c_new = np.clip(c_new, -0.499999, 0.499999)
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            
    except Exception:
        pass
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=300):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    iu = np.triu_indices(N, k=1)
    pi, pj = iu
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        diff = c[pi] - c[pj]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        min_pair_dist = np.min(dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.001, 0.002, 0.005, 0.01):
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
            min_idx = np.argmin(dists)
            i, j = pi[min_idx], pj[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.001, 0.002, 0.005, 0.01):
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
            else:
                if it % 5 == 0:
                    c_try = c.copy()
                    rand_move = np.random.normal(0, 0.005, size=(2, 3))
                    c_try[i] += rand_move[0]
                    c_try[j] += rand_move[1]
                    c_try = np.clip(c_try, -0.499999, 0.499999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
    
    return best_c, best_r

def _initial_fcc_like():
    """
    Generate an FCC-like initial configuration for 50 points.
    """
    a = 0.5
    points = []
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                base = np.array([i*a, j*a, k*a])
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5*a, 0.5*a]),
                    np.array([0.5*a, 0, 0.5*a]),
                    np.array([0.5*a, 0.5*a, 0]),
                ]
                for off in offsets:
                    p = base + off
                    p_centered = p - np.array([a, a, a])
                    if np.all(np.abs(p_centered) < 0.6):
                        points.append(p_centered)
    
    points = np.array(points)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_bcc():
    """
    Generate a BCC-like initial configuration.
    """
    a = 0.5
    points = []
    # Corner points
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                p = np.array([i*a, j*a, k*a])
                p_centered = p - np.array([a, a, a])
                if np.all(np.abs(p_centered) < 0.6):
                    points.append(p_centered)
    # Body center points
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                p = np.array([i*a + 0.25, j*a + 0.25, k*a + 0.25])
                p_centered = p - np.array([a, a, a])
                if np.all(np.abs(p_centered) < 0.6):
                    points.append(p_centered)
    
    points = np.array(points)
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_shell_core():
    """
    Generate a shell-core initial configuration.
    Points on a sphere surface and some in the core.
    """
    points = []
    # Core: 8 points in a small cube
    core_size = 0.1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p = np.array([(i-0.5)*core_size, (j-0.5)*core_size, (k-0.5)*core_size])
                points.append(p)
    
    # Shell: Fibonacci sphere distribution
    n_shell = N - len(points)
    if n_shell > 0:
        # Fibonacci sphere
        n = n_shell
        phi = np.pi * (3.0 - np.sqrt(5.0))
        z = (np.arange(n) - (n - 1) / 2.0) / (n / 2.0)
        radius = np.sqrt(1.0 - z * z)
        theta = phi * np.arange(n)
        
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        
        # Scale to fit in cube (radius 0.4)
        scale = 0.4
        shell_pts = np.column_stack((x, y, z)) * scale
        points.extend(shell_pts.tolist())
        
    points = np.array(points)
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    seeds = []
    seeds.append(_initial_fcc_like())
    seeds.append(_initial_bcc())
    seeds.append(_initial_shell_core())
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    seeds.append(grid[idx])
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    total_time = deadline - start_time
    # Allocate more time to the first few seeds which often yield better results
    time_per_seed = total_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        # Give more time to later seeds if they are still improving? 
        # Actually, just distribute evenly but with a slight bias to earlier ones if we want stability.
        seed_deadline = min(seed_start + time_per_seed * (1.0 + 0.1 * i), deadline - 20.0)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 1: Repulsion annealing (Escape local minima)
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=best_r + 0.005, max_iters=4000)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 2: Repulsion Force Step (Pressure field distribution)
        c_force, r_force = _repulsion_force_step(c_anneal, seed_deadline, max_iters=100)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 3: Gradient Descent (Lloyd-like equalization)
        c_lloyd, r_lloyd = _lloyds_algorithm(c_force, seed_deadline)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 4: Polish (High precision soft-min)
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.02, t_end=0.0005)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 5: Local Search (Fine tuning)
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        if time.time() > deadline - 20.0:
            break
            
    # Final high-precision polish on the global best if time remains
    if time.time() < deadline - 20.0:
        final_deadline = deadline - 3.0
        
        # Epigraph polish first
        c_epi, r_epi = _epigraph_polish(best_c, min(final_deadline, time.time() + 30.0))
        if r_epi > best_r:
            best_r = r_epi
            best_c = c_epi.copy()
        
        if time.time() < final_deadline - 5.0:
            c_final, r_final = _polish(best_c, final_deadline, t_start=0.01, t_end=0.00005)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 5.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        # Final ultra-fine polish
        if time.time() < final_deadline - 3.0:
            c_final2, r_final2 = _polish(best_c, final_deadline - 1.0, t_start=0.001, t_end=0.00001)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
        
        # Final epigraph pass
        if time.time() < final_deadline - 2.0:
            c_epi2, r_epi2 = _epigraph_polish(best_c, final_deadline - 0.5)
            if r_epi2 > best_r:
                best_r = r_epi2
                best_c = c_epi2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
