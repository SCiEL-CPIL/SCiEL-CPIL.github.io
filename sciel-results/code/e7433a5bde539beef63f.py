"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Structural change: Wasserstein-optimal-transport-inspired potential +
multi-stage optimization with improved exploration via particle swarm.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# EVOLVE-BLOCK-START

# Precompute upper triangle indices for N=50
_IU = np.triu_indices(N, k=1)
_PI, _PJ = _IU
_N_PAIRS = len(_PI)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff * diff, axis=1)
    min_pair = np.sqrt(np.min(dists_sq))
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _wasserstein_potential(c, r_est, sigma=0.05):
    """
    Compute a Wasserstein-inspired potential that penalizes non-uniform density.
    This acts as a smooth surrogate that encourages equal spacing.
    """
    diff = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff * diff, axis=1)
    
    # Gaussian kernel density estimate at each point
    kernel_vals = np.exp(-dists_sq / (2.0 * sigma * sigma))
    density = np.zeros(N)
    np.add.at(density, _PI, kernel_vals)
    np.add.at(density, _PJ, kernel_vals)
    
    # Target density for uniform distribution in cube
    # Expected number of neighbors within radius r_est
    target_density = N * (4.0 / 3.0) * np.pi * (r_est ** 3)
    
    # Penalty for deviation from target density
    deviation = np.abs(density - target_density)
    return np.sum(deviation ** 2)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    diff_pair = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff_pair * diff_pair, axis=1)
    pair_dists = np.sqrt(dists_sq)
    pair_vals = pair_dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    wall_flat = wall_vals.ravel()
    
    all_vals = np.concatenate([pair_vals, wall_flat])
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)
        
    m = np.min(all_vals)
    exp_vals = np.exp(-(all_vals - m) / t)
    s = np.sum(exp_vals)
    radius = m - t * np.log(s)
    
    weights = exp_vals / s
    
    grad = np.zeros_like(c)
    
    pair_w = weights[:_N_PAIRS]
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    factor = pair_w / (2.0 * safe_dists)
    
    np.add.at(grad, _PI, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad, _PJ, factor[:, np.newaxis] * diff_pair)
    
    wall_w = weights[_N_PAIRS:]
    wall_w = wall_w.reshape(N, 3)
    
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _epigraph_objective(x, t):
    """
    Epigraph formulation: maximize t subject to all constraints >= t.
    x = [c.flatten(), t]
    """
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    
    diff = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff * diff, axis=1)
    pair_vals = np.sqrt(dists_sq) / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    violations = np.maximum(0, t_val - all_vals)
    penalty = np.sum(violations ** 2)
    
    return -t_val + penalty

def _epigraph_grad(x, t):
    """
    Gradient of epigraph objective.
    """
    c = x[:N*3].reshape(N, 3)
    t_val = x[N*3]
    
    diff_pair = c[_PI] - c[_PJ]
    dists_sq = np.sum(diff_pair * diff_pair, axis=1)
    pair_dists = np.sqrt(dists_sq)
    pair_vals = pair_dists / 2.0
    wall_vals = (0.5 - np.abs(c)).ravel()
    
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    violations = np.maximum(0, t_val - all_vals)
    
    pair_viol = violations[:_N_PAIRS]
    wall_viol = violations[_N_PAIRS:]
    
    grad = np.zeros_like(x)
    
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    factor = pair_viol / safe_dists
    
    grad_c = grad[:N*3].reshape(N, 3)
    np.add.at(grad_c, _PI, -factor[:, np.newaxis] * diff_pair)
    np.add.at(grad_c, _PJ, factor[:, np.newaxis] * diff_pair)
    
    wall_viol_reshaped = wall_viol.reshape(N, 3)
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad[:N*3] += (2.0 * wall_viol_reshaped * signs).reshape(N*3)
    
    grad[N*3] = -1.0 + 2.0 * np.sum(violations)
    
    return grad

def _particle_swarm(c_init, deadline, r_target=0.13, max_iters=300):
    """
    Particle swarm optimization for exploring different basins.
    Each particle is a configuration of 50 points.
    """
    n_particles = 8
    particles = np.array([c_init + np.random.normal(0, 0.02, size=c_init.shape) for _ in range(n_particles)])
    velocities = np.random.normal(0, 0.005, size=(n_particles, N, 3))
    
    best_particle = c_init.copy()
    best_r = _compute_radius(c_init)
    
    p_best = particles.copy()
    p_best_r = np.array([_compute_radius(p) for p in particles])
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Update velocities
        for i in range(n_particles):
            r_i = _compute_radius(particles[i])
            
            # Find global best
            global_best_idx = np.argmax(p_best_r)
            global_best = p_best[global_best_idx]
            
            # Social and cognitive components
            w = 0.7  # inertia
            c1 = 1.5  # cognitive
            c2 = 1.5  # social
            r1 = np.random.random()
            r2 = np.random.random()
            
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best[i] - particles[i]) + 
                           c2 * r2 * (global_best - particles[i]))
            
            # Limit velocity
            v_max = 0.01
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], -0.49, 0.49)
            
            # Update personal best
            r_new = _compute_radius(particles[i])
            if r_new > p_best_r[i]:
                p_best[i] = particles[i].copy()
                p_best_r[i] = r_new
            
            # Update global best
            if r_new > best_r:
                best_r = r_new
                best_particle = particles[i].copy()
        
        # Occasional reset of worst particle
        if it % 20 == 0 and it > 0:
            worst_idx = np.argmin(p_best_r)
            p_best[worst_idx] = c_init + np.random.normal(0, 0.03, size=c_init.shape)
            p_best_r[worst_idx] = _compute_radius(p_best[worst_idx])
            particles[worst_idx] = p_best[worst_idx].copy()
            velocities[worst_idx] = np.random.normal(0, 0.005, size=(N, 3))
    
    return best_particle, best_r

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=3000):
    """
    Simulated annealing with repulsion to spread points apart.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.005
    
    temp = 0.025
    decay = 0.9985
    pert_scale = 0.005
    no_improve_count = 0
    patience = 400
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        diff = c[_PI] - c[_PJ]
        dists_sq = np.sum(diff * diff, axis=1)
        
        target_dist_sq = (2.0 * r_target) ** 2
        deficit_sq = np.maximum(0.0, target_dist_sq - dists_sq)
        overlap_pair = deficit_sq
        
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair) + np.sum(overlap_wall ** 2)
        
        current_pert = pert_scale * (0.5 + temp)
        move = np.random.normal(0.0, current_pert, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        diff_new = c_new[_PI] - c_new[_PJ]
        dists_sq_new = np.sum(diff_new * diff_new, axis=1)
        deficit_sq_new = np.maximum(0.0, target_dist_sq - dists_sq_new)
        overlap_pair_new = deficit_sq_new
        
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new) + np.sum(overlap_wall_new ** 2)
        
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            if it % 5 == 0:
                r_check = _compute_radius(c)
                if r_check > best_r + 1e-10:
                    best_r = r_check
                    best_c = c.copy()
                    no_improve_count = 0
                    if E < 1e-4:
                        r_target = max(r_target, best_r + 0.002)
                else:
                    no_improve_count += 5
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        temp *= decay
        
        if no_improve_count > patience:
            kick = np.random.normal(0.0, 0.015, size=c.shape)
            c_kick = c + kick
            c_kick = np.clip(c_kick, -0.49, 0.49)
            r_kick = _compute_radius(c_kick)
            if r_kick > best_r:
                best_r = r_kick
                best_c = c_kick.copy()
                c = c_kick
            no_improve_count = 0
            temp = max(temp, 0.01)
    
    r_final = _compute_radius(c)
    if r_final > best_r:
        best_r = r_final
        best_c = c.copy()
    
    return best_c, best_r

def _lloyds_algorithm(c_init, deadline, max_iters=100, step_size=0.12):
    """
    Lloyd's algorithm variant: move each center toward the centroid of its
    Voronoi cell (clipped to the cube) to equalize clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    boundary_points = []
    eps = 0.15
    
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign * (0.5 + eps)
            boundary_points.append(p)
    
    for i in range(3):
        for j in range(i+1, 3):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(3)
                    p[i] = si * (0.5 + eps)
                    p[j] = sj * (0.5 + eps)
                    boundary_points.append(p)
    
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = np.array([sx * (0.5 + eps), sy * (0.5 + eps), sz * (0.5 + eps)])
                boundary_points.append(p)
    
    boundary_points = np.array(boundary_points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        all_points = np.vstack([c, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except Exception:
            break
        
        new_c = c.copy()
        
        for i in range(N):
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            
            verts = vor.vertices[region]
            verts_clipped = np.clip(verts, -0.5, 0.5)
            centroid = np.mean(verts_clipped, axis=0)
            
            new_c[i] = c[i] + step_size * (centroid - c[i])
            new_c[i] = np.clip(new_c[i], -0.499, 0.499)
        
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
        else:
            c = best_c.copy()
        
        step_size *= 0.95
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.04, t_end=0.0001, iters_per_temp=50):
    """
    Polish the configuration using gradient descent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    n_temps = 15
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
            
        max_iter = 50 + 50 * (ti // (n_temps // 3))
        
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

def _epigraph_polish(c_init, deadline, t_offset=0.0005):
    """
    Final high-precision polish using epigraph formulation.
    t_offset controls how far below current best radius to start.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    t0 = max(best_r - t_offset, 0.0)
    x0 = np.concatenate([c.flatten(), [t0]])
    
    bounds = [(-0.499999, 0.499999)] * (N * 3) + [(0.0, 0.5)]
    
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            jac=_epigraph_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
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

def _local_search(c_init, deadline, max_iters=200):
    """
    Local search: Perturb points that are constraining the radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
            c = best_c.copy()
            r = best_r
             
        diff = c[_PI] - c[_PJ]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        min_pair_dist = np.min(dists)
        
        improved = False
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.0008, 0.0015, 0.003, 0.006, 0.01):
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        else:
            min_idx = np.argmin(dists)
            i, j = _PI[min_idx], _PJ[min_idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.0008, 0.0015, 0.003, 0.006, 0.01):
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 15 and it % 3 == 0:
                c_try = c.copy()
                sorted_dists = np.argsort(dists)
                for idx in sorted_dists[:3]:
                    ii, jj = _PI[idx], _PJ[idx]
                    rand_move = np.random.normal(0, 0.003, size=(2, 3))
                    c_try[ii] += rand_move[0]
                    c_try[jj] += rand_move[1]
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    no_improve = 0
    
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
                    if np.all(np.abs(p_centered) < 0.55):
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
    a = 0.28
    points = []
    
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p = np.array([i, j, k]) * a
                p_centered = p - np.array([0.5*a, 0.5*a, 0.5*a])
                if np.all(np.abs(p_centered) < 0.48):
                    points.append(p_centered)
                p_bcc = (np.array([i, j, k]) + 0.5) * a
                p_bcc_centered = p_bcc - np.array([0.5*a, 0.5*a, 0.5*a])
                if np.all(np.abs(p_bcc_centered) < 0.48):
                    points.append(p_bcc_centered)
    
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
    """
    points = []
    
    a_core = 0.2
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                p = np.array([i, j, k]) * a_core
                if np.linalg.norm(p) < 0.32:
                    points.append(p)
                p_bcc = (np.array([i, j, k]) + 0.5) * a_core
                if np.linalg.norm(p_bcc) < 0.32:
                    points.append(p_bcc)
    
    a_shell = 0.36
    face_positions = []
    for axis in range(3):
        for sign in [-1, 1]:
            for u in np.linspace(-0.28, 0.28, 3):
                for v in np.linspace(-0.28, 0.28, 3):
                    p = np.zeros(3)
                    p[axis] = sign * a_shell
                    other_axes = [ax for ax in range(3) if ax != axis]
                    p[other_axes[0]] = u
                    p[other_axes[1]] = v
                    if np.all(np.abs(p) < 0.48):
                        face_positions.append(p)
    
    points.extend(face_positions)
    
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p_corner = np.array([sx, sy, sz]) * 0.4
                if np.all(np.abs(p_corner) < 0.48):
                    points.append(p_corner)
    
    points = np.array(points)
    
    if len(points) > 0:
        unique = [points[0]]
        for p in points[1:]:
            if not any(np.linalg.norm(p - u) < 0.05 for u in unique):
                unique.append(p)
        points = np.array(unique)
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_grid():
    """
    Generate a structured grid initial configuration.
    """
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    seeds = []
    seeds.append(_initial_fcc_like())
    seeds.append(_initial_bcc())
    seeds.append(_initial_shell_core())
    seeds.append(_initial_grid())
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Give more time to first seeds
    time_allocs = [35.0, 30.0, 22.0, 16.0, 12.0]
    
    for i, s in enumerate(seeds):
        remaining = deadline - time.time()
        if remaining < 12.0:
            break
        
        alloc = min(time_allocs[i] if i < len(time_allocs) else 10.0, remaining - 12.0)
        if alloc < 7.0:
            break
        
        seed_deadline = time.time() + alloc
        
        # Stage 1: Particle swarm for exploration
        c_swarm, r_swarm = _particle_swarm(s, seed_deadline, r_target=max(best_r + 0.002, 0.12), max_iters=200)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 2: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(c_swarm, seed_deadline, r_target=max(best_r + 0.003, 0.12), max_iters=2500)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 3: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_anneal, seed_deadline, max_iters=60)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 4: Polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.03, t_end=0.0005)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 5: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=100)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Final high-precision polish on the global best if time remains
    remaining = deadline - time.time()
    if remaining > 8.0:
        final_deadline = deadline - 3.0
        
        # Multiple epigraph polish attempts
        for t_offset in [0.0005, 0.002, 0.005]:
            if time.time() > final_deadline:
                break
            c_epi, r_epi = _epigraph_polish(best_c, min(final_deadline, time.time() + 12.0), t_offset=t_offset)
            if r_epi > best_r:
                best_r = r_epi
                best_c = c_epi.copy()
        
        remaining = deadline - time.time()
        if remaining > 7.0:
            c_final, r_final = _polish(best_c, final_deadline, t_start=0.01, t_end=0.00005)
            if r_final > best_r:
                best_r = r_final
                best_c = c_final.copy()
            
        remaining = deadline - time.time()
        if remaining > 5.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=180)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        remaining = deadline - time.time()
        if remaining > 3.0:
            c_final2, r_final2 = _polish(best_c, final_deadline - 0.5, t_start=0.001, t_end=0.000005)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
        
        remaining = deadline - time.time()
        if remaining > 2.0:
            c_epi2, r_epi2 = _epigraph_polish(best_c, final_deadline - 0.2, t_offset=0.001)
            if r_epi2 > best_r:
                best_r = r_epi2
                best_c = c_epi2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
