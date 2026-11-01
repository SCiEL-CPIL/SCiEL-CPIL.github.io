"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like lattice, grid, random)
2. Repulsion-based simulated annealing for global search
3. Voronoi Lloyd's algorithm for balanced spacing
4. Softmin gradient polishing for fine-tuning
5. Local search on binding constraints
6. Epigraph formulation with softplus smoothing for final precision
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi
from scipy.spatial import cKDTree

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _seed_fcc():
    """
    Generate an FCC-inspired initial configuration for 50 spheres.
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
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_grid():
    """
    Generate a grid-based initial configuration.
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

def _seed_bcc():
    """
    Generate a BCC-inspired initial configuration.
    """
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                # BCC: corner + center
                points.append([x, y, z])
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_shell_core():
    """
    Generate a shell-core initial configuration.
    """
    n_core = 20
    n_shell = N - n_core
    
    core = np.random.uniform(-0.25, 0.25, size=(n_core, 3))
    
    theta = np.random.uniform(0, 2 * np.pi, size=n_shell)
    phi = np.random.uniform(0, np.pi, size=n_shell)
    r = 0.35 + np.random.uniform(-0.05, 0.05, size=n_shell)
    shell = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    
    return np.vstack([core, shell])

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
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
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Use log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)
    
    # Weights for pairs
    pair_weights = np.exp(-pair_vals / t)
    
    # Weights for walls
    wall_weights = np.exp(-wall_vals / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m_val, np.zeros_like(c)
        
    # Softmin value
    softmin_val = -t * np.log(total_sum)
    
    # Gradient calculation
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # For each pair (i,j), the gradient contribution to c[i] is -0.5 * w * (c[i]-c[j])/d
    # and to c[j] is +0.5 * w * (c[i]-c[j])/d
    pair_grad = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    
    for idx in range(_NUM_PAIRS):
        i, j = _PAIR_IDX[idx]
        grad_c[i] -= pair_grad[idx]
        grad_c[j] += pair_grad[idx]
    
    # Wall contributions
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    # Normalize by total_sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=50):
    """
    Repulsion-based simulated annealing.
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
            diff = c[_iu] - c[_ju]
            dists = np.sqrt(np.sum(diff * diff, axis=-1))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / (dists * dists)
            force_mag = np.where(force_mag > 1e6, 1e6, force_mag)
            
            unit_vecs = diff / dists[:, np.newaxis]
            unit_vecs[np.isnan(unit_vecs)] = 0.0
            
            pair_forces = force_mag[:, np.newaxis] * unit_vecs
            
            for idx in range(_NUM_PAIRS):
                i, j = _PAIR_IDX[idx]
                forces[i] += pair_forces[idx]
                forces[j] -= pair_forces[idx]
            
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
            if delta_r > 0 or np.random.random() < np.exp(-delta_r / max(t, 1e-8)):
                c = c_new
    
    return best_c, best_r

def _lloyds_algorithm(c_init, deadline, max_iters=200, step_size=0.1):
    """
    Lloyd's algorithm variant: move each center toward the centroid of its
    Voronoi cell (clipped to the cube) to equalize clearance.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Add boundary points for Voronoi computation
    boundary_points = []
    eps = 0.1
    
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

def _local_search(c_init, deadline, max_iters=300):
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
             
        diff = c[_iu] - c[_ju]
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
            i, j = _iu[min_idx], _ju[min_idx]
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

def _softmin_epigraph_polish(c_init, deadline):
    """
    High-precision polish using epigraph formulation with soft-min smoothing.
    Objective: minimize -t + lambda * sum(softplus(t - constraint_i)^2)
    This smooths the non-differentiable min in the constraints, allowing
    gradient-based optimization to find a more precise local optimum.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    # Start t slightly below current radius to ensure feasibility initially
    t0 = best_r - 0.0005
    x0 = np.concatenate([c.flatten(), [t0]])
    
    bounds = [(-0.499999, 0.499999)] * (N * 3) + [(0.0, 0.5)]
    
    # Precompute indices for speed
    iu = _iu
    ju = _ju
    n_pairs = _NUM_PAIRS
    
    def _softplus(x):
        # Numerically stable softplus: log(1 + exp(x))
        return np.where(x > 30, x, np.log1p(np.exp(x)))
    
    def _softplus_grad(x):
        # sigmoid(x)
        return np.where(x > 30, 1.0, 1.0 / (1.0 + np.exp(-x)))
    
    def obj(x, lam=100.0):
        c_x = x[:N*3].reshape(N, 3)
        t_val = x[N*3]
        
        # Compute all constraint values
        diff = c_x[iu] - c_x[ju]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        pair_vals = dists / 2.0
        
        wall_vals = (0.5 - np.abs(c_x)).ravel()
        
        all_vals = np.concatenate([pair_vals, wall_vals])
        
        # softplus(t - val) for each constraint
        sp_vals = _softplus(t_val - all_vals)
        
        return -t_val + lam * np.sum(sp_vals ** 2)
    
    def grad(x, lam=100.0):
        c_x = x[:N*3].reshape(N, 3)
        t_val = x[N*3]
        
        # Compute all constraint values
        diff = c_x[iu] - c_x[ju]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        pair_vals = dists / 2.0
        
        wall_vals = (0.5 - np.abs(c_x)).ravel()
        
        all_vals = np.concatenate([pair_vals, wall_vals])
        
        # Compute softplus and its gradient
        sp_vals = _softplus(t_val - all_vals)
        sp_grad = _softplus_grad(t_val - all_vals)
        
        # d/dt: -1 + 2*lam*sum(sp_vals * sp_grad)
        grad_t = -1.0 + 2.0 * lam * np.sum(sp_vals * sp_grad)
        
        # d/d(c): -2*lam*sum(sp_vals * sp_grad * d(val)/d(c))
        # For pair (i,j): d(pair_val)/d(c_i) = (c_i - c_j)/(2*dist)
        #                 d(pair_val)/d(c_j) = (c_j - c_i)/(2*dist)
        # For wall: d(wall_val)/d(c_ik) = -sign(c_ik)
        
        grad_c = np.zeros_like(c_x)
        
        # Pair contributions
        pair_sp = sp_vals[:n_pairs]
        pair_sp_grad = sp_grad[:n_pairs]
        
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        
        # factor[k] = -2 * lam * pair_sp[k] * pair_sp_grad[k] / (2 * dist[k])
        #           = -lam * pair_sp[k] * pair_sp_grad[k] / dist[k]
        factor = -lam * pair_sp * pair_sp_grad / safe_dists
        
        diff_pair = c_x[iu] - c_x[ju]
        
        # d(pair_val)/d(c_i) = (c_i - c_j) / (2*dist)
        # So d(obj)/d(c_i) += factor[k] * (c_i - c_j) / 2 ... wait
        # Let me redo:
        # obj = -t + lam * sum(sp(t - val)^2)
        # d(obj)/d(c_i) = lam * sum_k 2*sp_k * sp_grad_k * d(t - val_k)/d(c_i)
        #              = lam * sum_k 2*sp_k * sp_grad_k * (-d(val_k)/d(c_i))
        # For pair k=(i,j): d(val_k)/d(c_i) = (c_i - c_j)/(2*dist)
        # So d(obj)/d(c_i) = -lam * sum_k 2*sp_k*sp_grad_k * (c_i - c_j)/(2*dist)
        #                  = -lam * sum_k sp_k*sp_grad_k * (c_i - c_j)/dist
        
        factor2 = -lam * pair_sp * pair_sp_grad / safe_dists
        
        np.add.at(grad_c, iu, factor2[:, np.newaxis] * diff_pair)
        np.add.at(grad_c, ju, -factor2[:, np.newaxis] * diff_pair)
        
        # Wall contributions
        wall_sp = sp_vals[n_pairs:]
        wall_sp_grad = sp_grad[n_pairs:]
        
        signs = np.sign(c_x)
        signs[np.abs(c_x) < 1e-10] = 0.0
        
        # d(wall_val)/d(c_ik) = -sign(c_ik)
        # d(obj)/d(c_ik) = -lam * 2 * wall_sp * wall_sp_grad * (-sign(c_ik))
        #                = lam * 2 * wall_sp * wall_sp_grad * sign(c_ik)
        
        wall_grad = (2.0 * lam * wall_sp * wall_sp_grad).reshape(N, 3) * signs
        grad_c += wall_grad
        
        return np.concatenate([grad_c.ravel(), [grad_t]])
    
    try:
        res = minimize(
            obj,
            x0,
            jac=grad,
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    seeds = []
    seeds.append(_seed_fcc())
    seeds.append(_seed_bcc())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_random())
    seeds.append(_seed_grid())
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    total_time = deadline - start_time
    time_per_seed = total_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed * (1.0 + 0.1 * i), deadline - 20.0)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 2: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_anneal, seed_deadline, max_iters=80)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 3: Polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.001)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=150)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        if time.time() > deadline - 20.0:
            break
            
    # Final high-precision polish on the global best if time remains
    if time.time() < deadline - 20.0:
        final_deadline = deadline - 3.0
        
        # Epigraph polish first
        c_epi, r_epi = _softmin_epigraph_polish(best_c, min(final_deadline, time.time() + 30.0))
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
            c_epi2, r_epi2 = _softmin_epigraph_polish(best_c, final_deadline - 0.5)
            if r_epi2 > best_r:
                best_r = r_epi2
                best_c = c_epi2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c
