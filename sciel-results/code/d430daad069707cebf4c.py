"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initialization (FCC-like, BCC, grid, random)
2. Simulated annealing with repulsion to escape local minima
3. DEM (Discrete Element Method) simulation with Hertzian contacts
4. Lloyd's algorithm (Voronoi centroid moves) to equalize clearance
5. Smooth gradient-based polishing (soft-min via log-sum-exp)
6. Final high-precision polish pass
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# DEM parameters
_DEM_K_CONTACT = 500.0      # Hertzian contact stiffness
_DEM_K_WALL = 200.0         # Wall repulsion stiffness
_DEM_DAMPING = 0.98         # Velocity damping per step
_DEM_DT_INIT = 0.0005       # Initial time step
_DEM_DT_MIN = 0.00001       # Minimum time step
_DEM_DT_MAX = 0.002         # Maximum time step

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
    
    # For each unique pair (i, j) with i < j:
    # constraint = dist(i,j)/2
    # d(constraint)/d(c_i) = (c_i - c_j) / dist(i,j)  [note: d(dist/2)/d(c_i) = (c_i-c_j)/dist]
    # Wait: d(dist)/d(c_i) = (c_i - c_j)/dist, so d(dist/2)/d(c_i) = (c_i-c_j)/(2*dist)
    # d(constraint)/d(c_j) = (c_j - c_i) / (2*dist)
    
    # grad[i] -= w * (c_i - c_j)/(2*dist)  for the i-index of the pair
    # grad[j] -= w * (c_j - c_i)/(2*dist)  for the j-index of the pair
    
    pi, pj = iu
    safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
    
    # Vectorized: for each pair, compute contribution
    # diff_pair[k] = c[pi[k]] - c[pj[k]]
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
    
    # d(wall_vals[i,k])/d(c[i,k]) = -sign(c[i,k])
    # grad[i,k] -= wall_w[i,k] * (-sign(c[i,k])) = wall_w[i,k] * sign(c[i,k])
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _repulsion_anneal(c_init, deadline, r_target=None, max_iters=5000):
    """
    Simulated annealing with repulsion to spread points apart.
    Uses adaptive perturbation size and tracks best radius.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    if r_target is None:
        r_target = best_r + 0.01
    
    temp = 0.05
    decay = 0.998
    pert_scale = 0.008
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute energy: sum of squared overlaps
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Target: all pair distances >= 2*r_target, wall distances >= r_target
        overlap_pair = np.maximum(0.0, 2.0 * r_target - dists)
        wall_dists = 0.5 - np.abs(c)
        overlap_wall = np.maximum(0.0, r_target - wall_dists)
        
        E = np.sum(overlap_pair ** 2) + np.sum(overlap_wall ** 2)
        
        # Adaptive perturbation: larger when stuck, smaller when improving
        move = np.random.normal(0.0, pert_scale, size=c.shape)
        c_new = c + move
        c_new = np.clip(c_new, -0.49, 0.49)
        
        # Compute new energy
        diff_new = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists_new = np.sqrt(np.sum(diff_new * diff_new, axis=-1))
        np.fill_diagonal(dists_new, np.inf)
        
        overlap_pair_new = np.maximum(0.0, 2.0 * r_target - dists_new)
        wall_dists_new = 0.5 - np.abs(c_new)
        overlap_wall_new = np.maximum(0.0, r_target - wall_dists_new)
        
        E_new = np.sum(overlap_pair_new ** 2) + np.sum(overlap_wall_new ** 2)
        
        # Accept
        dE = E_new - E
        if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-6)):
            c = c_new
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
                # Reduce perturbation on improvement
                pert_scale *= 0.95
            else:
                # Slightly increase perturbation if no improvement
                pert_scale = min(pert_scale * 1.02, 0.02)
        else:
            # No improvement, increase perturbation
            pert_scale = min(pert_scale * 1.03, 0.02)
        
        temp *= decay
        
        # Periodically try to increase r_target
        if it % 300 == 0 and E < 1e-4:
            r_target += 0.003
    
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
    # We need points outside the cube to define the Voronoi cells properly
    # Add points at the faces, edges, and corners extended outward
    
    # For Voronoi, we need to include the cube boundaries
    # Add points just outside each face
    boundary_points = []
    eps = 0.1
    
    # Face centers extended outward
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign * (0.5 + eps)
            boundary_points.append(p)
    
    # Edge midpoints extended outward
    for i in range(3):
        for j in range(i+1, 3):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = np.zeros(3)
                    p[i] = si * (0.5 + eps)
                    p[j] = sj * (0.5 + eps)
                    boundary_points.append(p)
    
    # Corners extended outward
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                p = np.array([sx * (0.5 + eps), sy * (0.5 + eps), sz * (0.5 + eps)])
                boundary_points.append(p)
    
    boundary_points = np.array(boundary_points)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute Voronoi diagram
        all_points = np.vstack([c, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except Exception:
            # Voronoi computation failed, break
            break
        
        # For each original point, find its Voronoi cell and compute centroid
        new_c = c.copy()
        
        for i in range(N):
            # Find the region corresponding to point i
            # vor.regions gives lists of vertex indices
            # vor.point_region[i] gives the region index for point i
            
            region_idx = vor.point_region[i]
            if region_idx == -1:
                continue
                
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            
            # Get vertices of this region
            verts = vor.vertices[region]
            
            # Clip vertices to the cube [-0.5, 0.5]^3
            verts_clipped = np.clip(verts, -0.5, 0.5)
            
            # Compute centroid of the clipped region
            # Simple centroid: mean of vertices (approximate)
            # Better: use the actual polygon area, but mean is a good approximation
            centroid = np.mean(verts_clipped, axis=0)
            
            # Move toward centroid
            new_c[i] = c[i] + step_size * (centroid - c[i])
            new_c[i] = np.clip(new_c[i], -0.499, 0.499)
        
        # Check if this improved the radius
        r_new = _compute_radius(new_c)
        if r_new > best_r:
            best_r = r_new
            best_c = new_c.copy()
            c = new_c.copy()
        else:
            # If it got worse, revert
            c = best_c.copy()
        
        # Decrease step size
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
    
    # Use more temperature steps for finer polishing
    n_temps = 20
    temps = np.logspace(np.log10(t_start), np.log10(t_end), n_temps)
    
    for ti, t in enumerate(temps):
        if time.time() > deadline:
            break
            
        # More iterations for later (smaller t) temperatures
        max_iter = 50 + 50 * (ti // (n_temps // 2))
        
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
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
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
             
        # Find the binding constraint
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall binding
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.002, 0.005, 0.01):
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
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            for step in (0.002, 0.005, 0.01):
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

def _initial_fcc_like():
    """
    Generate an FCC-like initial configuration for 50 points.
    FCC lattice in a cube has points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    scaled and shifted to [-0.5, 0.5]^3.
    """
    # FCC basis vectors
    a = 0.5  # lattice constant
    
    # Generate FCC points in a larger region, then select 50
    points = []
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                # FCC has 4 atoms per unit cell
                # Positions: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) * a
                base = np.array([i*a, j*a, k*a])
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0, 0.5*a, 0.5*a]),
                    np.array([0.5*a, 0, 0.5*a]),
                    np.array([0.5*a, 0.5*a, 0]),
                ]
                for off in offsets:
                    p = base + off
                    # Shift to center around origin
                    p_centered = p - np.array([a, a, a])
                    if np.all(np.abs(p_centered) < 0.6):
                        points.append(p_centered)
    
    points = np.array(points)
    
    # Randomly select 50
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]
    else:
        # Pad with random points
        remaining = N - len(points)
        random_pts = np.random.uniform(-0.4, 0.4, size=(remaining, 3))
        return np.vstack([points, random_pts])

def _initial_bcc_like():
    """
    Generate a BCC-like initial configuration for 50 points.
    BCC lattice has points at corners and body center of each cube.
    """
    a = 0.5  # lattice constant
    points = []
    
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                base = np.array([i*a, j*a, k*a])
                # BCC: corners + body center
                offsets = [
                    np.array([0, 0, 0]),
                    np.array([0.5*a, 0.5*a, 0.5*a]),
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

def _dem_simulate(c_init, deadline, r_target=None, max_iters=5000):
    """
    DEM (Discrete Element Method) simulation using Hertzian contact forces.
    Spheres are rigid bodies with repulsive forces when overlapping.
    Uses velocity-Verlet integration with adaptive time-stepping.
    """
    c = c_init.copy()
    v = np.zeros_like(c)  # velocities
    
    if r_target is None:
        r_target = _compute_radius(c) + 0.005
    
    dt = _DEM_DT_INIT
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    # Precompute pair indices for efficiency
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Hertzian contact: F = k * (2*r_target - dist)^2 * (dist > 2*r_target - delta)
        # Use smooth quadratic repulsion for overlapping pairs
        pair_dists = dists[iu]
        overlap = np.maximum(0.0, 2.0 * r_target - pair_dists)
        
        # Force magnitude (Hertzian: proportional to overlap^1.5, but we use quadratic for stability)
        force_mag = _DEM_K_CONTACT * overlap ** 2
        force_mag[overlap == 0] = 0.0
        
        # Force direction (from j to i, pushing apart)
        safe_dists = np.where(pair_dists > 1e-10, pair_dists, 1.0)
        dir_vecs = (c[iu[0]] - c[iu[1]]) / safe_dists[:, np.newaxis]
        
        forces = np.zeros_like(c)
        np.add.at(forces, iu[0], force_mag[:, np.newaxis] * dir_vecs)
        np.add.at(forces, iu[1], -force_mag[:, np.newaxis] * dir_vecs)
        
        # Wall repulsion
        wall_dists = 0.5 - np.abs(c)
        wall_overlap = np.maximum(0.0, r_target - wall_dists)
        wall_force_mag = _DEM_K_WALL * wall_overlap ** 2
        signs = np.sign(c)
        signs[np.abs(c) < 1e-10] = 0.0
        forces += wall_force_mag * signs
        
        # Velocity-Verlet integration (simplified: semi-implicit Euler for stability)
        v = v * _DEM_DAMPING + forces * dt
        c = c + v * dt
        
        # Clip to cube (with small margin to avoid exact boundary)
        c = np.clip(c, -0.4999, 0.4999)
        
        # Check for improvement
        if it % 50 == 0:
            r_check = _compute_radius(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
        
        # Adaptive time-stepping: reduce dt if velocities are too large
        max_vel = np.max(np.linalg.norm(v, axis=1))
        if max_vel * dt > 0.01:
            dt = max(_DEM_DT_MIN, dt * 0.5)
        elif max_vel * dt < 0.001 and dt < _DEM_DT_MAX:
            dt = min(_DEM_DT_MAX, dt * 1.1)
        
        # Gradually increase target radius if energy is low
        if it % 500 == 0:
            total_overlap = np.sum(overlap ** 2) + np.sum(wall_overlap ** 2)
            if total_overlap < 1e-6:
                r_target += 0.002
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. FCC-like
    seeds.append(_initial_fcc_like())
    
    # 2. BCC-like
    seeds.append(_initial_bcc_like())
    
    # 3. Random
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # 4. Grid-like
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    seeds.append(grid[idx])
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: 4 seeds, give more time to later seeds
    total_time = deadline - start_time
    # Allocate ~20% to each seed, with increasing allocation
    time_weights = [0.20, 0.22, 0.25, 0.33]
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + total_time * time_weights[i]
        # Ensure we don't exceed global deadline
        seed_deadline = min(seed_deadline, deadline - 10.0)
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=best_r + 0.005, max_iters=2000)
        
        # Stage 2: DEM simulation (new)
        c_dem, r_dem = _dem_simulate(c_anneal, seed_deadline, r_target=best_r + 0.003, max_iters=3000)
        
        # Use the better of anneal and DEM
        if r_dem > r_anneal:
            c_stage = c_dem
        else:
            c_stage = c_anneal
        
        # Stage 3: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_stage, seed_deadline, max_iters=80)
        
        # Stage 4: Polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.001)
        
        # Stage 5: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=150)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 20.0:
            break
            
    # Final high-precision polish on the global best if time remains
    if time.time() < deadline - 15.0:
        final_deadline = deadline - 3.0
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.01, t_end=0.00005)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # DEM polish pass
        if time.time() < final_deadline - 5.0:
            c_dem_final, r_dem_final = _dem_simulate(best_c, final_deadline - 3.0, r_target=best_r + 0.001, max_iters=2000)
            if r_dem_final > best_r:
                best_r = r_dem_final
                best_c = c_dem_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=300)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
        # Final ultra-fine polish
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _polish(best_c, final_deadline - 1.0, t_start=0.001, t_end=0.00001)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
