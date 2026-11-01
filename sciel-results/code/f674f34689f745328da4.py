"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-start with diverse seeds, epigraph-based smooth optimization,
Delaunay-guided local search, and simulated annealing for escape.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient using log-sum-exp formulation.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-16)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Use log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    max_val = np.max(all_vals)
    
    pair_weights = np.exp(-(pair_vals - max_val) / t)
    wall_weights = np.exp(-(wall_vals - max_val) / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        m = min(np.min(pair_vals), np.min(wall_vals))
        return m, np.zeros_like(c)
        
    softmin_val = max_val - t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    grad_contrib = -0.5 * pair_weights[:, np.newaxis] * unit_vecs
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        grad_c[i] += grad_contrib[idx]
        grad_c[j] -= grad_contrib[idx]
    
    signs = np.sign(c)
    grad_wall_contrib = wall_weights * signs
    grad_c += grad_wall_contrib
    
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _epigraph_objective(x, t=0.01):
    """
    Epigraph formulation: minimize -r + penalty for constraint violations.
    x = [c_0, ..., c_{N-1}, r] where c_i are 3D centers and r is the radius.
    """
    c = x[:N*3].reshape(N, 3)
    r = x[N*3]
    
    # Compute softmin radius
    r_soft, _ = _softmin_radius_and_grad(c, t)
    
    # Penalty for r being larger than achievable
    penalty = max(0, r - r_soft) ** 2
    
    # Objective: maximize r, so minimize -r + penalty
    return -r + 100.0 * penalty

def _epigraph_grad(x, t=0.01):
    """
    Gradient of epigraph objective.
    """
    c = x[:N*3].reshape(N, 3)
    r = x[N*3]
    
    _, grad_c = _softmin_radius_and_grad(c, t)
    
    r_soft, _ = _softmin_radius_and_grad(c, t)
    penalty = max(0, r - r_soft)
    
    # Gradient of -r is -1 for the r component
    grad_r = -1.0
    if penalty > 0:
        grad_r += 200.0 * penalty
    
    # Gradient of penalty w.r.t. c
    grad_penalty_c = np.zeros_like(c)
    if penalty > 0:
        grad_penalty_c = -200.0 * penalty * grad_c
    
    grad = np.zeros_like(x)
    grad[:N*3] = grad_penalty_c.ravel()
    grad[N*3] = grad_r
    
    return grad

def _delaunay_refinement(c, deadline, max_iters=100):
    """
    Delaunay-based refinement: identify tight constraints and optimize locally.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    iu = np.triu_indices(N, k=1)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Build Delaunay triangulation
        try:
            tri = Delaunay(c)
            # Build adjacency
            adj = [set() for _ in range(N)]
            for simplex in tri.simplices:
                for i in range(4):
                    for j in range(i+1, 4):
                        a, b = simplex[i], simplex[j]
                        adj[a].add(b)
                        adj[b].add(a)
        except Exception:
            break
        
        # Find the most constraining pair or wall
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        r = _compute_radius(c)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Wall constraint is tighter
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            
            # Try moving point away from wall
            for step in (0.001, 0.002, 0.005, 0.01):
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
            # Pair constraint is tighter - focus on Delaunay neighbors
            # Find tightest pair among Delaunay neighbors
            best_pair_idx = -1
            best_pair_dist = np.inf
            for i in range(N):
                for j in adj[i]:
                    if j > i:
                        d = np.linalg.norm(c[i] - c[j])
                        if d < best_pair_dist:
                            best_pair_dist = d
                            best_pair_idx = i * N + j
            
            if best_pair_idx >= 0:
                i = best_pair_idx // N
                j = best_pair_idx % N
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
                    c_try = np.clip(c_try, -0.4999, 0.4999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
                        break
    
    return best_c, best_r

def _lloyd_step(c, iterations=30, clip_margin=0.001):
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
                new_c[i] = c[i] * 0.9
                continue
            
            verts = vor.vertices[region]
            
            if len(verts) == 0:
                new_c[i] = c[i] * 0.9
                continue
            
            verts = np.clip(verts, -0.5 + clip_margin, 0.5 - clip_margin)
            centroid = np.mean(verts, axis=0)
            new_c[i] = 0.6 * c[i] + 0.4 * centroid
        
        new_c = np.clip(new_c, -0.4999, 0.4999)
        c = new_c
    
    return c

def _smooth_polish(c_init, deadline, t_start=0.02, t_end=0.001):
    """
    Polish the configuration using gradient ascent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    n_temps = 8
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
                obj, x0, jac=grad,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60}
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

def _simulated_annealing(c_init, deadline, n_steps=3000):
    """
    Simulated annealing to escape local minima.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temp = 0.05
    step_size = 0.03
    iu = np.triu_indices(N, k=1)
    
    for it in range(n_steps):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        i = np.random.randint(N)
        old_pos = c[i].copy()
        
        c[i] = old_pos + np.random.normal(0, step_size, size=3)
        c[i] = np.clip(c[i], -0.4999, 0.4999)
        r_after = _compute_radius(c)
        
        c[i] = old_pos
        r_before = _compute_radius(c)
        
        # Accept if better or with probability exp((r_after - r_before)/temp)
        if r_after > r_before or np.random.random() < np.exp((r_after - r_before) / temp):
            if r_after > best_r:
                best_r = r_after
                best_c = c.copy()
        else:
            c[i] = old_pos
        
        # Adaptive step size
        if r_after > best_r:
            step_size *= 1.005
        else:
            step_size *= 0.998
        step_size = max(0.001, min(step_size, 0.06))
        
        # Cool down
        temp *= 0.9997
        temp = max(temp, 0.005)
    
    return best_c, best_r

def _seed_fcc():
    """
    Generate an FCC-like initial configuration.
    """
    a = 1.0 / 4.0
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
                if i < 3:
                    points.append((i*a - 0.5 + a/2, j*a - 0.5, k*a - 0.5))
                if j < 3:
                    points.append((i*a - 0.5, j*a - 0.5 + a/2, k*a - 0.5))
                if k < 3:
                    points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5 + a/2))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
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
    core = np.random.uniform(-0.15, 0.15, size=(8, 3))
    c[:8] = core
    shell = np.random.uniform(-0.45, 0.45, size=(42, 3))
    c[8:] = shell
    return c

def _seed_bcc():
    """
    Generate a BCC-like initial configuration.
    """
    a = 1.0 / 3.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append((i*a - 0.5 + a/2, j*a - 0.5 + a/2, k*a - 0.5 + a/2))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                points.append((i*a - 0.5, j*a - 0.5, k*a - 0.5))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])

def _seed_hcp():
    """
    Generate an HCP-like initial configuration.
    """
    a = 1.0 / 4.0
    points = []
    for layer in range(5):
        z = layer * a * np.sqrt(2/3) - 0.5 + a/2
        if layer % 2 == 0:
            for i in range(4):
                for j in range(4):
                    x = i * a - 0.5 + a/2
                    y = j * a - 0.5 + a/2
                    points.append((x, y, z))
        else:
            for i in range(4):
                for j in range(4):
                    x = i * a - 0.5
                    y = j * a - 0.5
                    points.append((x, y, z))
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.49, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx] + np.random.normal(0, 0.02, size=(N, 3))
    else:
        pad = np.random.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, pad])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_shell_core(),
        _seed_bcc(),
        _seed_hcp(),
        _seed_random(),
    ]
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    # Allocate time: reserve 25% for final polish
    total_time = deadline - start_time
    final_time = total_time * 0.25
    search_time = total_time - final_time
    time_per_seed = search_time / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, deadline - final_time - 5.0)
        
        if time.time() > seed_deadline:
            continue
        
        # Stage 1: Lloyd's algorithm
        c_lloyd = _lloyd_step(s, iterations=25)
        c_lloyd = np.clip(c_lloyd, -0.4999, 0.4999)
        r_lloyd = _compute_radius(c_lloyd)
        
        if r_lloyd > best_r:
            best_r = r_lloyd
            best_c = c_lloyd.copy()
        
        # Stage 2: Smooth polish
        c_polish, r_polish = _smooth_polish(c_lloyd, seed_deadline, t_start=0.03, t_end=0.001)
        
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
        
        # Stage 3: Simulated annealing
        c_sa, r_sa = _simulated_annealing(c_polish, seed_deadline, n_steps=1500)
        
        if r_sa > best_r:
            best_r = r_sa
            best_c = c_sa.copy()
        
        # Stage 4: Delaunay refinement
        c_dr, r_dr = _delaunay_refinement(c_sa, seed_deadline, max_iters=150)
        
        if r_dr > best_r:
            best_r = r_dr
            best_c = c_dr.copy()
            
    # Final polish on the global best if time remains
    final_deadline = deadline - 5.0
    if time.time() < final_deadline:
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.003, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        if time.time() < final_deadline - 3.0:
            c_dr, r_dr = _delaunay_refinement(best_c, final_deadline, max_iters=300)
            if r_dr > best_r:
                best_r = r_dr
                best_c = c_dr.copy()
                
        if time.time() < final_deadline - 2.0:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.0008, t_end=0.00002)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
        # One more Delaunay refinement pass after the final polish
        if time.time() < final_deadline - 1.0:
            c_dr2, r_dr2 = _delaunay_refinement(best_c, final_deadline, max_iters=200)
            if r_dr2 > best_r:
                best_r = r_dr2
                best_c = c_dr2.copy()
                
    return best_c

# EVOLVE-BLOCK-END
