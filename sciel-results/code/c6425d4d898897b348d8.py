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
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + 115.0  # 5s safety margin before 120s kill

# EVOLVE-BLOCK-START

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    r = min( min_{i<j} dist(i,j)/2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(c) == 0:
        return 0.0
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Uses upper-triangle pairs to avoid double-counting.
    Returns (radius, gradient) where gradient is (N, 3).
    """
    n = len(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Use only upper triangle for pairs (avoid double counting)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    pair_vals = pair_dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m = np.min(all_vals)
    
    # Shift for numerical stability
    pair_shifted = np.exp(-(pair_vals - m) / t)
    wall_shifted = np.exp(-(wall_vals.ravel() - m) / t)
    
    total_sum = np.sum(pair_shifted) + np.sum(wall_shifted)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
    
    softmin_val = m - t * np.log(total_sum)
    
    grad_c = np.zeros_like(c)
    
    # Pair gradient: for each pair (i,j), d(dist_ij/2)/dc_i = (c_i - c_j)/dist_ij
    # and d(dist_ij/2)/dc_j = (c_j - c_i)/dist_ij
    # Using upper triangle, we add contributions to both i and j
    pair_weights = pair_shifted / total_sum
    dists_iu = pair_dists
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists = 1.0 / dists_iu
    inv_dists[np.isnan(inv_dists)] = 0.0
    
    # For each pair (i,j): grad wrt c_i += w * (c_i - c_j) / dist_ij
    # grad wrt c_j += w * (c_j - c_i) / dist_ij
    ii = iu[0]
    jj = iu[1]
    vecs = diff[ii, jj]  # c_i - c_j
    contrib = pair_weights * inv_dists
    
    for idx in range(len(ii)):
        i = ii[idx]
        j = jj[idx]
        w = contrib[idx]
        grad_c[i] += w * vecs[idx]
        grad_c[j] -= w * vecs[idx]
    
    # Wall gradient
    wall_weights_flat = wall_shifted / total_sum
    wall_weights = wall_weights_flat.reshape(n, 3)
    signs = np.sign(c)
    grad_c += wall_weights * signs
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, cooling=0.98, max_iters=5000):
    """
    Simulated annealing with repulsion forces to find a good global configuration.
    Uses a potential energy that penalizes close approaches and wall violations.
    Optimized: batch evaluation, fewer redundant computations.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    def energy(x):
        """Repulsion energy: sum of (1/d)^4 for pairs + wall penalties"""
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1e20)
        # Strong repulsion at short range: (1/d)^4 = 1/d^4 = 1/(d^2)^2
        pair_energy = np.sum(1.0 / (dists_sq ** 2))
        # Wall penalty
        wall_dists = 0.5 - np.abs(x)
        wall_penalty = np.sum(np.maximum(0, 0.05 - wall_dists) ** 2)
        return pair_energy + 100 * wall_penalty
    
    e_curr = energy(c)
    t = t_start
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Propose a move: perturb a random point
        i = np.random.randint(N)
        delta = np.random.normal(0, 0.05 * t, size=3)
        old_val = c[i].copy()
        c[i] += delta
        c[i] = np.clip(c[i], -0.4999, 0.4999)
        
        # Evaluate new energy
        e_new = energy(c)
        de = e_new - e_curr
        
        if de < 0 or np.random.random() < np.exp(-de / t):
            e_curr = e_new
        else:
            c[i] = old_val
        
        # Update best periodically (every 50 iters)
        if it % 50 == 0:
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        
        t *= cooling
        if t < t_end:
            t = t_end
    
    # Final check
    r = _compute_radius(c)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    return best_c, best_r

def _lloyd_step(c, step_size=0.5):
    """
    Perform one Lloyd's algorithm iteration: move each center toward the centroid
    of its Voronoi cell, clipped to the cube boundary.
    """
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    
    c_new = c.copy()
    
    for i in range(N):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            continue
            
        verts = vor.vertices[region]
        
        if len(verts) == 0:
            continue
            
        centroid = np.mean(verts, axis=0)
        centroid = np.clip(centroid, -0.499999, 0.499999)
        
        c_new[i] = c[i] + step_size * (centroid - c[i])
        c_new[i] = np.clip(c_new[i], -0.499999, 0.499999)
    
    return c_new

def _lloyd_optimize(c_init, deadline, max_iters=300, step_decay=0.995):
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
        
        # Check for coincident points
        diff = c_new[:, np.newaxis, :] - c_new[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        if np.min(dists_sq) < 1e-12:
            c_new += np.random.normal(0, 1e-4, size=c_new.shape)
            c_new = np.clip(c_new, -0.499999, 0.499999)
        
        r_new = _compute_radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new.copy()
        else:
            step_size *= 0.9
        
        step_size *= step_decay
        step_size = max(step_size, 0.001)
    
    return best_c, best_r

def _polish(c_init, deadline, t_start=0.05, t_end=0.001):
    """
    Polish the configuration using gradient descent on the softmin radius.
    More temperature stages and tighter tolerances for better convergence.
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
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
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
    Improved: try multiple directions and smaller steps.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist_sq = np.min(dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] > 0 else -1.0
            for step in (0.002, 0.005, 0.01, 0.02):
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
            idx = np.unravel_index(np.argmin(dists_sq), dists_sq.shape)
            i, j = idx
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            # Try moving apart along the connecting vector and perpendicular
            directions = [vec, -vec]
            # Add a few random perpendicular directions
            for _ in range(2):
                rand_vec = np.random.normal(size=3)
                rand_vec -= np.dot(rand_vec, vec) * vec
                rand_norm = np.linalg.norm(rand_vec)
                if rand_norm > 1e-8:
                    directions.append(rand_vec / rand_norm)
            
            for d in directions:
                for step in (0.002, 0.005, 0.01):
                    c_try = c.copy()
                    c_try[i] += step * d
                    c_try[j] -= step * d
                    c_try = np.clip(c_try, -0.499999, 0.499999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
                        break
                if best_r > _compute_radius(c_init):
                    break
    return best_c, best_r

def _seed_fcc():
    """
    Create an FCC-inspired initial configuration.
    FCC has 4 atoms per unit cell. We'll create a 4x4x4 grid of FCC-like positions.
    """
    # FCC basis positions in a unit cell
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    # Create positions in a 4x4x4 supercell
    positions = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for b in basis:
                    pos = np.array([i, j, k]) + b
                    positions.append(pos)
    
    positions = np.array(positions)
    # Normalize to fit in [-0.5, 0.5]^3
    positions = positions - positions.mean(axis=0)
    positions = positions * (1.0 / np.max(np.abs(positions))) * 0.45
    
    # Select 50 random positions
    idx = np.random.choice(len(positions), N, replace=False)
    return positions[idx]

def _seed_grid():
    """
    Create a grid-based initial configuration.
    """
    # Try different grid configurations
    configs = [
        (5, 5, 2),  # 50 points
        (5, 2, 5),
        (2, 5, 5),
        (4, 4, 4),  # 64 points, select 50
    ]
    
    best = None
    best_r = -1
    
    for (nx, ny, nz) in configs:
        if nx * ny * nz < N:
            continue
        
        ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
        ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
        ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
        
        grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
        
        if len(grid) > N:
            idx = np.random.choice(len(grid), N, replace=False)
            grid = grid[idx]
        
        r = _compute_radius(grid)
        if r > best_r:
            best_r = r
            best = grid
    
    return best if best is not None else np.random.uniform(-0.4, 0.4, size=(N, 3))

def _seed_random():
    """Random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_shell_core():
    """
    Create a shell-core configuration: some points on a spherical shell,
    others in the core. This can help with wall constraints.
    """
    # 10 points on a spherical shell
    shell_pts = []
    for i in range(10):
        # Random point on sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = 0.4
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        shell_pts.append([x, y, z])
    
    # 40 points in the core
    core_pts = np.random.uniform(-0.3, 0.3, size=(40, 3))
    
    return np.vstack([shell_pts, core_pts])

def _optimize_single(c_init, deadline):
    """
    Run the full optimization pipeline on a single initial configuration.
    """
    # Stage 1: Repulsion annealing for global search
    c1, r1 = _repulsion_anneal(c_init, deadline, t_start=1.0, t_end=0.05, max_iters=2000)
    
    # Stage 2: Lloyd's algorithm for balanced spacing
    c2, r2 = _lloyd_optimize(c1, deadline, max_iters=200)
    
    # Stage 3: Polish with softmin gradient
    c3, r3 = _polish(c2, deadline, t_start=0.05, t_end=0.005)
    
    # Stage 4: Local search on binding constraints
    c4, r4 = _local_search(c3, deadline, max_iters=200)
    
    return c4, r4

def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _get_deadline()
    
    # Generate diverse initial configurations
    seeds = [
        _seed_fcc(),
        _seed_grid(),
        _seed_random(),
        _seed_shell_core(),
        np.random.uniform(-0.45, 0.45, size=(N, 3)),
        np.random.uniform(-0.45, 0.45, size=(N, 3)),
    ]
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time: give more time to earlier seeds (they benefit from full pipeline)
    # First 3 seeds get more time, last 3 get less
    time_budget = deadline - time.time()
    weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    total_weight = sum(weights)
    
    for i, s in enumerate(seeds):
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        
        # Time for this seed: proportional weight of remaining budget
        seed_time = time_budget * weights[i] / total_weight
        seed_deadline = time.time() + seed_time
        
        c_opt, r_opt = _optimize_single(s, seed_deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # Update remaining budget
        time_budget = deadline - time.time()
    
    # Final polishing on the global best if time remains
    if time.time() < deadline - 3.0:
        final_deadline = deadline - 1.0
        
        # Try Lloyd's again on the best configuration
        c_lloyd2, r_lloyd2 = _lloyd_optimize(best_c, final_deadline, max_iters=200)
        if r_lloyd2 > best_r:
            best_r = r_lloyd2
            best_c = c_lloyd2.copy()
        
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.03, t_end=0.002)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 1.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=400)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    return best_c

# EVOLVE-BLOCK-END
