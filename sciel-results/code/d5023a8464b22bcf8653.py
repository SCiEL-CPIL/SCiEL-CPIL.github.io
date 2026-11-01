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
from scipy.spatial import Voronoi

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
    Returns (radius, gradient) where gradient is (N, 3).
    softmin(x) = -t * log(sum(exp(-x/t)))
    """
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Concatenate all constraints
    all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(c)
        
    # Stable log-sum-exp
    m = np.min(all_vals)
    s = np.sum(np.exp(-(all_vals - m) / t))
    radius = m - t * np.log(s)
    
    # Gradient computation
    # d(softmin)/d(c_i) = -sum_j w_j * d(constraint_j)/d(c_i)
    # where w_j = exp(-constraint_j/t) / sum_k exp(-constraint_k/t)
    
    exp_vals = np.exp(-(all_vals - m) / t)
    weights = exp_vals / s  # normalized weights
    
    grad = np.zeros_like(c)
    
    # Process pair distances
    pair_flat = pair_vals.ravel()
    pair_weights = weights[:len(pair_flat)]
    
    # Map flat index back to (i, j)
    # pair_vals is (N, N) with diagonal inf, so we have N*N entries
    # But we only kept finite ones. Let's recompute properly.
    
    # Actually, let's compute gradient directly from the pair and wall terms
    grad = np.zeros_like(c)
    
    # Pair contribution
    # For pair (i,j): constraint = dist(i,j)/2
    # d(dist(i,j)/2)/d(c_i) = (c_i - c_j) / (2 * dist(i,j))
    # d(dist(i,j)/2)/d(c_j) = (c_j - c_i) / (2 * dist(i,j))
    
    # We need to sum over all pairs with their weights
    # Reconstruct pair_vals with inf diagonal for indexing
    pair_vals_full = pair_vals.copy()  # (N, N)
    
    # Get weights for pairs (excluding diagonal)
    # all_vals = [pair_vals.ravel(), wall_vals.ravel()]
    # So first N*N entries are pairs, next N*3 are walls
    
    n_pairs = N * N
    pair_w = weights[:n_pairs].reshape(N, N)
    
    # For each pair (i, j), weight is pair_w[i, j]
    # Gradient contribution to c_i from pair (i,j):
    # -pair_w[i,j] * (c_i - c_j) / (2 * dist(i,j))
    
    # Vectorized computation
    # dists is (N, N)
    # We need to avoid division by zero (diagonal)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # (c_i - c_j) / (2 * dist(i,j)) for all i, j
    # diff[i,j] = c[i] - c[j]
    # grad_pair[i] = sum_j pair_w[i,j] * diff[i,j] / (2 * safe_dists[i,j])
    
    # But note: softmin = -t * log(sum(exp(-x/t)))
    # d(softmin)/dx = -sum w_k * dx_k/dx where w_k = exp(-x_k/t)/sum
    # So d(softmin)/d(c_i) = -sum_k w_k * d(constraint_k)/d(c_i)
    
    # For pair (i,j): d(constraint)/d(c_i) = (c_i - c_j)/(2*dist) = diff[i,j]/(2*safe_dists[i,j])
    # For pair (j,i): d(constraint)/d(c_i) = (c_i - c_j)/(2*dist) = -diff[j,i]/(2*safe_dists[j,i])
    
    # Since dist is symmetric, safe_dists[i,j] = safe_dists[j,i]
    # diff[i,j] = -diff[j,i]
    
    # Contribution to grad[i] from all pairs:
    # -sum_j pair_w[i,j] * diff[i,j]/(2*safe_dists[i,j]) - sum_j pair_w[j,i] * (-diff[j,i]/(2*safe_dists[j,i]))
    # = -sum_j pair_w[i,j] * diff[i,j]/(2*safe_dists[i,j]) + sum_j pair_w[j,i] * diff[i,j]/(2*safe_dists[i,j])
    # = sum_j (pair_w[j,i] - pair_w[i,j]) * diff[i,j]/(2*safe_dists[i,j])
    
    # This is getting complex. Let's just do it more directly.
    
    grad = np.zeros_like(c)
    
    # Pair contributions
    # For each i, sum over j != i:
    # -w_ij * (c_i - c_j)/(2*dist_ij) for the term where i is first index
    # -w_ji * (c_i - c_j)/(2*dist_ij) for the term where j is first index (but constraint is same)
    
    # Actually, constraint for pair (i,j) is the same as (j,i), both = dist/2
    # In all_vals, we have both pair_vals[i,j] and pair_vals[j,i]
    # So weight for constraint dist(i,j)/2 appears twice: once as w[i,j] and once as w[j,i]
    # Total weight for this constraint = w[i,j] + w[j,i]
    
    # d(constraint)/d(c_i) = (c_i - c_j)/(2*dist)
    # d(constraint)/d(c_j) = (c_j - c_i)/(2*dist)
    
    # So contribution to grad[i] from pair (i,j):
    # -(w[i,j] + w[j,i]) * (c_i - c_j)/(2*dist)
    
    # Let's compute this
    total_pair_w = pair_w + pair_w.T  # (N, N), total weight for each pair
    np.fill_diagonal(total_pair_w, 0.0)
    
    # grad[i] -= sum_j total_pair_w[i,j] * (c_i - c_j)/(2*safe_dists[i,j])
    # = -sum_j total_pair_w[i,j] * diff[i,j]/(2*safe_dists[i,j])
    
    # Vectorized:
    # factor[i,j] = total_pair_w[i,j] / (2 * safe_dists[i,j])
    factor = total_pair_w / (2.0 * safe_dists)
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = -sum_j factor[i,j] * diff[i,j]
    # diff is (N, N, 3)
    grad -= np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Wall contributions
    # wall_vals[i,k] = 0.5 - |c[i,k]|
    # d(wall_vals[i,k])/d(c[i,k]) = -sign(c[i,k])
    # d(wall_vals[i,k])/d(c[i,m]) = 0 for m != k
    
    wall_w = weights[n_pairs:]  # N*3 weights
    wall_w = wall_w.reshape(N, 3)
    
    # grad[i,k] -= wall_w[i,k] * (-sign(c[i,k]))
    # = wall_w[i,k] * sign(c[i,k])
    signs = np.sign(c)
    signs[np.abs(c) < 1e-10] = 0.0
    grad += wall_w * signs
    
    return radius, grad

def _repulsion_anneal(c_init, deadline, r_target=0.12, max_iters=5000):
    """
    Simulated annealing with repulsion to spread points apart.
    """
    c = c_init.copy()
    best_r = _compute_radius(c)
    best_c = c.copy()
    
    temp = 0.05
    decay = 0.998
    
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
        
        # Perturbation
        move = np.random.normal(0.0, 0.005, size=c.shape)
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
        
        temp *= decay
        
        # Periodically try to increase r_target
        if it % 500 == 0 and E < 1e-4:
            r_target += 0.002
    
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
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 15)
    
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
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
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

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    
    # 1. FCC-like
    seeds.append(_initial_fcc_like())
    
    # 2. Random
    seeds.append(np.random.uniform(-0.4, 0.4, size=(N, 3)))
    
    # 3. Grid-like
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    seeds.append(grid[idx])
    
    best_c = seeds[0]
    best_r = _compute_radius(best_c)
    
    # Allocate time to each seed
    time_per_seed = (deadline - start_time) / len(seeds)
    
    for i, s in enumerate(seeds):
        seed_start = time.time()
        seed_deadline = seed_start + time_per_seed
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, r_target=0.12, max_iters=3000)
        
        # Stage 2: Lloyd's algorithm
        c_lloyd, r_lloyd = _lloyds_algorithm(c_anneal, seed_deadline, max_iters=100)
        
        # Stage 3: Polish
        c_polish, r_polish = _polish(c_lloyd, seed_deadline, t_start=0.05, t_end=0.001)
        
        # Stage 4: Local Search
        c_ls, r_ls = _local_search(c_polish, seed_deadline, max_iters=200)
        
        # Update global best
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
            
        # If time is running out, break early
        if time.time() > deadline - 5.0:
            break
            
    # Final polish on the global best if time remains
    if time.time() < deadline - 10.0:
        final_deadline = deadline - 3.0
        c_final, r_final = _polish(best_c, final_deadline, t_start=0.02, t_end=0.0001)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
            
        # One last local search
        if time.time() < final_deadline - 3.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=500)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
                
    # Ensure all points are strictly inside the cube
    best_c = np.clip(best_c, -0.4999999, 0.4999999)
    
    return best_c

# EVOLVE-BLOCK-END
