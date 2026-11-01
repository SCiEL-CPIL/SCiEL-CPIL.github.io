"""Packing 50 equal spheres in a unit cube using smooth optimization with annealing."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _softmin_and_grad(x, beta):
    """
    Smooth minimum of all pairwise distances and wall distances.
    x: flat array of 3N coordinates.
    Returns (softmin_value, gradient).
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Wall distances: for each center, distance to nearest wall
    # w_i = min over 6 walls of (0.5 - |c_i_k|)
    # We'll compute all 6 wall distances per center
    abs_c = np.abs(centers)  # (n, 3)
    # 6 walls: -x, +x, -y, +y, -z, +z
    wall_dists = np.concatenate([
        0.5 - abs_c[:, 0:1],
        0.5 + abs_c[:, 0:1],
        0.5 - abs_c[:, 1:2],
        0.5 + abs_c[:, 1:2],
        0.5 - abs_c[:, 2:3],
        0.5 + abs_c[:, 2:3],
    ], axis=1)  # (n, 6)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, n, 3)
    dists = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-30)  # (n, n)
    np.fill_diagonal(dists, np.inf)
    
    # Collect all distances into a flat array for softmin
    # Wall distances: n*6 values
    # Pair distances: n*(n-1)/2 values (upper triangle)
    wall_flat = wall_dists.flatten()  # (n*6,)
    triu_idx = np.triu_indices(n, k=1)
    pair_flat = dists[triu_idx]  # (n*(n-1)/2,)
    
    all_dists = np.concatenate([wall_flat, pair_flat])  # (n*6 + n*(n-1)/2,)
    
    # Softmin: -logsumexp(-beta * d) / beta
    # For numerical stability
    min_d = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - min_d))
    logsum = np.log(np.sum(exp_terms))
    softmin_val = min_d - logsum / beta
    
    # Gradient of softmin with respect to x
    # d(softmin)/dx = -sum_j (exp(-beta*d_j) / sum_k exp(-beta*d_k)) * beta * d(d_j)/dx
    weights = exp_terms / np.sum(exp_terms)  # (m,) where m = len(all_dists)
    weights = beta * weights  # (m,)
    
    grad = np.zeros_like(x)
    
    # Wall contributions
    # For each center i and wall w, d/dx of wall_dist
    # wall_dist for center i, wall (axis, sign):
    #   -x wall: 0.5 - x_i -> d/dx_i = -1
    #   +x wall: 0.5 + x_i -> d/dx_i = +1
    #   -y wall: 0.5 - y_i -> d/dy_i = -1
    #   +y wall: 0.5 + y_i -> d/dy_i = +1
    #   -z wall: 0.5 - z_i -> d/dz_i = -1
    #   +z wall: 0.5 + z_i -> d/dz_i = +1
    
    wall_weights = weights[:n*6].reshape(n, 6)  # (n, 6)
    # Signs: [-1, +1, -1, +1, -1, +1] for the 6 walls
    signs = np.array([-1, 1, -1, 1, -1, 1])  # (6,)
    wall_grad = wall_weights * signs  # (n, 6)
    
    # Sum over walls for each coordinate
    # x-coordinate: walls 0 (-x) and 1 (+x)
    grad[0::3] += wall_grad[:, 0] + wall_grad[:, 1]
    # y-coordinate: walls 2 (-y) and 3 (+y)
    grad[1::3] += wall_grad[:, 2] + wall_grad[:, 3]
    # z-coordinate: walls 4 (-z) and 5 (+z)
    grad[2::3] += wall_grad[:, 4] + wall_grad[:, 5]
    
    # Pair contributions
    pair_weights = weights[n*6:]  # (n*(n-1)/2,)
    # For pair (i, j) with i < j, dist = |c_i - c_j|
    # d(dist)/dc_i = (c_i - c_j) / dist
    # d(dist)/dc_j = (c_j - c_i) / dist = -(c_i - c_j) / dist
    
    # Reconstruct pair distances and differences
    for idx, (i, j) in enumerate(zip(*triu_idx)):
        if idx >= len(pair_weights):
            break
        w = pair_weights[idx]
        dij = dists[i, j]
        if dij < 1e-15:
            continue
        diff_ij = centers[i] - centers[j]
        unit = diff_ij / dij
        grad[i*3:(i+1)*3] += w * unit
        grad[j*3:(j+1)*3] -= w * unit
    
    return softmin_val, grad

def _objective_and_grad(x, beta):
    """Objective: -softmin_radius + small regularization to keep in bounds."""
    softmin_val, grad = _softmin_and_grad(x, beta)
    
    # Small quadratic penalty for being near walls (helps with numerical stability)
    centers = x.reshape(-1, 3)
    penalty = 0.01 * np.sum(np.maximum(0, np.abs(centers) - 0.45)**2)
    penalty_grad = 0.02 * np.maximum(0, np.abs(centers) - 0.45) * np.sign(centers)
    
    obj = -softmin_val + penalty
    total_grad = -grad + penalty_grad.reshape(-1)
    
    return obj, total_grad

def _seed_grid():
    """5x5x2 grid initialization."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return centers

def _seed_bcc():
    """Body-centered cubic lattice initialization."""
    # BCC lattice: corners + center of each cube
    # Scale to fit in [-0.5, 0.5]^3
    # Use a 4x4x4 BCC lattice scaled down
    n = 4
    spacing = 0.9 / (n - 1)
    centers = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = -0.45 + i * spacing
                y = -0.45 + j * spacing
                z = -0.45 + k * spacing
                centers.append([x, y, z])
                # Center of the cube cell
                if i < n-1 and j < n-1 and k < n-1:
                    cx = x + spacing/2
                    cy = y + spacing/2
                    cz = z + spacing/2
                    if cx < 0.45 and cy < 0.45 and cz < 0.45:
                        centers.append([cx, cy, cz])
    
    centers = np.array(centers[:N])
    # If we have more than N, take first N; if less, pad with random
    if len(centers) < N:
        rng = np.random.default_rng(123)
        extra = rng.uniform(-0.4, 0.4, (N - len(centers), 3))
        centers = np.vstack([centers, extra])
    
    return centers[:N]

def _seed_fcc():
    """Face-centered cubic lattice initialization."""
    # FCC lattice points in a unit cell: corners + face centers
    # Generate FCC points and scale
    n = 3
    spacing = 0.9 / n
    centers = []
    for i in range(n+1):
        for j in range(n+1):
            for k in range(n+1):
                x = -0.45 + i * spacing
                y = -0.45 + j * spacing
                z = -0.45 + k * spacing
                centers.append([x, y, z])
                # Face centers
                if i < n:
                    centers.append([x + spacing/2, y, z])
                if j < n:
                    centers.append([x, y + spacing/2, z])
                if k < n:
                    centers.append([x, y, z + spacing/2])
    
    centers = np.array(centers)
    # Shuffle and take N
    rng = np.random.default_rng(456)
    if len(centers) > N:
        idx = rng.choice(len(centers), N, replace=False)
        centers = centers[idx]
    elif len(centers) < N:
        extra = rng.uniform(-0.4, 0.4, (N - len(centers), 3))
        centers = np.vstack([centers, extra])
    
    return centers[:N]

def _seed_random(seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.4, 0.4, (N, 3))

def _repulsion_anneal(x0, time_budget=30.0):
    """
    Annealing phase: use a repulsion potential to spread out spheres.
    Start with strong repulsion (large beta) and anneal down.
    """
    x = x0.copy()
    start_time = time.time()
    
    # Annealing schedule: beta from high to low
    betas = [200, 100, 50, 25, 10, 5, 2]
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        def obj(x):
            val, grad = _objective_and_grad(x, beta)
            return val, grad
        
        try:
            result = minimize(
                obj, x, method='L-BFGS-B', jac=True,
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if np.all(np.isfinite(result.x)):
                x = result.x
        except Exception:
            pass
    
    return x

def _smooth_polish(x, time_budget=30.0):
    """
    Final polishing with a smooth objective.
    Use increasing beta to sharpen the softmin.
    """
    start_time = time.time()
    
    # Increase beta for sharper optimization
    betas = [2, 5, 10, 20, 50, 100]
    
    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        def obj(x):
            val, grad = _objective_and_grad(x, beta)
            return val, grad
        
        try:
            result = minimize(
                obj, x, method='L-BFGS-B', jac=True,
                bounds=[(-0.49, 0.49)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            if np.all(np.isfinite(result.x)):
                x = result.x
        except Exception:
            pass
    
    return x

def _local_search(x, time_budget=20.0):
    """
    Local search: try small random perturbations and keep if better.
    """
    best_x = x.copy()
    best_r = _compute_radius(best_x.reshape(-1, 3))
    start_time = time.time()
    
    rng = np.random.default_rng(789)
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        # Try a random perturbation
        x_try = best_x + rng.normal(0, step_size, best_x.shape)
        x_try = np.clip(x_try, -0.49, 0.49)
        
        r_try = _compute_radius(x_try.reshape(-1, 3))
        if r_try > best_r:
            best_x = x_try
            best_r = r_try
            step_size = max(step_size * 0.9, 0.001)
        else:
            step_size = max(step_size * 0.95, 0.001)
    
    return best_x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    best_x = None
    best_score = -np.inf
    
    # Try multiple initializations
    initializations = [
        ("grid", _seed_grid()),
        ("bcc", _seed_bcc()),
        ("fcc", _seed_fcc()),
        ("random_1", _seed_random(1)),
        ("random_2", _seed_random(2)),
    ]
    
    for name, centers in initializations:
        elapsed = time.time() - start_time
        if elapsed > deadline - 25:
            break
        
        x0 = centers.flatten()
        current_score = _compute_radius(centers)
        
        if best_x is None or current_score > best_score:
            best_x = x0.copy()
            best_score = current_score
        
        # Phase 1: Repulsion annealing
        if time.time() - start_time < deadline - 20:
            try:
                x_annealed = _repulsion_anneal(x0, time_budget=15.0)
                r_annealed = _compute_radius(x_annealed.reshape(-1, 3))
                if r_annealed > best_score:
                    best_score = r_annealed
                    best_x = x_annealed.copy()
            except Exception:
                pass
        
        # Phase 2: Smooth polish
        if best_x is not None and time.time() - start_time < deadline - 15:
            try:
                x_polished = _smooth_polish(best_x, time_budget=15.0)
                r_polished = _compute_radius(x_polished.reshape(-1, 3))
                if r_polished > best_score:
                    best_score = r_polished
                    best_x = x_polished.copy()
            except Exception:
                pass
        
        # Phase 3: Local search
        if best_x is not None and time.time() - start_time < deadline - 10:
            try:
                x_local = _local_search(best_x, time_budget=10.0)
                r_local = _compute_radius(x_local.reshape(-1, 3))
                if r_local > best_score:
                    best_score = r_local
                    best_x = x_local.copy()
            except Exception:
                pass
    
    # Final refinement on the best solution
    if best_x is not None and time.time() - start_time < deadline - 5:
        try:
            x_final = _smooth_polish(best_x, time_budget=10.0)
            r_final = _compute_radius(x_final.reshape(-1, 3))
            if r_final > best_score:
                best_score = r_final
                best_x = x_final.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    if best_x is not None:
        centers = best_x.reshape(-1, 3)
        # Ensure all points are strictly inside the cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Final check for validity
        if not np.all(np.isfinite(centers)):
            centers = _seed_grid()
        
        return centers
    else:
        return _seed_grid()

# EVOLVE-BLOCK-END
