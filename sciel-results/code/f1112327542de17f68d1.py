"""
Optimization of 100 equal spheres in a unit cube.
Uses a barrier method with L-BFGS-B optimization to maximize the minimum radius.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    # Exclude self-distance (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_radius(x, beta=50.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns a smooth surrogate for the objective with analytical gradient.
    Vectorized implementation for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2) + 1e-15
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Combine all constraints into a single array
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Soft min via log-sum-exp
    min_d = np.min(all_dists)
    if min_d < 1e-10:
        return 0.0, np.zeros_like(x)
    
    shifted = all_dists - min_d
    weights = np.exp(-beta * shifted)
    S = np.sum(weights)
    if S < 1e-300:
        return 0.0, np.zeros_like(x)
    w_norm = weights / S
    
    soft_min = min_d - np.log(S) / beta
    
    # Compute gradient using vectorized operations
    grad = np.zeros_like(x)
    
    # Pairwise distance gradients (vectorized)
    pair_w = w_norm[:len(i_idx)]
    pair_d = pair_dists
    valid = pair_d > 1e-10
    
    if np.any(valid):
        # Vector = centers[i] - centers[j] for valid pairs
        vecs = centers[i_idx[valid]] - centers[j_idx[valid]]
        # Factor = -w / (4 * pair_d)
        factors = -pair_w[valid] / (4.0 * pair_d[valid])
        
        # Accumulate gradients
        np.add.at(grad[::3], i_idx[valid], factors[:, None] * vecs)
        np.add.at(grad[::3], j_idx[valid], -factors[:, None] * vecs)
    
    # Wall distance gradients (vectorized)
    wall_w = w_norm[len(i_idx):]
    abs_centers = np.abs(centers)
    max_vals = max_abs
    
    # For each center, find which coordinate is the max
    # Use argmax to find the dominant coordinate
    max_coord_idx = np.argmax(abs_centers, axis=1)
    
    for i in range(n):
        w = wall_w[i]
        if w < 1e-15:
            continue
        k = max_coord_idx[i]
        if max_vals[i] < 1e-10:
            continue
        grad[3*i+k] += w * (-np.sign(centers[i, k]))
    
    return -soft_min, grad

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Uses vectorized operations for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Vectorized computation for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        # Gradient: for each violating pair, add force
        # dE/dc_i = -2 * v_ij * (c_i - c_j) / d_ij
        # dE/c_j = 2 * v_ij * (c_i - c_j) / d_ij
        for k in range(len(v_ij)):
            if v_ij[k] <= 0:
                continue
            i = i_idx[k]
            j = j_idx[k]
            v = v_ij[k]
            d = d_ij[k]
            if d < 1e-10:
                continue
            factor = -2.0 * v / d
            vec = centers[i] - centers[j]
            force = factor * vec
            grad[i] += force
            grad[j] -= force

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        for i in range(n):
            if v_wall[i] <= 0:
                continue
            v = v_wall[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()

def _polish_centers(centers, max_iter=500, time_budget=10.0):
    """
    High-precision polish pass to maximize the minimum radius.
    Uses a smooth objective with tight tolerances and analytical gradient.
    Multiple stages with increasing beta for convergence.
    """
    start_time = time.time()
    current = centers.copy()
    
    # Stage 1: Moderate beta for broad convergence
    def objective_and_grad_beta1(x):
        val, g = _soft_min_radius(x, beta=50.0)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad_beta1,
            current.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//2, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
        )
        current = res.x.reshape(-1, 3)
        current = np.clip(current, -0.499999, 0.499999)
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.6:
        return current
    
    # Stage 2: Higher beta for fine convergence
    def objective_and_grad_beta2(x):
        val, g = _soft_min_radius(x, beta=100.0)
        return val, g
    
    try:
        res = minimize(
            objective_and_grad_beta2,
            current.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter//2, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
        )
        current = res.x.reshape(-1, 3)
        current = np.clip(current, -0.499999, 0.499999)
    except Exception:
        pass
    
    if time.time() - start_time > time_budget * 0.85:
        return current
    
    # Stage 3: Very high beta for final precision
    def objective_and_grad_beta3(x):
        val, g = _soft_min_radius(x, beta=200.0)
        return val, g
    
    try:
        res = minimize(
            objective_and_grad_beta3,
            current.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1000}
        )
        current = res.x.reshape(-1, 3)
        current = np.clip(current, -0.499999, 0.499999)
    except Exception:
        pass
    
    return current

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_budget=5.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > time_budget:
             pass
            
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy 1: Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # Multiple grid configurations
    grid_configs = [
        (5, 4, 5),  # 100
        (4, 5, 5),  # 100
        (5, 5, 4),  # 100
        (10, 10, 1),  # 100 (flat)
        (1, 10, 10),  # 100 (flat)
        (10, 1, 10),  # 100 (flat)
    ]
    
    rng = np.random.RandomState(42)
    
    # Create a set of initial guesses
    inits = []
    for nx, ny, nz in grid_configs:
        if nx * ny * nz == N:
            inits.append(make_grid(nx, ny, nz))
    
    # Add random inits
    for i in range(8):
        rc = rng.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(rc)
    
    # Add structured random: start from grid, add noise
    base_grid = make_grid(5, 4, 5)
    for i in range(4):
        noise_scale = 0.02 * (i + 1)
        noisy = base_grid + rng.normal(0, noise_scale, size=(N, 3))
        noisy = np.clip(noisy, -0.45, 0.45)
        inits.append(noisy)
    
    # Add lattice-based initialization
    def make_lattice_init(target_r):
        """Create a dense lattice and greedily select non-overlapping points using farthest-point sampling."""
        # Use a fine grid to place points
        spacing = 2.0 * target_r * 0.95  # slightly less than diameter
        if spacing < 0.01:
            spacing = 0.01
        if spacing > 0.5:
            spacing = 0.5
        
        # Generate lattice points
        n_pts_per_dim = int(1.0 / spacing) + 1
        if n_pts_per_dim < 3:
            n_pts_per_dim = 3
        if n_pts_per_dim > 15:
            n_pts_per_dim = 15
            
        xs = np.linspace(-0.5 + spacing/2, 0.5 - spacing/2, n_pts_per_dim)
        lattice = []
        for z in xs:
            for y in xs:
                for x in xs:
                    lattice.append([x, y, z])
        lattice = np.array(lattice)
        
        # Remove points too close to walls
        max_abs = np.max(np.abs(lattice), axis=1)
        lattice = lattice[max_abs <= 0.5 - target_r]
        
        if len(lattice) < N:
            return None
            
        # Greedy farthest-point sampling
        selected = []
        remaining = set(range(len(lattice)))
        
        # Start from center
        dists_to_center = np.sum(lattice**2, axis=1)
        idx = np.argmin(dists_to_center)
        selected.append(idx)
        remaining.remove(idx)
        
        # Track minimum distance from each remaining point to selected set
        min_dist_to_selected = np.full(len(lattice), np.inf)
        min_dist_to_selected[idx] = 0
        
        while len(selected) < N and len(remaining) > 0:
            # Pick point that maximizes minimum distance to selected
            valid_remaining = list(remaining)
            if not valid_remaining:
                break
            
            # Find the point with maximum min_dist_to_selected
            best_idx = max(valid_remaining, key=lambda i: min_dist_to_selected[i])
            
            # Check if this point is valid (sufficient distance from all selected)
            sel_pts = lattice[selected]
            d_to_sel = np.sqrt(np.sum((lattice[best_idx] - sel_pts)**2, axis=1))
            if np.min(d_to_sel) < 2.0 * target_r * 0.98:
                # This point would overlap, try next best
                remaining.remove(best_idx)
                continue
            
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Update min_dist_to_selected for remaining points
            d_to_new = np.sqrt(np.sum((lattice[remaining] - lattice[best_idx])**2, axis=1))
            min_dist_to_selected[list(remaining)] = np.minimum(min_dist_to_selected[list(remaining)], d_to_new)
            
            # Check if we can still fit N points
            if len(remaining) < N - len(selected):
                break
        
        if len(selected) < N:
            return None
            
        return lattice[selected]

    # Try lattice-based init with various target radii
    for tr in [0.10, 0.105, 0.11]:
        if time.time() > deadline - 30.0:
            break
        lat_init = make_lattice_init(tr)
        if lat_init is not None:
            inits.append(lat_init)
    
    for init in inits:
        if time.time() > deadline - 25.0:
            break
            
        # 1. Calculate initial radius
        r_est = _compute_radius(init)
        if r_est <= 0:
            continue
        
        # Skip if this init is unlikely to beat current best
        if r_est < best_radius * 0.95 and best_radius > 0.1:
            continue
        
        # 2. Iteratively increase target radius
        target_R = max(r_est * 1.01, 0.08)
        
        current_centers = init.copy()
        max_iter_per_step = 60
        
        stuck_count = 0
        max_stuck = 3
        while time.time() < deadline - 20.0:
            # Try to optimize for target_R
            time_budget_for_step = min(4.0, max(1.0, (deadline - 15.0 - time.time())))
            optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=max_iter_per_step, time_budget=time_budget_for_step)
            
            # Check if we achieved a valid packing for this radius
            if energy < 1e-6:
                r_achieved = _compute_radius(optimized_centers)
                
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = optimized_centers.copy()
                
                # Try to increase the radius
                increment = 0.003 if target_R < 0.108 else 0.001
                target_R *= (1.0 + increment)
                current_centers = optimized_centers
                stuck_count = 0
            else:
                # Failed to fit this radius
                r_curr = _compute_radius(current_centers)
                if r_curr > best_radius:
                    best_radius = r_curr
                    best_centers = current_centers.copy()
                
                stuck_count += 1
                
                # Try to heal with slightly smaller radius
                heal_attempts = 0
                heal_R = target_R * 0.98
                healed = False
                while heal_attempts < 2 and time.time() < deadline - 20.0:
                    h_centers, h_energy = _optimize_for_radius(current_centers, heal_R, max_iter=30)
                    if h_energy < 1e-5:
                        current_centers = h_centers
                        target_R = heal_R * 1.002
                        healed = True
                        break
                    else:
                        heal_R *= 0.98
                        heal_attempts += 1
                        if h_energy > 0.3:
                            break
                
                if not healed or stuck_count > max_stuck:
                    break
        
        # Update best with current optimized centers
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final high-precision polish pass
    if best_centers is not None and time.time() < deadline - 10.0:
        polish_time = min(20.0, deadline - 10.0 - time.time())
        polished = _polish_centers(best_centers, max_iter=600, time_budget=polish_time)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
        
        # Second polish with even tighter settings
        if time.time() < deadline - 5.0:
            polish_time2 = min(10.0, deadline - 5.0 - time.time())
            polished2 = _polish_centers(best_centers, max_iter=300, time_budget=polish_time2)
            r_polished2 = _compute_radius(polished2)
            if r_polished2 > best_radius:
                best_radius = r_polished2
                best_centers = polished2
        
        # Third polish: very fine-tuned
        if time.time() < deadline - 2.0:
            polish_time3 = min(5.0, deadline - 2.0 - time.time())
            polished3 = _polish_centers(best_centers, max_iter=100, time_budget=polish_time3)
            r_polished3 = _compute_radius(polished3)
            if r_polished3 > best_radius:
                best_radius = r_polished3
                best_centers = polished3
    
    # Final safety check
    if best_centers is None:
        best_centers = make_grid(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        # Ensure all values are finite
        if not np.all(np.isfinite(best_centers)):
            best_centers = make_grid(5, 4, 5)
        
    return best_centers
# EVOLVE-BLOCK-END
