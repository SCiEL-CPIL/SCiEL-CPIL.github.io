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
    
    # Compute gradient (vectorized)
    grad = np.zeros_like(x)
    
    # Pairwise distance gradients
    valid = pair_dists > 1e-10
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        w_v = w_norm[valid]
        d_v = pair_dists[valid]
        vecs = centers[i_v] - centers[j_v]  # (m, 3)
        factors = w_v / (4.0 * d_v)         # (m,)
        forces = factors[:, np.newaxis] * vecs  # (m, 3)
        np.add.at(grad, 3 * i_v, -forces)
        np.add.at(grad, 3 * j_v, forces)
    
    # Wall distance gradients (vectorized)
    wall_start = len(i_idx)
    wall_w = w_norm[wall_start:]
    valid_wall = max_abs > 1e-10
    if np.any(valid_wall):
        for i in np.where(valid_wall)[0]:
            w = wall_w[i]
            if w < 1e-15:
                continue
            max_val = max_abs[i]
            for k in range(3):
                if np.isclose(np.abs(centers[i, k]), max_val, atol=1e-12):
                    grad[3*i+k] += w * (-np.sign(centers[i, k]))
                    break
    
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
        
        # Gradient: vectorized
        valid = (v_ij > 0) & (d_ij > 1e-10)
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            v_v = v_ij[valid]
            d_v = d_ij[valid]
            vecs = centers[i_v] - centers[j_v]  # (m, 3)
            factors = -2.0 * v_v / d_v           # (m,)
            forces = factors[:, np.newaxis] * vecs
            np.add.at(grad, i_v, forces)
            np.add.at(grad, j_v, -forces)

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        valid_w = v_wall > 0
        if np.any(valid_w):
            idxs = np.where(valid_w)[0]
            for i in idxs:
                v = v_wall[i]
                max_val = max_abs[i]
                for k in range(3):
                    if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                        grad[i, k] += 2.0 * v * np.sign(centers[i, k])
                        break
            
    return energy, grad.flatten()

def _polish_centers(centers, max_iter=500, time_budget=10.0, beta=80.0):
    """
    High-precision polish pass to maximize the minimum radius.
    Uses a smooth objective with tight tolerances and analytical gradient.
    """
    start_time = time.time()
    
    def objective_and_grad(x):
        val, g = _soft_min_radius(x, beta=beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 5000}
        )
        
        polished = res.x.reshape(-1, 3)
        # Ensure centers are within bounds
        polished = np.clip(polished, -0.499999, 0.499999)
        return polished
    except Exception:
        return centers

def _nelder_mead_polish(centers, time_budget=5.0):
    """
    Final Nelder-Mead polish: derivative-free local search on the exact
    (nonsmooth) radius objective. Handles the kink at the argmin of the
    distance set better than a smooth surrogate.
    """
    start_time = time.time()
    
    def objective(x):
        c = x.reshape(-1, 3)
        # Quick wall check
        max_abs = np.max(np.abs(c), axis=1)
        min_wall = 0.5 - np.max(max_abs)
        if min_wall < -1e-12:
            return 1.0  # penalty for going outside
        # Pairwise min
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        r = min(min_pair, min_wall)
        return -r  # minimize negative radius

    try:
        res = minimize(
            objective,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.499999, 0.499999)
        return polished
    except Exception:
        return centers

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

def _generate_lattice_init(target_r, rng):
    """
    Generate a structured lattice-based initialization.
    This creates a dense lattice and greedily selects non-overlapping points.
    """
    spacing = 2.0 * target_r * 0.95
    if spacing < 0.01:
        spacing = 0.01
    if spacing > 0.5:
        spacing = 0.5
        
    n_pts_per_dim = int(1.0 / spacing) + 1
    if n_pts_per_dim < 3:
        n_pts_per_dim = 3
    if n_pts_per_dim > 20:
        n_pts_per_dim = 20
        
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
        
    # Greedy selection: pick points that are farthest from already-selected
    selected = []
    remaining = np.arange(len(lattice))
    
    while len(selected) < N and len(remaining) > 0:
        if len(selected) == 0:
            # Start from center
            dists_to_center = np.sum(lattice[remaining]**2, axis=1)
            idx = remaining[np.argmin(dists_to_center)]
        else:
            # Pick point that maximizes minimum distance to selected
            min_dists = np.full(len(remaining), np.inf)
            for sel_idx in selected[-5:]:
                d = np.sqrt(np.sum((lattice[remaining] - lattice[sel_idx])**2, axis=1))
                min_dists = np.minimum(min_dists, d)
            idx = remaining[np.argmax(min_dists)]
        
        selected.append(idx)
        remaining = remaining[remaining != idx]
        
        if len(remaining) < N - len(selected):
            break
    
    if len(selected) < N:
        return None
        
    return lattice[selected]

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
    for tr in [0.10, 0.105, 0.11]:
        if time.time() > deadline - 30.0:
            break
        lat_init = _generate_lattice_init(tr, rng)
        if lat_init is not None:
            inits.append(lat_init)
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # 1. Calculate initial radius
        r_est = _compute_radius(init)
        if r_est <= 0:
            continue
        
        # 2. Iteratively increase target radius
        target_R = max(r_est * 1.01, 0.08)
        
        current_centers = init.copy()
        max_iter_per_step = 80
        
        stuck_count = 0
        while time.time() < deadline - 15.0:
            # Try to optimize for target_R
            time_budget_for_step = min(6.0, max(1.5, (deadline - 10.0 - time.time())))
            optimized_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=max_iter_per_step, time_budget=time_budget_for_step)
            
            # Check if we achieved a valid packing for this radius
            if energy < 1e-6:
                r_achieved = _compute_radius(optimized_centers)
                
                if r_achieved > best_radius:
                    best_radius = r_achieved
                    best_centers = optimized_centers.copy()
                
                # Try to increase the radius (adaptive: smaller steps as we get closer)
                if target_R < 0.10:
                    increment = 0.008
                elif target_R < 0.108:
                    increment = 0.004
                elif target_R < 0.112:
                    increment = 0.002
                else:
                    increment = 0.001
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
                heal_R = target_R * 0.96
                healed = False
                while heal_attempts < 5 and time.time() < deadline - 15.0:
                    h_centers, h_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50)
                    if h_energy < 1e-5:
                        current_centers = h_centers
                        target_R = heal_R * 1.002
                        healed = True
                        break
                    else:
                        heal_R *= 0.96
                        heal_attempts += 1
                        if h_energy > 0.5:
                            break
                
                if not healed or stuck_count > 5:
                    break
        
        # Update best with current optimized centers
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final high-precision polish pass
    if best_centers is not None and time.time() < deadline - 5.0:
        # Phase 1: L-BFGS-B with smooth surrogate (high beta)
        polish_time = min(12.0, deadline - 5.0 - time.time())
        polished = _polish_centers(best_centers, max_iter=500, time_budget=polish_time, beta=120.0)
        r_polished = _compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
        
        # Phase 2: Nelder-Mead on the exact nonsmooth objective
        if time.time() < deadline - 3.0:
            nm_budget = min(8.0, deadline - 3.0 - time.time())
            nm_polished = _nelder_mead_polish(best_centers, time_budget=nm_budget)
            r_nm = _compute_radius(nm_polished)
            if r_nm > best_radius:
                best_radius = r_nm
                best_centers = nm_polished
        
        # Phase 3: One more L-BFGS-B polish from the NM result
        if time.time() < deadline - 2.0:
            final_time = min(5.0, deadline - 2.0 - time.time())
            final_polished = _polish_centers(best_centers, max_iter=300, time_budget=final_time, beta=200.0)
            r_final2 = _compute_radius(final_polished)
            if r_final2 > best_radius:
                best_radius = r_final2
                best_centers = final_polished
    
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
