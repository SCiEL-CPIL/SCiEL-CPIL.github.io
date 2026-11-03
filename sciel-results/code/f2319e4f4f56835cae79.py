import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

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
            # d/dx_i (2R - d)^2 = 2(2R-d) * (-1/d) * (x_i - x_j)
            # = -2(2R-d)/d * (x_i - x_j)
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
                # Find the dimension that is closest to the wall
                for k in range(3):
                    if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                        # d/dx_k (R - (0.5 - |x_k|))^2 
                        # = 2(R - (0.5 - |x_k|)) * sign(x_k)
                        grad[i, k] += 2.0 * v * np.sign(centers[i, k])
                        break
            
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=100, time_budget=5.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    Returns (centers, energy).
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
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        # Check time
        if time.time() - start_time > time_budget:
             # We might have run over, but we return what we have
             pass
            
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf

def _nelder_mead_polish(centers, time_budget=5.0):
    """
    Final Nelder-Mead polish: derivative-free local search on the exact
    (nonsmooth) radius objective.
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
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.499999, 0.499999)
        return polished
    except Exception:
        return centers

def _generate_grid_init(nx, ny, nz):
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append([x, y, z])
    return np.array(grid)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    # Initial configurations
    inits = []
    
    # 1. Grid 5x4x5
    inits.append(_generate_grid_init(5, 4, 5))
    
    # 2. Grid 4x5x5
    inits.append(_generate_grid_init(4, 5, 5))
    
    # 3. Grid 5x5x4
    inits.append(_generate_grid_init(5, 5, 4))
    
    # 4. Random
    for i in range(5):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
        
    # 5. Jittered Grid
    base_grid = _generate_grid_init(5, 4, 5)
    for i in range(3):
        noise_scale = 0.01 * (i + 1)
        noisy = base_grid + rng.normal(0, noise_scale, size=(N, 3))
        noisy = np.clip(noisy, -0.45, 0.45)
        inits.append(noisy)

    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        current_centers = init.copy()
        r_est = _compute_radius(current_centers)
        if r_est <= 0:
            continue
            
        # Start with a target slightly higher than current
        target_R = r_est * 1.01
        if target_R < 0.05:
            target_R = 0.05
            
        max_iter_per_step = 50
        stuck_count = 0
        consecutive_failures = 0
        
        while time.time() < deadline - 15.0:
            # Time budget for this step
            time_budget_for_step = min(5.0, max(1.0, (deadline - 15.0 - time.time())))
            
            optimized_centers, energy = _optimize_for_radius(
                current_centers, 
                target_R, 
                max_iter=max_iter_per_step, 
                time_budget=time_budget_for_step
            )
            
            r_achieved = _compute_radius(optimized_centers)
            
            if r_achieved > best_radius:
                best_radius = r_achieved
                best_centers = optimized_centers.copy()
                
            if energy < 1e-6:
                # Success: we can increase the target radius
                # Adaptive step: smaller steps as we get larger
                if target_R < 0.08:
                    increment = 0.01
                elif target_R < 0.10:
                    increment = 0.005
                elif target_R < 0.11:
                    increment = 0.002
                else:
                    increment = 0.001
                    
                target_R *= (1.0 + increment)
                current_centers = optimized_centers
                stuck_count = 0
                consecutive_failures = 0
            else:
                # Failure: cannot achieve this radius
                consecutive_failures += 1
                
                if consecutive_failures > 3:
                    # Try to back off and heal
                    heal_R = target_R * 0.98
                    h_centers, h_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50)
                    if h_energy < 1e-5:
                        current_centers = h_centers
                        target_R = heal_R * 1.001
                        consecutive_failures = 0
                    else:
                        # Stuck, try a random restart from this neighborhood
                        if time.time() < deadline - 20.0:
                            perturbed = current_centers + rng.normal(0, 0.005, size=(N, 3))
                            perturbed = np.clip(perturbed, -0.45, 0.45)
                            current_centers = perturbed
                            target_R = _compute_radius(current_centers) * 1.01
                            consecutive_failures = 0
                        else:
                            break
                else:
                    # Small backoff
                    target_R *= 0.995
                    current_centers = optimized_centers # Keep the improved centers even if target not met
                    
                stuck_count += 1
                if stuck_count > 10:
                    break

    # Final Polish
    if best_centers is not None and time.time() < deadline - 5.0:
        # Nelder-Mead on the exact objective
        nm_budget = min(10.0, deadline - 5.0 - time.time())
        if nm_budget > 1.0:
            nm_polished = _nelder_mead_polish(best_centers, time_budget=nm_budget)
            r_nm = _compute_radius(nm_polished)
            if r_nm > best_radius:
                best_radius = r_nm
                best_centers = nm_polished
                
        # One more L-BFGS polish if time remains
        if time.time() < deadline - 2.0:
            final_time = min(5.0, deadline - 2.0 - time.time())
            if final_time > 0.5:
                # Use a very tight tolerance
                final_polished, _ = _optimize_for_radius(best_centers, best_radius * 1.0001, max_iter=200, time_budget=final_time)
                r_final = _compute_radius(final_polished)
                if r_final > best_radius:
                    best_radius = r_final
                    best_centers = final_polished

    # Final safety check
    if best_centers is None:
        best_centers = _generate_grid_init(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        if not np.all(np.isfinite(best_centers)):
            best_centers = _generate_grid_init(5, 4, 5)
            
    return best_centers

# EVOLVE-BLOCK-END
