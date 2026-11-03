import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def energy_and_gradient(x, target_R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Violations:
    1. Pairwise distance < 2R
    2. Wall distance < R
    
    Returns: (energy, gradient_vector)
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Mask for pairs closer than 2R
    mask = dists < 2.0 * target_R
    if np.any(mask):
        # We only care about i < j to avoid double counting in energy
        # But gradient needs to be accumulated correctly
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        # Get distances and violations for upper triangle
        d_ij = dists[i_idx, j_idx]
        v_ij = 2.0 * target_R - d_ij
        
        # Only count violations where v_ij > 0
        active_mask = v_ij > 0
        if np.any(active_mask):
            v_active = v_ij[active_mask]
            d_active = d_ij[active_mask]
            i_active = i_idx[active_mask]
            j_active = j_idx[active_mask]
            
            # Energy: sum(v^2)
            energy += np.sum(v_active**2)
            
            # Gradient:
            # dE/dx_i = sum_j 2(2R - d_ij) * (-1) * (x_i - x_j) / d_ij
            # Factor for pair (i,j): -2 * v_ij / d_ij
            # Force on i: factor * (x_i - x_j)
            # Force on j: -factor * (x_i - x_j)
            
            factors = -2.0 * v_active / d_active
            vecs = diff[i_active, j_active] # x_i - x_j
            
            # Accumulate using np.add.at for efficiency
            # Force on i is factors * vecs
            # Force on j is -factors * vecs
            
            forces_i = factors[:, np.newaxis] * vecs
            forces_j = -forces_i
            
            np.add.at(grad, i_active, forces_i)
            np.add.at(grad, j_active, forces_j)
    
    # 2. Wall terms
    # d_wall_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < target_R
    if np.any(mask_wall):
        v_wall = target_R - wall_dists
        v_wall[~mask_wall] = 0.0
        
        # Energy: sum(v^2)
        energy += np.sum(v_wall**2)
        
        # Gradient:
        # E = (R - d_wall)^2
        # dE/dx = 2(R - d_wall) * (-1) * d(d_wall)/dx
        # d(d_wall)/dx_k = -sign(x_k) if x_k is the max component, else 0
        # So dE/dx_k = 2(R - d_wall) * sign(x_k) (for the max component)
        
        # We need to identify the max component for each violating sphere
        # For efficiency, we can compute the subgradient for all and mask
        
        # signs shape (N, 3)
        signs = np.sign(centers)
        # Handle zero: if center is 0, sign is 0, but derivative of |x| is undefined. 
        # Usually we pick a direction. Let's default to 1.0 if 0.
        signs[signs == 0] = 1.0
        
        # We need to zero out signs for non-max components
        # max_abs shape (N,)
        # is_max shape (N, 3)
        is_max = np.isclose(abs_centers, max_abs[:, np.newaxis], rtol=1e-12, atol=1e-12)
        
        # Gradient contribution: 2 * v_wall * signs * is_max
        # v_wall shape (N,), signs (N,3), is_max (N,3)
        grad_wall = 2.0 * v_wall[:, np.newaxis] * signs * is_max
        
        # Only add for violating walls
        grad += grad_wall * mask_wall[:, np.newaxis]
        
    return energy, grad.flatten()

def _optimize_for_radius(init_centers, target_R, max_iter=200, time_limit=10.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    Returns the optimized centers.
    """
    def objective_and_grad(x):
        val, g = energy_and_gradient(x, target_R)
        return val, g

    # L-BFGS-B requires bounds. Centers must be in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        # Check time
        if time.time() > start_time + time_limit:
             # If it took too long, we might still have a good result, 
             # but we should be careful about returning incomplete iterations if they are bad.
             # However, minimize usually returns the best x found.
             pass
             
        return res.x.reshape(-1, 3)
    except Exception:
        return init_centers

def _polish_hard_min(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """
    High-precision polish pass using the hard-min objective (maximize min dist).
    This is used to refine a feasible solution.
    """
    def hard_min_obj_and_grad(flat_pos):
        pos = flat_pos.reshape(N, 3)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.abs(pos)
        min_wall_val = np.min(d_walls)
        min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
        wi, wk = min_wall_idx
        
        r_pair = min_pair_val / 2.0
        r_wall = min_wall_val
        
        grad = np.zeros_like(pos)
        
        if r_pair <= r_wall:
            d = min_pair_val
            if d > 1e-12:
                vec = (pos[i] - pos[j]) / d
                grad[i] += 0.5 * vec
                grad[j] += -0.5 * vec
            return -r_pair, -grad.flatten()
        else:
            s = np.sign(pos[wi, wk])
            if s == 0:
                s = 1.0
            grad[wi, wk] = -s
            return -r_wall, -grad.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Grid
    # 100 points. 5x4x5 grid.
    # x: 5, y: 4, z: 5
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
    
    best_pos = make_grid(5, 4, 5)
    best_r = compute_min_dist(best_pos)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Incremental Radius Increase
    # Start from current best radius and try to push higher.
    # If we fail, we try a slightly lower target or different initialization.
    
    current_target = best_r
    
    # We will iterate, increasing the target radius.
    # If we find a feasible config, we update best_r and best_pos.
    # Then we increase target again.
    
    # To avoid getting stuck, we also do random restarts with the current best_r as target
    # to find better local minima for that radius, which might allow a higher radius later.
    
    attempt_count = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        attempt_count += 1
        
        # Determine target radius
        # If we haven't improved in a while, try a more aggressive jump or random restart
        if time.time() - last_improvement_time > 10.0:
             # Stagnation: try a random restart with a target slightly higher than best
             # or just a random config with target = best_r * 1.01
             target_R = best_r * 1.01
             init_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
        else:
             # Normal progression: try to increase radius by a small step
             # Step size depends on how much time we have
             # Smaller steps are safer, larger steps are faster if they work
             step = 0.001 
             target_R = best_r + step
             
             # Use best_pos as init, with small perturbation
             init_pos = best_pos + rng.normal(0, 0.01, size=(N, 3))
             init_pos = np.clip(init_pos, -0.5, 0.5)
        
        # Run optimization for this target
        # Limit time per optimization to ensure we finish
        time_left = deadline - time.time()
        opt_time = min(10.0, time_left * 0.8)
        
        if opt_time < 1.0:
            break
            
        new_pos = _optimize_for_radius(init_pos, target_R, max_iter=100, time_limit=opt_time)
        
        # Check if the result is feasible for target_R
        # We compute the actual radius achieved
        achieved_r = compute_min_dist(new_pos)
        
        # If we achieved the target (or better), it's a success
        # Note: due to numerical precision, achieved_r might be slightly less than target_R
        # We accept if achieved_r is close to target_R or better than best_r
        
        if achieved_r > best_r:
            best_r = achieved_r
            best_pos = new_pos
            last_improvement_time = time.time()
        elif achieved_r > target_R * 0.99: # Close to target, maybe useful as next init?
             # If we got close to target but didn't beat best_r, it might be a good starting point
             # for a slightly higher target next time, but for now, we stick to best_pos
             pass

        # If we are close to deadline, do a final polish
        if time.time() > deadline - 15.0:
            break

    # Final Polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish_hard_min(best_pos, max_iter=300)
        final_r = compute_min_dist(polished_pos)
        if final_r > best_r:
            best_r = final_r
            best_pos = polished_pos

    return best_pos

# EVOLVE-BLOCK-END
