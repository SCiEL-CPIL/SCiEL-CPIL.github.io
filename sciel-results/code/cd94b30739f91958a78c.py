"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    
    # Minimum distance to boundary
    min_boundary = HALF - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _soft_min_obj_grad(flat_pos, beta=100.0):
    """
    Soft-min objective: maximize the soft-minimum of all effective distances.
    soft_min = -logsumexp(-beta * all_eff_dists) / beta
    This approximates the minimum as beta -> infinity.
    We minimize -soft_min.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    wall_dists = d_walls.flatten()
    
    # All effective distances (pair distances divided by 2, wall distances as-is)
    # Note: pair dist d corresponds to radius d/2. Wall dist d corresponds to radius d.
    all_eff_dists = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft-min: -logsumexp(-beta * x) / beta
    scaled = -beta * all_eff_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. positions
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient
    # d(soft_min)/d(pos_i) = sum_j w_j * d(d_ij/2)/d(pos_i)
    # d(d_ij)/d(pos_i) = (pos_i - pos_j) / d_ij
    # So d(soft_min)/d(pos_i) = sum_j w_j * (pos_i - pos_j) / (2 * d_ij)
    
    n_pairs = len(iu[0])
    pair_weights = weights[:n_pairs]
    
    # Vectorized accumulation
    # i_idx, j_idx
    i_idx = iu[0]
    j_idx = iu[1]
    d_vals = pair_dists
    
    # Avoid division by zero
    valid = d_vals > 1e-14
    if np.any(valid):
        i_v = i_idx[valid]
        j_v = j_idx[valid]
        d_v = d_vals[valid]
        w_v = pair_weights[valid]
        
        vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
        coeffs = w_v / (2.0 * d_v) # shape (n_valid,)
        
        # grad_pos[i] += coeff * vec
        np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
        # grad_pos[j] -= coeff * vec
        np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    # d(soft_min)/d(pos_ik) = w_k * d(d_wall_ik)/d(pos_ik)
    # d_wall_ik = HALF - |pos_ik|
    # d(d_wall)/d(pos_ik) = -sign(pos_ik)
    # So d(soft_min)/d(pos_ik) = -w_k * sign(pos_ik)
    
    wall_weights = weights[n_pairs:]
    signs = np.sign(pos)
    # Handle zero sign
    signs[signs == 0] = 1.0
    
    # grad_pos[i, k] += -w_k * sign(pos_ik)
    # wall_weights is flattened (N*3)
    grad_wall = -wall_weights.reshape(N, 3) * signs
    grad_pos += grad_wall
    
    # We want to MAXIMIZE the soft-min, so we minimize -soft_min
    # Gradient of -soft_min is -grad(soft_min)
    return -soft_min_val, -grad_pos.flatten()

def _energy_obj_grad(flat_pos, r_target):
    """
    Smooth energy objective: penalize pair overlaps and wall violations.
    E = 0.5 * sum_{i<j} max(0, 2*r_target - d_ij)^2 + 0.5 * sum_{i,k} max(0, r_target - (HALF - |x_ik|))^2
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    # Compute violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
    wall_viol = np.maximum(0.0, r_target - d_walls)
    
    energy = 0.5 * np.sum(pair_viol**2) + 0.5 * np.sum(wall_viol**2)
    
    # Gradient - vectorized
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient: for each violating pair, push apart
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_viol_vals = pair_viol[iu]
    
    mask = pair_viol_vals > 0
    if np.any(mask):
        i_idx = iu[0][mask]
        j_idx = iu[1][mask]
        d_vals = pair_dists[mask]
        v_vals = pair_viol_vals[mask]
        
        # Avoid division by zero
        valid = d_vals > 1e-14
        if np.any(valid):
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            d_v = d_vals[valid]
            v_v = v_vals[valid]
            
            vecs = pos[i_v] - pos[j_v]  # shape (n_valid, 3)
            coeffs = v_v / d_v  # shape (n_valid,)
            
            # Accumulate gradients
            # dE/dpos_i = sum_j v_ij * (pos_i - pos_j) / d_ij
            np.add.at(grad_pos, i_v, coeffs[:, np.newaxis] * vecs)
            np.add.at(grad_pos, j_v, -coeffs[:, np.newaxis] * vecs)
    
    # Wall gradient
    wall_viol_mask = wall_viol > 0
    if np.any(wall_viol_mask):
        i_idx = np.where(wall_viol_mask)[0]
        k_idx = np.where(wall_viol_mask)[1]
        v_vals = wall_viol[wall_viol_mask]
        
        signs = np.sign(pos[i_idx, k_idx])
        signs[signs == 0] = 1.0
        
        # dE/dpos_ik = v_ik * sign(pos_ik)
        grad_pos[i_idx, k_idx] += v_vals * signs
    
    return energy, grad_pos.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Initialize with a grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.01, centers.shape)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    bounds = [(-HALF, HALF)] * (N * 3)
    
    # Main optimization loop
    # Alternate between soft-min (exploration) and energy (exploitation/polish)
    current_pos = centers.copy()
    
    # Initial soft-min optimization to spread out
    try:
        res = minimize(
            lambda x: _soft_min_obj_grad(x, beta=50.0)[0],
            current_pos.flatten(),
            jac=lambda x: _soft_min_obj_grad(x, beta=50.0)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if np.isfinite(res.fun):
            current_pos = res.x.reshape(N, 3)
            r = _compute_radius(current_pos)
            if r > best_radius:
                best_radius = r
                best_centers = current_pos.copy()
    except Exception:
        pass
    
    # Iterative improvement
    beta_soft = 50.0
    polish_factor = 1.01
    
    while time.time() < deadline - 10:
        # 1. Soft-min optimization to push spheres apart
        # Increase beta slightly to make the approximation tighter to the true min
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, beta=beta_soft)[0],
                current_pos.flatten(),
                jac=lambda x: _soft_min_obj_grad(x, beta=beta_soft)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if np.isfinite(res.fun):
                current_pos = res.x.reshape(N, 3)
        except Exception:
            pass
        
        # 2. Compute actual radius
        current_r = _compute_radius(current_pos)
        
        if current_r > best_radius:
            best_radius = current_r
            best_centers = current_pos.copy()
        
        # 3. Energy polish with slightly larger target radius
        # This helps escape flat regions where soft-min gradient is small
        target_r = current_r * polish_factor
        if target_r > 0.12: # Safety cap
            break
            
        try:
            res = minimize(
                lambda x: _energy_obj_grad(x, target_r)[0],
                current_pos.flatten(),
                jac=lambda x: _energy_obj_grad(x, target_r)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if np.isfinite(res.fun):
                new_pos = res.x.reshape(N, 3)
                new_r = _compute_radius(new_pos)
                if new_r > current_r:
                    current_pos = new_pos
                    current_r = new_r
                    if current_r > best_radius:
                        best_radius = current_r
                        best_centers = current_pos.copy()
                else:
                    # If energy minimization didn't improve, stick with soft-min result
                    pass
        except Exception:
            pass
        
        # Increase beta for tighter approximation in next iteration
        beta_soft *= 1.2
        if beta_soft > 500.0:
            beta_soft = 500.0
            
        # If improvement stalls, try a small random perturbation to escape local minima
        # (Only if we have plenty of time left)
        if time.time() < deadline - 30:
            # Check for stagnation by comparing with previous best
            # For simplicity, we just rely on the optimizer's convergence
            pass

    # Final polish with high beta soft-min
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _soft_min_obj_grad(x, beta=200.0)[0],
                best_centers.flatten(),
                jac=lambda x: _soft_min_obj_grad(x, beta=200.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if np.isfinite(res.fun):
                final_pos = res.x.reshape(N, 3)
                final_r = _compute_radius(final_pos)
                if final_r > best_radius:
                    best_radius = final_r
                    best_centers = final_pos
        except Exception:
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
