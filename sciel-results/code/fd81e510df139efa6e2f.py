import numpy as np
import time
import sys
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
TIME_LIMIT = 145.0  # Seconds, leaving margin before 150s kill
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME

def _compute_radius(centres):
    """Compute the packing radius for a set of centres.
    Returns r if valid, 0.0 if invalid.
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0

    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    return min(min_dist / 2.0, min_wall)

def _generate_fibonacci_config():
    """
    Generate a hexagonal lattice configuration that fills the square [-0.5, 0.5]^2.
    """
    # Hex grid: for N=200, use a 20x10 grid with hex offset
    # This gives a good initial packing density
    cols, rows = 20, 10
    
    # Create grid points
    points = []
    for row in range(rows):
        for col in range(cols):
            x = col - (cols - 1) / 2.0
            y = row - (rows - 1) / 2.0
            # Hex offset for odd rows
            if row % 2 != 0:
                x += 0.5
            points.append([x, y])
    
    grid = np.array(points)
    
    # Scale to fit within [-0.5, 0.5]^2 with margin
    max_abs = np.max(np.abs(grid))
    if max_abs > 0.45:
        grid *= 0.45 / max_abs
    
    return grid

def _push_optimize(centres, time_limit):
    """
    Optimize using a vectorized repulsive force / push method.
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    step_size = 0.003
    min_step_size = 1e-7
    damping = 0.85
    no_improve_count = 0
    max_no_improve = 50
    
    velocity = np.zeros_like(best_centres)
    
    while time.time() < deadline and step_size > min_step_size and no_improve_count < max_no_improve:
        # Calculate pairwise distances
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(best_centres)
        
        # Target distance: slightly above current radius to encourage improvement
        target_d = 2.0 * best_r * 1.02
        
        # Vectorized repulsion: push apart pairs closer than target_d
        # Force = (target_d - d) / d * direction, for d < target_d
        mask = dists < target_d
        mask = mask & (dists > 1e-12)
        
        # For each pair (i,j) with i < j and dist < target_d:
        idx_i, idx_j = np.where(np.triu(mask, k=1))
        if len(idx_i) > 0:
            diffs = best_centres[idx_i] - best_centres[idx_j]
            dists_pair = dists[idx_i, idx_j]
            overlaps = np.maximum(target_d - dists_pair, 0)
            # Force magnitude
            force_mag = overlaps / dists_pair
            force_vecs = diffs * force_mag[:, np.newaxis]
            
            # Accumulate forces using np.add.at
            for k in range(len(idx_i)):
                i, j = idx_i[k], idx_j[k]
                forces[i] += force_vecs[k]
                forces[j] -= force_vecs[k]
        
        # Wall repulsion (vectorized)
        wx = 0.5 - np.abs(best_centres[:, 0])
        wy = 0.5 - np.abs(best_centres[:, 1])
        
        # Points close to vertical walls
        close_x = wx < best_r
        if np.any(close_x):
            signs_x = np.sign(best_centres[close_x, 0])
            signs_x[signs_x == 0] = 1.0
            forces[close_x, 0] += signs_x * (best_r - wx[close_x])
        
        # Points close to horizontal walls
        close_y = wy < best_r
        if np.any(close_y):
            signs_y = np.sign(best_centres[close_y, 1])
            signs_y[signs_y == 0] = 1.0
            forces[close_y, 1] += signs_y * (best_r - wy[close_y])
        
        # Update positions
        velocity = velocity * damping + forces * step_size
        new_centres = best_centres + velocity
        new_centres = np.clip(new_centres, -0.4999, 0.4999)
        
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r:
            best_r = new_r
            best_centres = new_centres
            no_improve_count = 0
            step_size *= 0.998
        else:
            no_improve_count += 1
            step_size *= 0.92
            velocity *= 0.5
        
        if time.time() > deadline - 2:
            break
            
    return best_centres, best_r

def _local_optimize_lbfgs(centres, time_limit):
    """
    Optimize using L-BFGS-B with soft-min energy (log-sum-exp surrogate).
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    
    # Try multiple beta values, starting moderate and increasing
    betas = [30.0, 80.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() >= deadline - 8:
            break
            
        def obj_and_grad(x):
            if time.time() >= deadline - 3:
                return 1e10, np.zeros_like(x)
            c = x.reshape(N, 2)
            c = np.clip(c, -0.4999, 0.4999)
            
            # Wall distances
            wall_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
            
            # Pairwise distances (only upper triangle to save memory/time)
            idx_i, idx_j = np.triu_indices(N, k=1)
            dx = c[idx_i, 0] - c[idx_j, 0]
            dy = c[idx_i, 1] - c[idx_j, 1]
            pair_dists = np.sqrt(dx**2 + dy**2) / 2.0
            
            all_constraints = np.concatenate([wall_dists, pair_dists])
            
            # Soft-min objective via log-sum-exp
            neg_beta_c = -beta * all_constraints
            max_val = np.max(neg_beta_c)
            logsumexp_val = max_val + np.log(np.sum(np.exp(neg_beta_c - max_val)))
            objective = logsumexp_val / beta
            
            # Gradient
            weights = np.exp(neg_beta_c - max_val)
            weights /= np.sum(weights)
            
            grad = np.zeros_like(c)
            
            # Wall gradient
            wall_weights = weights[:N]
            for i in range(N):
                wx = 0.5 - abs(c[i, 0])
                wy = 0.5 - abs(c[i, 1])
                if wx <= wy:
                    s = np.sign(c[i, 0])
                    if s == 0: s = 1.0
                    grad[i, 0] -= wall_weights[i] * s
                else:
                    s = np.sign(c[i, 1])
                    if s == 0: s = 1.0
                    grad[i, 1] -= wall_weights[i] * s
            
            # Pairwise gradient
            pair_weights = weights[N:]
            for k in range(len(idx_i)):
                w = pair_weights[k]
                if w < 1e-14:
                    continue
                i, j = idx_i[k], idx_j[k]
                d = pair_dists[k] * 2.0  # actual distance
                if d < 1e-12:
                    continue
                gf = w / d
                grad[idx_i[k], 0] -= gf * dx[k]
                grad[idx_i[k], 1] -= gf * dy[k]
                grad[idx_j[k], 0] += gf * dx[k]
                grad[idx_j[k], 1] += gf * dy[k]
            
            return objective, grad.ravel()
        
        try:
            result = minimize(
                obj_and_grad,
                best_centres.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 150,
                    'maxfun': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            
            new_centres = result.x.reshape(N, 2)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres
        except Exception:
            pass
    
    return best_centres, best_r

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration (Hex Grid)
    centres = _generate_fibonacci_config()
    current_r = _compute_radius(centres)
    
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # 2. Optimization Strategy
    # Phase 1: Push method for quick improvement (use ~40% of budget)
    push_time = min(time_budget * 0.4, 50.0)
    if push_time > 5:
        centres, current_r = _push_optimize(centres, push_time)
    
    # Phase 2: L-BFGS refinement with increasing beta (use remaining time)
    remaining = time_budget - (time.time() - start_time) - 5
    if remaining > 10:
        centres, current_r = _local_optimize_lbfgs(centres, remaining)
    
    # 3. Final Validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return centres
