import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    # Wall clearance: min over i,k of (0.5 - |c_ik|)
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    # Inter-sphere clearance: min over i<j of |c_i - c_j| / 2
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def _smooth_min(dists, beta=50.0):
    """Soft-min approximation using log-sum-exp.
    For positive values, -logsumexp(-beta*x)/beta approximates min(x).
    """
    # dists is a 1D array of positive values
    return -np.log(np.sum(np.exp(-beta * dists))) / beta

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Objective function for L-BFGS-B:
    # We want to maximize r = min(min_pair_dist/2, min_wall_dist)
    # Equivalent to minimizing -r, but -r is non-smooth.
    # Instead, we minimize a penalty function that is zero when all constraints
    # are satisfied with radius R, and positive otherwise.
    # Actually, a better approach for L-BFGS-B is to minimize:
    #   sum over pairs of max(0, 2R - d_ij)^2 + sum over i of max(0, R - wall_i)^2
    # But this is non-smooth at the kink.
    # 
    # Alternative: Use a smooth surrogate.
    # Minimize: -log(sum(exp(-beta * (d_ij/2 - R))) * ... ) is complex.
    #
    # Let's use the standard penalty method but with a smooth approximation.
    # Or, we can directly optimize the "soft-min" objective.
    #
    # Let's define the objective as:
    #   f(x) = -smooth_min(dists/2, wall_dists, beta)
    # This is differentiable. We minimize f(x) to maximize the soft-min radius.
    #
    # However, L-BFGS-B with bounds might struggle if the objective is flat.
    # A more robust approach: iterative radius increase.
    # Start with R0, find a feasible configuration for R0, then increase R.
    
    def energy_penalty(x, R):
        """Penalty function: sum of squared violations for radius R."""
        c = x.reshape(N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        
        # Pairwise violations: max(0, 2R - d_ij)
        pair_viol = np.maximum(0.0, 2.0 * R - dists)
        
        # Wall violations: max(0, R - (0.5 - |c_ik|)) for each coordinate
        # wall_dist_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
        # violation = max(0, R - wall_dist_i)
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_dist = 0.5 - max_coord
        wall_viol = np.maximum(0.0, R - wall_dist)
        
        return np.sum(pair_viol ** 2) + np.sum(wall_viol ** 2)

    def smooth_objective(x, beta=100.0):
        """
        Smooth objective: -soft_min(min_pair_dist/2, min_wall_dist)
        Using log-sum-exp trick.
        """
        c = x.reshape(N, 3)
        # Pairwise distances / 2
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        
        # Wall distances
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_vals = 0.5 - max_coord
        
        # Combine all constraints
        # We want min(all pair_vals, all wall_vals)
        # soft_min = -1/beta * log(sum(exp(-beta * vals)))
        # To avoid overflow, use logsumexp trick
        all_vals = np.concatenate([pair_vals.ravel(), wall_vals.ravel()])
        # Filter out very large values (inf) to avoid issues
        all_vals = all_vals[np.isfinite(all_vals)]
        
        if len(all_vals) == 0:
            return 0.0
            
        max_val = np.max(all_vals)
        # soft_min ≈ max_val - 1/beta * log(sum(exp(-beta * (all_vals - max_val))))
        # Actually: soft_min = -1/beta * log(sum(exp(-beta * all_vals)))
        # = -1/beta * log(exp(-beta * max_val) * sum(exp(-beta * (all_vals - max_val))))
        # = max_val - 1/beta * log(sum(exp(-beta * (all_vals - max_val))))
        
        shifted = -beta * (all_vals - max_val)
        # Clip to avoid underflow/overflow
        shifted = np.clip(shifted, -700, 0)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min_val = max_val - log_sum / beta
        
        return -soft_min_val  # Minimize negative soft_min to maximize soft_min

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Start with a 5x5x5 grid, take first 100, add jitter
        # 5x5x5 = 125 points
        ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        jitter = rng.uniform(-0.02, 0.02, size=(N, 3))
        grid = grid + jitter
        grid = np.clip(grid, -0.5, 0.5)
        return grid.flatten()

    best_r = 0.0
    best_c = None
    
    # Strategy:
    # 1. Try multiple random initializations
    # 2. For each, use iterative radius increase with penalty method
    # 3. Also try direct smooth optimization
    
    for seed in range(5):
        if time.time() - start_time > time_limit - 10:
            break
        
        x0 = make_initial_guess(seed)
        
        # Phase 1: Iterative radius increase with penalty
        R = 0.08
        max_R = 0.12
        x_current = x0.copy()
        
        while R < max_R:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Optimize penalty for current R
            res = minimize(
                lambda x: energy_penalty(x, R),
                x_current,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'maxfev': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-8:
                c_test = res.x.reshape(N, 3)
                r_test = _compute_radius(c_test)
                
                if r_test >= R - 1e-6:
                    # Accept
                    x_current = res.x
                    R += 0.002
                else:
                    # Stuck
                    break
            else:
                break
        
        # Phase 2: Polish with smooth objective
        if time.time() - start_time < time_limit - 5:
            res2 = minimize(
                lambda x: smooth_objective(x, beta=100.0),
                x_current,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res2.success or res2.fun < res.fun if hasattr(res, 'fun') else True:
                c_test2 = res2.x.reshape(N, 3)
                r_test2 = _compute_radius(c_test2)
                
                if r_test2 > best_r:
                    best_r = r_test2
                    best_c = c_test2.copy()
        
        # Also record the result from Phase 1
        c_phase1 = x_current.reshape(N, 3)
        r_phase1 = _compute_radius(c_phase1)
        if r_phase1 > best_r:
            best_r = r_phase1
            best_c = c_phase1.copy()
    
    # Final check: if we haven't improved, try a few more random starts with direct smooth opt
    if time.time() - start_time < time_limit - 10:
        for seed in range(5, 10):
            if time.time() - start_time > time_limit - 5:
                break
            
            x0 = make_initial_guess(seed)
            res = minimize(
                lambda x: smooth_objective(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfev': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            c_test = res.x.reshape(N, 3)
            r_test = _compute_radius(c_test)
            
            if r_test > best_r:
                best_r = r_test
                best_c = c_test.copy()
    
    if best_c is None:
        # Fallback
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    return best_c
