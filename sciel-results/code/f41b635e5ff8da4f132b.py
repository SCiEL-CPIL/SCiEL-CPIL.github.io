import numpy as np
import time
import scipy.optimize

N = 100

def _compute_radius(c):
    """
    Compute the largest common radius for given centers.
    c: (N, 3) array of centers.
    Returns: min(min_pair_dist/2, min_wall_dist)
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _smooth_radius_surrogate(c, beta=50.0):
    """
    Smooth surrogate for the radius using log-sum-exp soft-min.
    Maximizes a smooth approximation of min(min_pair_dist/2, min_wall_dist).
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    wall_dists_flat = wall_dists.flatten()
    
    # Combine all constraints into a single array
    # We want to maximize t such that:
    #   t <= dists[i,j]/2 for all i<j
    #   t <= wall_dists[i,k] for all i,k
    # Use soft-min: -1/beta * log(sum(exp(-beta * constraints)))
    
    # Pair constraints: dists[i,j]/2 for i<j
    iu = np.triu_indices(n, k=1)
    pair_constraints = dists[iu] / 2.0
    
    # Wall constraints
    wall_constraints = wall_dists_flat
    
    # Combine
    all_constraints = np.concatenate([pair_constraints, wall_constraints])
    
    # Soft-min
    min_constraint = np.min(all_constraints)
    # Shift for numerical stability
    shifted = -beta * (all_constraints - min_constraint)
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    soft_min = min_constraint - log_sum_exp / beta
    
    return soft_min

def _objective_and_grad(c, beta=50.0):
    """
    Compute objective (negative of smooth radius) and gradient.
    We minimize -smooth_radius, which equals maximizing smooth_radius.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0, np.zeros_like(c)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    wall_dists_flat = wall_dists.flatten()
    
    # Constraints
    iu = np.triu_indices(n, k=1)
    pair_constraints = dists[iu] / 2.0
    wall_constraints = wall_dists_flat
    all_constraints = np.concatenate([pair_constraints, wall_constraints])
    
    # Soft-min computation
    min_c = np.min(all_constraints)
    shifted = -beta * (all_constraints - min_c)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    log_sum_exp = np.log(sum_exp)
    soft_min = min_c - log_sum_exp / beta
    
    # Gradient of soft-min w.r.t. each constraint
    # d(soft_min)/d(constraint_k) = -exp_shifted[k] / sum_exp
    weights = -exp_shifted / sum_exp
    
    # Gradient w.r.t. c
    grad = np.zeros_like(c)
    
    # Pair constraints gradient
    # d(pair_constraints[m])/d(c) where pair_constraints[m] = dists[i,j]/2
    # d(dists[i,j])/d(c[i]) = (c[i] - c[j]) / dists[i,j]
    # d(dists[i,j])/d(c[j]) = (c[j] - c[i]) / dists[i,j]
    
    pair_weights = weights[:len(pair_constraints)]
    for m in range(len(iu[0])):
        i, j = iu[0][m], iu[1][m]
        w = pair_weights[m]
        d = dists[i, j]
        if d > 1e-8:
            direction = (c[i] - c[j]) / d
            grad[i] += w * 0.5 * direction
            grad[j] += w * 0.5 * (-direction)
    
    # Wall constraints gradient
    # wall_dists[i,k] = 0.5 - |c[i,k]|
    # d/d(c[i,k]) = -sign(c[i,k])
    wall_weights = weights[len(pair_constraints):]
    for idx in range(n * 3):
        i = idx // 3
        k = idx % 3
        w = wall_weights[idx]
        if c[i, k] > 0:
            grad[i, k] += w * (-1.0)
        elif c[i, k] < 0:
            grad[i, k] += w * (1.0)
    
    # We minimize -soft_min, so objective = -soft_min, grad = -grad
    objective = -soft_min
    grad = -grad
    
    return objective, grad

def _optimize(c_init, time_limit, beta=50.0):
    """
    Gradient-based optimization using L-BFGS-B on the smooth surrogate.
    c_init: Initial centers (N, 3)
    time_limit: Seconds to spend on this optimization attempt.
    Returns: Optimized centers, best radius found.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    start_time = time.time()
    
    # Bounds for L-BFGS-B
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def callback(xk):
        nonlocal best_c, best_r
        r = _compute_radius(xk.reshape(N, 3))
        if r > best_r:
            best_r = r
            best_c = xk.reshape(N, 3).copy()
            
    try:
        # Use L-BFGS-B
        def obj_flat(x):
            c_2d = x.reshape(N, 3)
            val, grad = _objective_and_grad(c_2d, beta)
            return val, grad.flatten()
            
        result = scipy.optimize.minimize(
            obj_flat, 
            c.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            callback=callback,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < obj_flat(c.flatten())[0]:
            c_opt = result.x.reshape(N, 3)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
                
    except Exception as e:
        pass

    # If time allows, do a few steps of local refinement with exact radius
    t_left = time_limit - (time.time() - start_time)
    if t_left > 0.5:
        c = best_c.copy()
        step = 0.005
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for i in range(N):
                for d in range(3):
                    for sign in [-1, 1]:
                        c_new = c.copy()
                        c_new[i, d] += sign * step
                        if c_new[i, d] < -0.5 or c_new[i, d] > 0.5:
                            continue
                        r_new = _compute_radius(c_new)
                        if r_new > best_r:
                            c = c_new
                            best_c = c_new.copy()
                            best_r = r_new
                            improved = True
            if not improved:
                step *= 0.5
                if step < 1e-5:
                    break

    return best_c, best_r

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    """
    total_time_limit = 115.0  # Safety margin from 120s
    start_time = time.time()
    
    best_c = None
    best_r = -1.0
    
    # Strategy 1: Grid initialization
    nx, ny, nz = 4, 5, 5
    ticks_x = (np.arange(nx) + 0.5) / nx - 0.5
    ticks_y = (np.arange(ny) + 0.5) / ny - 0.5
    ticks_z = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid_points = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
    c_grid = grid_points[:N].copy()
    
    # Run optimization on grid start
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_grid, t_left - 1.0, beta=50.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 2: Random initialization
    c_rand = np.random.uniform(-0.4, 0.4, (N, 3))
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_rand, t_left - 1.0, beta=50.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 3: Perturbed Grid
    c_perturbed = c_grid + np.random.randn(N, 3) * 0.05
    c_perturbed = np.clip(c_perturbed, -0.5, 0.5)
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 5:
        c_opt, r_opt = _optimize(c_perturbed, t_left - 1.0, beta=50.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Strategy 4: Different Beta
    if best_c is not None:
        t_left = total_time_limit - (time.time() - start_time)
        if t_left > 5:
             c_opt, r_opt = _optimize(best_c, t_left - 1.0, beta=100.0)
             if r_opt > best_r:
                 best_r = r_opt
                 best_c = c_opt.copy()

    # If we still have time, do a final polish on the best solution
    t_left = total_time_limit - (time.time() - start_time)
    if t_left > 2.0 and best_c is not None:
        c_opt, r_opt = _optimize(best_c, t_left - 0.5, beta=100.0)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    if best_c is None:
        return c_grid
        
    return best_c

# EVOLVE-BLOCK-END
