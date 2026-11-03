"""
Optimization of 100 equal spheres in a unit cube.
Uses smooth soft-min energy with L-BFGS-B and a final polish pass.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

TIME_LIMIT = 115.0  # safety margin before 120s kill


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


def _soft_min_radius(centers, beta=50.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns a smooth surrogate for min(min_pair_dist/2, min_wall_dist).
    """
    centers = centers.reshape(-1, 3)
    n = len(centers)
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # Pairwise distances (upper triangle only)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    np.fill_diagonal(dists, np.inf)
    
    # Use upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Combine all constraints: we want min of {pair_dists/2, wall_dists}
    # Use soft-min via log-sum-exp
    all_constraints = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - np.log(np.sum(np.exp(-beta * shifted))) / beta
    
    return soft_min


def _energy_and_gradient(x, R, beta=50.0):
    """
    Calculates smooth energy and gradient for a target radius R.
    Uses soft-min based energy to avoid nonsmoothness.
    Energy = max(0, R - soft_min_radius)^2 + penalty for being outside cube
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    grad = np.zeros_like(centers)
    
    # Compute all constraint values
    # Wall distances
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # All constraints: pair_dists/2 and wall_dists
    # We want to maximize the minimum of these, so minimize max(0, R - constraint)
    
    # For smoothness, use soft-min
    all_constraints = np.concatenate([pair_dists / 2.0, wall_dists])
    
    # Soft min via log-sum-exp
    min_val = np.min(all_constraints)
    shifted = all_constraints - min_val
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_val - log_sum / beta
    
    # Energy: if soft_min_val < R, we have a violation
    if soft_min_val < R:
        violation = R - soft_min_val
        energy = violation ** 2
    else:
        energy = 0.0
        # If no violation, gradient is zero (or we can still compute for smoothness)
        # For better optimization, always compute gradient
        energy = 0.0
    
    # Gradient computation using chain rule
    # d(soft_min)/dx = -1/beta * d(log_sum)/dx
    # d(log_sum)/dx = -beta * sum_j exp(-beta * (x_j - min_val)) / sum_j exp(-beta * (x_j - min_val)) * dx_j/dx
    # So d(soft_min)/dx = sum_j w_j * dx_j/dx where w_j = exp(-beta*(x_j-min_val))/sum(exp(...))
    
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min with respect to centers
    # For pairwise constraints: d(pair_dist_ij)/d(centers[i]) = (centers[i] - centers[j]) / pair_dist_ij
    # For wall constraints: d(wall_dist_i)/d(centers[i]) = -sign(centers[i, k]) for the k with max abs
    
    grad_soft_min = np.zeros_like(centers)
    
    # Pairwise contributions
    for k in range(len(i_idx)):
        w = weights[k]  # weight for this pair constraint
        i = i_idx[k]
        j = j_idx[k]
        d = pair_dists[k]
        if d > 1e-15:
            vec = (centers[i] - centers[j]) / d
            # d(pair_dist/2)/d(centers[i]) = 0.5 * vec
            # d(soft_min)/d(centers[i]) += w * 0.5 * vec
            # d(soft_min)/d(centers[j]) += w * (-0.5 * vec)
            grad_soft_min[i] += 0.5 * w * vec
            grad_soft_min[j] -= 0.5 * w * vec
    
    # Wall contributions
    n_pairs = len(i_idx)
    for i in range(n):
        w = weights[n_pairs + i]  # weight for this wall constraint
        max_val = max_abs[i]
        for k in range(3):
            if np.isclose(abs_centers[i, k], max_val):
                # d(wall_dist)/d(centers[i,k]) = -sign(centers[i,k])
                grad_soft_min[i, k] += w * (-np.sign(centers[i, k]))
                break
    
    # Energy gradient: d/dx [max(0, R - soft_min)^2] = -2*(R - soft_min) * d(soft_min)/dx
    # But we only want this when soft_min < R
    if soft_min_val < R:
        factor = -2.0 * (R - soft_min_val)
        grad = factor * grad_soft_min
    else:
        # No violation, but we can still add a small gradient to push toward larger radius
        # This helps escape flat regions
        grad = 0.01 * grad_soft_min  # small push to increase radius
    
    # Add hard boundary penalty to keep centers inside cube
    # Penalty for being outside [-0.5, 0.5]
    for dim in range(3):
        below = centers[:, dim] < -0.5
        above = centers[:, dim] > 0.5
        if np.any(below):
            penalty = (centers[below, dim] + 0.5) ** 2
            energy += np.sum(penalty)
            grad[below, dim] += 2.0 * (centers[below, dim] + 0.5)
        if np.any(above):
            penalty = (centers[above, dim] - 0.5) ** 2
            energy += np.sum(penalty)
            grad[above, dim] += 2.0 * (centers[above, dim] - 0.5)
    
    return energy, grad.flatten()


def _polish_centers(centers, max_iter=500, time_budget=10.0):
    """
    Final high-precision polish pass with tighter tolerances.
    """
    start_time = time.time()
    current_radius = _compute_radius(centers)
    
    # Use a slightly smaller target to ensure we're in the feasible region
    target_R = current_radius * 0.999
    
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta=100.0)
        return val, g

    bounds = [(-0.49999, 0.49999)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.49999, 0.49999)
        polished_radius = _compute_radius(polished)
        
        if polished_radius >= current_radius and time.time() - start_time < time_budget:
            return polished
    except Exception:
        pass
    
    return centers


def _optimize_for_radius(init_centers, target_R, max_iter=200, beta=50.0):
    """
    Attempts to optimize the configuration to fit within radius target_R.
    """
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, target_R, beta=beta)
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
        
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return init_centers, np.inf


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_radius = 0.0
    best_centers = None
    
    # Grid Initialization
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/ny, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)

    # 5x4x5 = 100
    grid_centers = make_grid(5, 4, 5)
    
    # Random Initializations
    rng = np.random.RandomState(42)
    inits = [grid_centers.copy()]
    
    for i in range(7):
        rc = rng.uniform(-0.4, 0.4, size=(N, 3))
        inits.append(rc)
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # Calculate initial radius
        r_est = _compute_radius(init)
        
        # Start with a slightly larger target
        target_R = max(r_est * 1.02, 0.06)
        
        current_centers = init.copy()
        current_radius = r_est
        
        # Iterative optimization with increasing radius
        for step in range(50):
            if time.time() > deadline - 15.0:
                break
            
            # Optimize for current target_R
            opt_centers, energy = _optimize_for_radius(current_centers, target_R, max_iter=150, beta=50.0)
            opt_radius = _compute_radius(opt_centers)
            
            if opt_radius > current_radius:
                current_centers = opt_centers
                current_radius = opt_radius
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = current_centers.copy()
                
                # Increase target slightly
                target_R = current_radius * 1.003
            else:
                # Failed to improve, try a smaller increase or break
                if energy < 1e-8:
                    # Very small energy, we're at a local optimum for this R
                    # Try a much smaller increase
                    target_R = current_radius * 1.001
                else:
                    # Not converged, try to heal
                    heal_R = target_R * 0.995
                    heal_centers, heal_energy = _optimize_for_radius(current_centers, heal_R, max_iter=50, beta=50.0)
                    heal_radius = _compute_radius(heal_centers)
                    
                    if heal_radius > current_radius:
                        current_centers = heal_centers
                        current_radius = heal_radius
                        target_R = current_radius * 1.002
                    else:
                        # Stuck, move to next init
                        break
        
        # Update best
        r_final = _compute_radius(current_centers)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = current_centers.copy()

    # Final polish pass
    if best_centers is not None and time.time() < deadline - 5.0:
        polished = _polish_centers(best_centers, max_iter=300, time_budget=min(10.0, deadline - 5.0 - time.time()))
        polished_radius = _compute_radius(polished)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished
    
    # Final safety check
    if best_centers is None:
        best_centers = grid_centers
    else:
        best_centers = np.clip(best_centers, -0.4999, 0.4999)
        
    return best_centers
# EVOLVE-BLOCK-END
