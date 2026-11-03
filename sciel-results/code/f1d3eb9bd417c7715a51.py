"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _radius_for_centres(centres):
    """Compute the maximum common radius for given centres (exact)."""
    n = len(centres)
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(centres)
    min_wall = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall)

def _soft_min_logsumexp(values, beta):
    """
    Soft minimum via log-sum-exp: -1/beta * log(sum(exp(-beta * v))).
    Smooth approximation to min(v). Larger beta -> closer to true min.
    """
    # For numerical stability
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - np.log(np.sum(np.exp(-beta * (values - min_val)))) / beta

def _smooth_objective(centres, beta):
    """
    Smooth surrogate: maximize soft-min of all constraints.
    Constraints: pairwise distances / 2, and wall distances.
    Returns (value, gradient) for minimization of -soft_min.
    """
    n = len(centres)
    # Pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centres)
    
    # All constraint values (we want to maximize the minimum)
    # Pair constraints: dists / 2
    # Wall constraints: wall_dists
    
    # Use soft-min over all constraints
    # For pairs: values are dists[i,j]/2 for i<j
    # For walls: values are wall_dists[i,k]
    
    # Compute soft-min value
    pair_vals = dists / 2.0
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_vals[iu]
    
    all_vals = np.concatenate([pair_flat, wall_dists.ravel()])
    
    # Soft min
    beta_safe = max(beta, 1e-6)
    min_val = np.min(all_vals)
    exp_vals = np.exp(-beta_safe * (all_vals - min_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_val - log_sum / beta_safe
    
    # Gradient of -soft_min w.r.t. centres
    # d(soft_min)/d(all_vals) = -exp(-beta*(v - min)) / sum(exp(-beta*(v - min)))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(centres)
    
    # Pair contributions
    # For each pair (i,j) with i<j, weight w
    # d(soft_min)/d(c_i) += w * d(dist_ij/2)/d(c_i) = w * (c_i - c_j) / dist_ij
    # d(soft_min)/d(c_j) += w * d(dist_ij/2)/d(c_j) = w * (c_j - c_i) / dist_ij
    for idx in range(len(iu[0])):
        i = iu[0][idx]
        j = iu[1][idx]
        w = weights[idx]
        diff = centres[i] - centres[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            grad[i] += w * diff / dist
            grad[j] += w * (-diff) / dist
    
    # Wall contributions
    # wall_dists[i,k] = 0.5 - |c_ik|
    # d(wall_dists[i,k])/d(c_ik) = -sign(c_ik)
    wall_flat_idx = 0
    for i in range(n):
        for k in range(3):
            w = weights[wall_flat_idx]
            ck = centres[i, k]
            if ck > 0:
                grad[i, k] += w * (-1.0)
            elif ck < 0:
                grad[i, k] += w * (1.0)
            else:
                grad[i, k] += w * 0.0
            wall_flat_idx += 1
    
    # We want to maximize soft_min, so minimize -soft_min
    # Gradient of -soft_min is -grad
    return -soft_min_val, -grad

def _optimize_continuation(start, deadline, max_beta=200.0):
    """
    Continuation method: ramp up beta from small to large.
    At each beta, run L-BFGS-B to minimize -soft_min.
    """
    centres = start.copy()
    n = len(centres)
    
    # Beta schedule: geometric progression
    betas = [1.0, 5.0, 15.0, 40.0, 80.0, 150.0, max_beta]
    
    bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (n * 3)
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        def objective_flat(x):
            c = x.reshape(n, 3)
            val, grad = _smooth_objective(c, beta)
            return val, grad.ravel()
        
        x0 = centres.ravel()
        
        try:
            result = minimize(
                objective_flat,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            centres = result.x.reshape(n, 3)
        except Exception:
            pass
        
        # Clip to valid region
        centres = np.clip(centres, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centres, _radius_for_centres(centres)

def _initial_grid(n_spheres):
    """Create an initial grid configuration."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres]

def _random_perturb(centres, sigma=0.02):
    """Randomly perturb centres."""
    perturbed = centres + np.random.randn(*centres.shape) * sigma
    return np.clip(perturbed, -0.49, 0.49)

def construct():
    """
    Construct 50 sphere centres in the unit cube [-1/2, 1/2]^3 to maximize
    the common radius.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Start with grid
    best_centres = _initial_grid(N)
    best_r = _radius_for_centres(best_centres)
    
    # Continuation optimization from grid
    opt_centres, opt_r = _optimize_continuation(best_centres, deadline)
    if opt_r > best_r:
        best_r = opt_r
        best_centres = opt_centres
    
    # Multiple random restarts with perturbation
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate perturbed start
        if restart < 3:
            sigma = 0.03
        elif restart < 8:
            sigma = 0.06
        else:
            sigma = 0.1
        
        start = _random_perturb(best_centres, sigma=sigma)
        
        # Optimize with continuation
        opt_centres, opt_r = _optimize_continuation(start, deadline)
        
        if opt_r > best_r:
            best_r = opt_r
            best_centres = opt_centres
    
    # Final high-precision polish with very large beta
    if time.time() < deadline - 5:
        def objective_flat_high_beta(x):
            c = x.reshape(N, 3)
            val, grad = _smooth_objective(c, 500.0)
            return val, grad.ravel()
        
        bounds = [(-0.5 + 1e-10, 0.5 - 1e-10)] * (N * 3)
        x0 = best_centres.ravel()
        
        try:
            result = minimize(
                objective_flat_high_beta,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            final_centres = result.x.reshape(N, 3)
            final_r = _radius_for_centres(final_centres)
            if final_r > best_r:
                best_r = final_r
                best_centres = final_centres
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
