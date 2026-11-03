import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _score(centers):
    """Compute the maximum radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _soft_min_obj(x, beta):
    """
    Soft-min objective: -log(sum(exp(-beta * d_i))) / beta
    where d_i are the clearance values (pair distances / 2 and wall distances).
    As beta -> inf, this approaches -min(d_i).
    We minimize this (i.e., maximize the soft-min of clearances).
    """
    centers = x.reshape(N, 3)
    
    # Wall clearances: 0.5 - |c_ik| for each center and axis
    # For each center i, wall clearance = min over k of (0.5 - |c_ik|)
    # Actually, the wall clearance for center i is min_k (0.5 - |c_ik|)
    # But we need the min over all centers and all axes
    # wall_clearance_i = min_k (0.5 - |c_ik|)
    # total wall clearance = min_i wall_clearance_i
    
    # Pair clearances: |c_i - c_j| / 2 for i < j
    # total pair clearance = min_{i<j} |c_i - c_j| / 2
    
    # Overall clearance = min(wall_clearance, pair_clearance)
    
    # Collect all clearance values
    # Wall: for each i, k: 0.5 - |c_ik|
    wall_vals = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Pair: for i < j: |c_i - c_j| / 2
    # Use upper triangle
    ii, jj = np.triu_indices(N, k=1)
    diff = centers[ii] - centers[jj]
    pair_vals = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals])
    
    # Soft min: -log(sum(exp(-beta * v))) / beta
    # For numerical stability, subtract max
    v_min = np.min(all_vals)
    shifted = all_vals - v_min  # >= 0
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = -(v_min + log_sum / beta)
    
    return -soft_min_val  # We want to minimize this (maximize soft_min)


def _soft_min_grad(x, beta):
    """
    Gradient of the soft-min objective with respect to x.
    """
    centers = x.reshape(N, 3)
    
    # Wall values
    wall_vals = 0.5 - np.abs(centers)  # (N, 3)
    
    # Pair values
    ii, jj = np.triu_indices(N, k=1)
    diff = centers[ii] - centers[jj]
    pair_vals = np.sqrt(np.sum(diff ** 2, axis=1)) / 2.0
    
    all_vals = np.concatenate([wall_vals.ravel(), pair_vals])
    
    v_min = np.min(all_vals)
    shifted = all_vals - v_min
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Soft min = -(v_min + log(sum_exp) / beta)
    # d(soft_min)/d(v_k) = -1/beta * exp(-beta*(v_k - v_min)) / sum_exp
    # = -exp_vals[k] / (beta * sum_exp)
    
    weights = -exp_vals / (beta * sum_exp)  # d(soft_min)/d(v_k)
    
    # Now compute d(x)/d(v_k) for each clearance value
    grad = np.zeros_like(x)
    
    # Wall clearances: v = 0.5 - |c_ik|
    # d(v)/d(c_ik) = -sign(c_ik)
    # d(soft_min)/d(c_ik) = weights[k] * (-sign(c_ik))
    wall_weights = weights[:N * 3].reshape(N, 3)
    # For each (i, k): contribution to grad[i, k] = wall_weights[i, k] * (-sign(centers[i, k]))
    signs = np.sign(centers)
    # Handle zero: sign(0) = 0, which is fine (gradient undefined at 0, but measure zero)
    grad += -wall_weights * signs
    
    # Pair clearances: v = |c_i - c_j| / 2
    # d(v)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = diff / (2 * |diff|)
    # d(v)/d(c_j) = -(c_i - c_j) / (2 * |c_i - c_j|)
    pair_weights = weights[N * 3:]  # shape (num_pairs,)
    norms = np.sqrt(np.sum(diff ** 2, axis=1))
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    unit_vecs = diff / safe_norms[:, np.newaxis]  # (num_pairs, 3)
    
    # d(soft_min)/d(c_i) for pair (i,j) = pair_weights[p] * unit_vecs[p]
    # d(soft_min)/d(c_j) for pair (i,j) = -pair_weights[p] * unit_vecs[p]
    
    # Accumulate using np.add.at
    for p in range(len(pair_weights)):
        grad[ii[p]] += pair_weights[p] * unit_vecs[p]
        grad[jj[p]] -= pair_weights[p] * unit_vecs[p]
    
    return grad.ravel()


def _soft_min_obj_grad(x, beta):
    """Combined objective and gradient for L-BFGS-B."""
    obj = _soft_min_obj(x, beta)
    grad = _soft_min_grad(x, beta)
    return obj, grad


def _init_grid():
    """Structured grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_stage(x0, beta, maxiter, start_time, time_budget):
    """Run L-BFGS-B for one beta stage."""
    def fun_and_grad(x):
        if time.time() - start_time > time_budget:
            return _soft_min_obj(x, beta), np.zeros_like(x)
        return _soft_min_obj_grad(x, beta)
    
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    try:
        result = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'maxls': 50})
        return result.x
    except Exception:
        return x0


def construct():
    start_time = time.time()
    
    # Initialize with grid
    centers = _init_grid()
    x = centers.ravel()
    x = np.clip(x, -0.49, 0.49)
    
    best_x = x.copy()
    best_score = _score(x.reshape(N, 3))
    
    # Multi-stage beta annealing
    # Start with small beta (smooth) and increase to large beta (sharp)
    beta_schedule = [5, 10, 20, 50, 100]
    maxiter_schedule = [100, 150, 200, 300, 400]
    
    for i, (beta, maxiter) in enumerate(zip(beta_schedule, maxiter_schedule)):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_budget = min(DEADLINE - 5 - (time.time() - start_time), 30.0)
        if time_budget < 5:
            break
        
        x = _optimize_stage(x, beta, maxiter, start_time, time_budget)
        x = np.clip(x, -0.49, 0.49)
        
        current_score = _score(x.reshape(N, 3))
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
    
    # Perturbation restarts
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Perturb best solution
        x_perturbed = best_x + np.random.randn(N * 3) * 0.01
        x_perturbed = np.clip(x_perturbed, -0.49, 0.49)
        
        # Run a few stages
        for beta, maxiter in [(20, 100), (50, 150)]:
            if time.time() - start_time > DEADLINE - 5:
                break
            time_budget = min(DEADLINE - 5 - (time.time() - start_time), 15.0)
            if time_budget < 3:
                break
            x_perturbed = _optimize_stage(x_perturbed, beta, maxiter, start_time, time_budget)
            x_perturbed = np.clip(x_perturbed, -0.49, 0.49)
        
        current_score = _score(x_perturbed.reshape(N, 3))
        if current_score > best_score:
            best_score = current_score
            best_x = x_perturbed.copy()
    
    # Final polish with high beta
    if time.time() - start_time < DEADLINE - 10:
        for beta in [100, 200]:
            if time.time() - start_time > DEADLINE - 5:
                break
            time_budget = min(DEADLINE - 5 - (time.time() - start_time), 20.0)
            if time_budget < 3:
                break
            best_x = _optimize_stage(best_x, beta, 200, start_time, time_budget)
            best_x = np.clip(best_x, -0.49, 0.49)
        
        final_score = _score(best_x.reshape(N, 3))
        if final_score > best_score:
            best_score = final_score
    
    return best_x.reshape(N, 3)
