"""Packing of 50 equal spheres in a unit cube using L-BFGS-B on a soft-minimum objective."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START

def _pairwise_dists(centers):
    """Compute all pairwise distances (N x N matrix, diagonal = 0)."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(centers):
    """Return the radius: min(min_pair_dist/2, min_wall_dist)."""
    n = len(centers)
    if n < 2:
        mpd = float('inf')
    else:
        pd = _pairwise_dists(centers)
        np.fill_diagonal(pd, np.inf)
        mpd = np.min(pd)
    coords = np.abs(centers)
    max_coords = np.max(coords, axis=1)
    mwd = np.min(0.5 - max_coords)
    return min(mpd / 2.0, mwd)

def _soft_min_objective(x, t):
    """
    Objective to minimize: soft-min of (pairwise_dist - 2r) and (wall_dist - r) terms.
    We maximize r by minimizing -r + penalty for violations.
    
    Actually, let's use a different formulation:
    We want to maximize r such that:
      d_ij >= 2r for all i<j
      0.5 - |c_ik| >= r for all i,k
    
    Equivalently, minimize:
      f(x) = -r + (1/t) * log(sum(exp(-t * margin_ij)) + exp(-t * margin_wall))
    where margin_ij = d_ij - 2r, margin_wall = 0.5 - |c_ik| - r
    
    But r is not a variable here. Let's fix r and check feasibility, or use a penalty approach.
    
    Alternative: minimize the maximum violation.
    violation = max(0, 2r - d_ij) for pairs, max(0, r - (0.5 - |c_ik|)) for walls.
    We want to minimize violation + (1 - r) * some_weight? 
    
    Better approach: Use the standard trick.
    Let g(x) = min( min_{i<j} d_ij / 2, min_i min_k (0.5 - |c_ik|) ).
    We want to maximize g(x).
    
    Soft approximation: 
    G_t(x) = -(1/t) * log( sum_{i<j} exp(-t * d_ij / 2) + sum_i sum_k exp(-t * (0.5 - |c_ik|)) )
    This approximates -g(x) as t -> infinity.
    So we minimize G_t(x) to maximize g(x).
    """
    n = len(x)
    x = x.reshape(-1, 3)
    
    # Pairwise terms: d_ij / 2
    # Wall terms: 0.5 - |c_ik|
    
    # To avoid numerical issues with large t, we compute log-sum-exp carefully.
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    pair_terms = d / 2.0  # shape (N, N), symmetric, diag = inf
    
    # Wall distances
    abs_x = np.abs(x)
    wall_terms = 0.5 - np.max(abs_x, axis=1)  # shape (N,)
    
    # Combine all terms into one array
    # Number of pairs: N*(N-1)/2
    # Number of wall terms: N * 3? No, the constraint is min_k (0.5 - |c_ik|) >= r.
    # Actually the wall distance is min_k (0.5 - |c_ik|).
    # So the terms are the N values: w_i = min_k (0.5 - |c_ik|).
    
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)], wall_terms])
    
    # Soft min: -(1/t) * log(sum(exp(-t * term)))
    # Use log-sum-exp trick:
    # log(sum(exp(-t * a_i))) = -t * a_min + log(sum(exp(-t * (a_i - a_min))))
    a_min = np.min(all_terms)
    # If a_min is very negative, -t * a_min is large positive.
    # exp(-t * (a_i - a_min)) <= 1, so the sum is at most len(all_terms).
    shifted = -t * (all_terms - a_min)
    # Clip shifted to avoid overflow in exp (should be <= 0 mostly, but could be slightly positive due to fp)
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = a_min - log_sum / t
    
    # We want to maximize soft_min_val, so minimize -soft_min_val.
    return -soft_min_val

def _soft_min_grad(x, t):
    """Gradient of the soft-min objective."""
    n = len(x)
    x = x.reshape(-1, 3)
    
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    pair_terms = d / 2.0
    
    abs_x = np.abs(x)
    max_abs = np.max(abs_x, axis=1)
    wall_terms = 0.5 - max_abs
    
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)], wall_terms])
    
    a_min = np.min(all_terms)
    shifted = -t * (all_terms - a_min)
    shifted = np.clip(shifted, -50, 50)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp  # softmax weights
    
    # Gradient of -soft_min_val = - (a_min - log_sum/t) = -a_min + log_sum/t
    # d/dx_j of soft_min_val = sum_i weights_i * d(term_i)/dx_j
    # So d/dx_j of objective = - sum_i weights_i * d(term_i)/dx_j
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    n_pairs = n * (n - 1) // 2
    pair_weights = weights[:n_pairs]
    
    # Reconstruct indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # d(term_ij)/dx_i = (x_i - x_j) / (2 * d_ij)
    # term_ij = d_ij / 2
    # d(term_ij)/dx_i = (1/2) * d(d_ij)/dx_i = (1/2) * (x_i - x_j) / d_ij
    # d(term_ij)/dx_j = (1/2) * d(d_ij)/dx_j = (1/2) * (x_j - x_i) / d_ij
    
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        w = pair_weights[k]
        d_ij = d[i, j]
        if d_ij < 1e-12:
            continue
        factor = w / (2 * d_ij)
        grad[i] += factor * (x[i] - x[j])
        grad[j] += factor * (x[j] - x[i])
    
    # Wall part
    wall_weights = weights[n_pairs:]
    for i in range(n):
        w = wall_weights[i]
        # wall_term_i = 0.5 - max(|x_ix|, |x_iy|, |x_iz|)
        # d(wall_term_i)/dx_ik = -sign(x_ik) if |x_ik| is the max, else 0
        # If there are ties, we can average, but let's just pick one or use subgradient.
        # For simplicity, use the index of the max.
        max_k = np.argmax(np.abs(x[i]))
        if np.abs(x[i, max_k]) > 1e-12:
            grad[i, max_k] += -w * np.sign(x[i, max_k])
        else:
            # At origin, subgradient is any vector in unit ball. Use 0.
            pass
    
    # Objective gradient is - (gradient of soft_min_val)
    # Wait: objective = -soft_min_val.
    # d(objective)/dx = - d(soft_min_val)/dx = - sum weights * d(term)/dx
    # But I computed sum weights * d(term)/dx above as grad.
    # So return -grad.
    
    return -grad.reshape(-1)

def _ffd_packing():
    """First-Fit Decreasing heuristic: place spheres greedily on a grid."""
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    return centers

def _optimize_lbfgsb(centers, t, time_limit, max_iter=1000):
    """Run L-BFGS-B on the soft-min objective."""
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    start_time = time.time()
    
    def objective(x):
        return _soft_min_objective(x, t)
    
    def grad(x):
        return _soft_min_grad(x, t)
    
    # Use a callback to check time
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > time_limit - 2:
            stop_flag[0] = True
            return True
        return False
    
    result = minimize(
        objective, x0, method='L-BFGS-B', jac=grad,
        bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'callback': callback}
    )
    
    x_opt = result.x.reshape(-1, 3)
    # Ensure validity
    x_opt = np.clip(x_opt, -0.5, 0.5)
    return x_opt

def _local_search(centers, time_limit, seed=42):
    """
    Constraint-aware local search.
    """
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    
    start_time = time.time()
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < time_limit:
        # Compute score
        pd = _pairwise_dists(current)
        np.fill_diagonal(pd, np.inf)
        mpd = np.min(pd)
        coords = np.abs(current)
        max_coords = np.max(coords, axis=1)
        mwd = np.min(0.5 - max_coords)
        current_score = min(mpd / 2.0, mwd)
        
        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                current += rng.normal(0, 0.005, size=(N, 3))
                current = np.clip(current, -0.5, 0.5)
                no_improve_count = 0
                step_size = 0.02
                continue
        
        # Determine bottleneck
        if mpd / 2.0 < mwd:
            # Pair distance is bottleneck
            tree = cKDTree(current)
            dists, idxs = tree.query(current, k=2)
            nn_dists = dists[:, 1]
            min_nn = np.min(nn_dists)
            threshold = min_nn + 0.001
            candidates = np.where(nn_dists < threshold)[0]
            
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
            else:
                idx = rng.randint(N)
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
        else:
            # Wall distance is bottleneck
            wall_dists = 0.5 - np.max(np.abs(current), axis=1)
            min_wall_val = np.min(wall_dists)
            threshold = min_wall_val + 0.001
            candidates = np.where(wall_dists < threshold)[0]
            
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
            else:
                idx = rng.randint(N)
            
            norm = np.linalg.norm(current[idx])
            if norm > 1e-9:
                direction = -current[idx] / norm
            else:
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            
            current[idx] += direction * step_size
            
        current = np.clip(current, -0.5, 0.5)
        step_size *= 0.999
        if step_size < 0.001:
            step_size = 0.01

    return best, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Start with FFD grid
    centers = _ffd_packing()
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Multi-start L-BFGS-B with increasing temperature
    t_values = [50, 100, 200, 400]
    
    for t in t_values:
        if time.time() > deadline - 30:
            break
        
        # Run L-BFGS-B from current best
        time_for_lbfgsb = min(20.0, deadline - time.time() - 10)
        if time_for_lbfgsb < 5:
            break
        
        optimized = _optimize_lbfgsb(best_centers, t, time_limit=time_for_lbfgsb, max_iter=500)
        opt_score = _score(optimized)
        
        if opt_score > best_score:
            best_centers = optimized
            best_score = opt_score
    
    # Final polish with local search
    if time.time() < deadline - 10:
        ls_time = min(30.0, deadline - time.time() - 5)
        if ls_time > 5:
            ls_centers, ls_score = _local_search(best_centers, time_limit=ls_time, seed=42)
            if ls_score > best_score:
                best_centers = ls_centers
                best_score = ls_score
    
    return best_centers

# EVOLVE-BLOCK-END
