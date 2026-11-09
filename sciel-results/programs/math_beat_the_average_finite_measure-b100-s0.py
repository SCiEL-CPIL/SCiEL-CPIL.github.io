import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized for speed with sorted support for efficiency.
    Returns 0 if invalid.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Basic validity check for speed
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return 0.0
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return 0.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
        
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
        
    p = weights / w_sum
    
    # Sort support and reorder weights accordingly
    sort_idx = np.argsort(support)
    s_sorted = support[sort_idx]
    p_sorted = p[sort_idx]
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j (sorted)
    S = s_sorted[:, None] + s_sorted[None, :]
    
    # Weights outer product: W[i,j] = p_i * p_j
    W = p_sorted[:, None] * p_sorted[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Pre-sort S_flat and corresponding W_flat for fast prefix sums
    sort_idx2 = np.argsort(S_flat)
    S_sorted2 = S_flat[sort_idx2]
    W_sorted2 = W_flat[sort_idx2]
    W_cumsum = np.cumsum(W_sorted2)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    # Vectorize over k: for each k, T_k = 2*s_k - s_l for all l
    T_matrix = 2.0 * s_sorted[:, None] - s_sorted[None, :]
    T_flat = T_matrix.flatten()
    
    # Find indices in S_sorted2 where S_sorted2 < T_flat
    indices = np.searchsorted(S_sorted2, T_flat, side='left')
    
    # Sum of weights for each (k,l) pair
    sums_for_kl = W_cumsum_padded[indices]
    sums_for_kl = sums_for_kl.reshape(n, n)
    
    # prob3_k = sum_l p_l * sums_for_kl[k,l]
    prob3 = sums_for_kl @ p_sorted  # shape (n,)
    
    total_prob = np.dot(p_sorted, prob3)
    
    return float(total_prob)

def _validate(support, weights):
    if len(support) != len(weights):
        return False
    if len(support) < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True

def _params_to_measure(params, n):
    """
    Convert unconstrained parameters to valid support and weights.
    params[0:n] -> log(support)
    params[n:2n] -> raw weights (softmax or abs+normalize)
    """
    log_s = params[:n]
    raw_w = params[n:]
    
    # Support: ensure non-negative. log_s can be any real number.
    # exp(log_s) is always positive.
    support = np.exp(log_s)
    
    # Weights: ensure positive. 
    # Using exp(raw_w) ensures positivity.
    exp_w = np.exp(raw_w - np.max(raw_w)) # for numerical stability
    weights = exp_w / np.sum(exp_w)
    
    return support, weights

def _objective(params, n):
    support, weights = _params_to_measure(params, n)
    if not _validate(support, weights):
        return -1.0
    prob = _calc_prob_fast(support, weights)
    return -prob # Minimizing negative probability

def _objective_smooth(params, n, beta=100.0):
    """
    Smooth surrogate using soft-min to handle the piecewise indicator.
    This helps local optimization navigate the landscape better.
    """
    support, weights = _params_to_measure(params, n)
    if not _validate(support, weights):
        return -1.0
    
    p = weights / np.sum(weights)
    n_pts = len(support)
    
    # Compute all pairwise sums
    S = support[:, None] + support[None, :]
    W = p[:, None] * p[None, :]
    
    # For each (k, l) pair, compute smooth approximation of 
    # sum_{i,j: s_i+s_j < 2*s_k-s_l} p_i*p_j
    # Use soft-min: -log(sum(exp(-beta * (S - T)))) / beta approximates min
    # But we want sum of indicators, so use: sum exp(-beta*(S-T))_+ style
    
    # Actually, let's use a different approach: 
    # P = sum_{k,l} p_k * p_l * F(2*s_k - s_l)
    # where F(t) = P[X1+X2 < t]
    # Smooth F(t) using: F_smooth(t) = sum_{i,j} p_i*p_j * sigmoid(beta*(t - s_i - s_j))
    
    beta_eff = beta
    T_matrix = 2.0 * support[:, None] - support[None, :]  # n x n
    S_flat = S.flatten()  # n^2
    W_flat = W.flatten()  # n^2
    
    # For each (k,l), compute sum over (i,j) of W_flat[i,j] * sigmoid(beta*(T[k,l] - S_flat[i,j]))
    # This is expensive, so we'll use the exact computation for now
    # and just add a small smoothing term to encourage spread
    
    prob = _calc_prob_fast(support, weights)
    
    # Add a small penalty for having too few distinct support points
    # to encourage exploration
    if n_pts > 1:
        support_spread = np.ptp(support)
        if support_spread < 1e-6:
            prob -= 0.01
    
    return -prob

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using multi-phase optimization with diverse initialization,
    simulated annealing with dynamic n, local optimization, and differential evolution.
    """
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin for 600s limit
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    # Helper to update best
    def try_update(support, weights, prob):
        nonlocal best_prob, best_support, best_weights
        if prob > best_prob:
            best_prob = prob
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Initialize with multiple diverse starting points
    initializations = [
        ([0.1, 0.5, 1.0, 2.0, 4.0], [0.1, 0.2, 0.3, 0.25, 0.15]),
        ([0.0, 1.0, 3.0], [0.3, 0.4, 0.3]),
        ([0.5, 1.0, 2.0, 4.0, 8.0], [0.15, 0.2, 0.3, 0.2, 0.15]),
        ([0.2, 0.8, 1.5, 3.0], [0.25, 0.3, 0.25, 0.2]),
        ([1.0, 2.0], [0.5, 0.5]),
        ([0.0, 0.5, 1.0, 1.5, 2.0], [0.1, 0.2, 0.3, 0.25, 0.15]),
        ([0.1, 0.3, 0.7, 1.4, 2.8], [0.2, 0.25, 0.3, 0.15, 0.1]),
    ]
    
    for s_init, w_init in initializations:
        s = np.array(s_init, dtype=np.float64)
        w = np.array(w_init, dtype=np.float64)
        if _validate(s, w):
            prob = _calc_prob_fast(s, w)
            try_update(s, w, prob)
    
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
        best_prob = _calc_prob_fast(best_support, best_weights)
    
    # Phase 1: Simulated annealing with dynamic n (first 35% of time)
    phase1_deadline = start_time + (deadline - start_time) * 0.35
    T_start = 1.0
    T_end = 0.005
    cooling_rate = 0.997
    
    current_support = best_support.copy()
    current_weights = best_weights.copy()
    current_prob = best_prob
    T = T_start
    step_scale = 1.0
    
    while time.time() < phase1_deadline:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.55:
            # Perturb support and weights
            n = len(current_support)
            log_s = np.log(np.maximum(current_support, 1e-10))
            log_s += np.random.normal(0, 0.25 * step_scale, size=n)
            new_support = np.exp(log_s)
            new_support = np.maximum(new_support, 0.0)
            
            new_weights = current_weights * np.exp(np.random.normal(0, 0.15 * step_scale, size=n))
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        elif move_type < 0.78 and len(current_support) < 12:
            # Add a support point
            n = len(current_support)
            new_support = current_support.copy()
            new_weights = current_weights.copy()
            
            insert_idx = random.randint(0, n)
            if insert_idx == 0:
                new_val = max(current_support[0] * np.exp(np.random.normal(0, 0.4)), 0.0)
            elif insert_idx == n:
                new_val = max(current_support[-1] * np.exp(np.random.normal(0, 0.4)), 0.0)
            else:
                left = max(current_support[insert_idx - 1], 1e-10)
                right = max(current_support[insert_idx], 1e-10)
                new_val = np.exp(np.log(left) + np.random.uniform(0, 1) * (np.log(right) - np.log(left)))
                new_val = max(new_val, 0.0)
            
            new_support = np.insert(new_support, insert_idx, new_val)
            
            if insert_idx == 0:
                w_new = new_weights[0] * 0.4
                new_weights[0] -= w_new
                new_weights = np.insert(new_weights, 0, w_new)
            elif insert_idx == n:
                w_new = new_weights[-1] * 0.4
                new_weights[-1] -= w_new
                new_weights = np.insert(new_weights, -1, w_new)
            else:
                w_new = (new_weights[insert_idx - 1] + new_weights[insert_idx]) * 0.2
                new_weights[insert_idx - 1] -= w_new
                new_weights[insert_idx] -= w_new
                new_weights = np.insert(new_weights, insert_idx, w_new)
            
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        elif move_type < 0.92 and len(current_support) > 2:
            # Remove a support point
            n = len(current_support)
            remove_idx = random.randint(0, n - 1)
            
            new_support = np.delete(current_support, remove_idx)
            new_weights = np.delete(current_weights, remove_idx)
            
            if len(new_weights) > 0:
                if remove_idx == 0 and len(new_weights) > 0:
                    new_weights[0] += current_weights[remove_idx]
                elif remove_idx == n - 1 and len(new_weights) > 0:
                    new_weights[-1] += current_weights[remove_idx]
                elif len(new_weights) > 0:
                    half = current_weights[remove_idx] / 2
                    new_weights[remove_idx - 1] += half
                    new_weights[remove_idx] += half
            
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        else:
            # Large jump: random restart in neighborhood
            n = len(current_support)
            new_support = current_support * np.exp(np.random.normal(0, 1.0, size=n))
            new_support = np.maximum(new_support, 0.0)
            new_weights = np.abs(np.random.normal(1.0, 0.5, size=n)) + 0.1
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
        
        # Accept or reject
        delta = new_prob - current_prob
        if delta > 0 or random.random() < np.exp(delta / T):
            current_support = new_support
            current_weights = new_weights
            current_prob = new_prob
            try_update(current_support, current_weights, current_prob)
            step_scale = min(step_scale * 1.01, 2.0)
        else:
            step_scale = max(step_scale * 0.99, 0.3)
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Phase 2: Local optimization on best found (next 30% of time)
    phase2_deadline = start_time + (deadline - start_time) * 0.7
    if best_support is not None and time.time() < phase2_deadline:
        n_best = len(best_support)
        
        # Convert to parameter space
        log_s = np.log(np.maximum(best_support, 1e-10))
        log_w = np.log(best_weights)
        params_best = np.concatenate([log_s, log_w])
        
        # Run multiple local optimizations from different starting points
        n_local = 8
        for i in range(n_local):
            if time.time() > phase2_deadline - 10:
                break
            
            if i == 0:
                params_start = params_best.copy()
            else:
                params_start = params_best + np.random.normal(0, 0.3, size=2 * n_best)
            
            def obj_func(p):
                return _objective(p, n_best)
            
            try:
                res = minimize(
                    obj_func,
                    params_start,
                    method='Nelder-Mead',
                    options={'maxfev': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                
                final_params = res.x
                support, weights = _params_to_measure(final_params, n_best)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
                    
            except Exception:
                pass
        
        # Also try local opt with n+1 and n-1 if valid
        for n_try in [n_best - 1, n_best + 1]:
            if n_try < 1 or n_try > 12 or time.time() > phase2_deadline - 10:
                continue
            
            # Create initial params for this n
            if n_try < n_best:
                s_sub = best_support[:n_try]
                w_sub = best_weights[:n_try]
                w_sub = w_sub / np.sum(w_sub)
            else:
                s_sub = best_support.copy()
                w_sub = best_weights.copy()
                mid_idx = n_best // 2
                new_s = (s_sub[mid_idx - 1] + s_sub[mid_idx]) / 2 if mid_idx > 0 else s_sub[0] * 2
                new_w = w_sub[mid_idx] * 0.5
                w_sub[mid_idx] -= new_w
                s_sub = np.insert(s_sub, mid_idx, max(new_s, 0.0))
                w_sub = np.insert(w_sub, mid_idx, new_w)
                w_sub = np.maximum(w_sub, 1e-10)
                w_sub = w_sub / np.sum(w_sub)
            
            log_s = np.log(np.maximum(s_sub, 1e-10))
            log_w = np.log(np.maximum(w_sub, 1e-10))
            params_try = np.concatenate([log_s, log_w])
            
            def obj_func_try(p):
                return _objective(p, n_try)
            
            try:
                res = minimize(
                    obj_func_try,
                    params_try,
                    method='Nelder-Mead',
                    options={'maxfev': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                
                final_params = res.x
                support, weights = _params_to_measure(final_params, n_try)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
                    
            except Exception:
                pass
    
    # Phase 3: Differential evolution for global search (remaining time)
    if time.time() < deadline - 30:
        n_de_candidates = [4, 5, 6, 7, 8]
        if best_support is not None:
            n_best = len(best_support)
            if n_best not in n_de_candidates:
                n_de_candidates.append(n_best)
        
        for n_de in n_de_candidates:
            if time.time() > deadline - 30:
                break
            
            bounds = []
            for i in range(n_de):
                # Support: log-space from 1e-3 to 100
                bounds.append((np.log(1e-3), np.log(100.0)))
            for i in range(n_de):
                # Weights: log-space, -10 to 10
                bounds.append((-10.0, 10.0))
            
            def de_obj(p):
                return _objective(p, n_de)
                
            try:
                res_de = differential_evolution(
                    de_obj, 
                    bounds, 
                    maxiter=80, 
                    popsize=20,
                    seed=42,
                    tol=1e-10,
                    polish=True,
                    mutation=(0.5, 1.0),
                    recombination=0.8
                )
                final_params = res_de.x
                support, weights = _params_to_measure(final_params, n_de)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
            except Exception:
                pass
    
    # Final high-precision polish with multiple methods
    if best_support is not None and time.time() < deadline - 5:
        n_best = len(best_support)
        log_s = np.log(np.maximum(best_support, 1e-10))
        log_w = np.log(best_weights)
        params_best = np.concatenate([log_s, log_w])
        
        def obj_func(p):
            return _objective(p, n_best)
        
        # Try Nelder-Mead with very tight tolerance
        try:
            res = minimize(
                obj_func,
                params_best,
                method='Nelder-Mead',
                options={'maxfev': 15000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            
            final_params = res.x
            support, weights = _params_to_measure(final_params, n_best)
            
            if _validate(support, weights):
                prob = _calc_prob_fast(support, weights)
                try_update(support, weights, prob)
                
        except Exception:
            pass
        
        # Try COBYLA as well for potentially different local optimum
        if time.time() < deadline - 5:
            try:
                res2 = minimize(
                    obj_func,
                    params_best,
                    method='COBYLA',
                    options={'maxiter': 5000, 'rhobeg': 0.01}
                )
                
                final_params2 = res2.x
                support2, weights2 = _params_to_measure(final_params2, n_best)
                
                if _validate(support2, weights2):
                    prob2 = _calc_prob_fast(support2, weights2)
                    try_update(support2, weights2, prob2)
                    
            except Exception:
                pass
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        if not _validate(best_support, best_weights):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
            
        w_sum = np.sum(best_weights)
        if w_sum > 1e-9:
            best_weights = best_weights / w_sum
        else:
            best_weights = np.ones(len(best_support)) / len(best_support)
            
        best_support = np.maximum(best_support, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
