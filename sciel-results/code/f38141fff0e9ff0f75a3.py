import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized for speed.
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
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    
    # Weights outer product: W[i,j] = p_i * p_j
    W = p[:, None] * p[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    total_prob = 0.0
    
    # Pre-sort S_flat and corresponding W_flat for fast prefix sums
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    for k in range(n):
        threshold = 2.0 * support[k]
        # We need sum_l p_l * (sum_{i,j} p_i p_j I(s_i+s_j < threshold - s_l))
        T = threshold - support
        
        # Find indices in S_sorted where S_sorted < T
        indices = np.searchsorted(S_sorted, T, side='left')
        
        # Sum of weights for indices < T
        sums_for_l = W_cumsum_padded[indices]
        
        # prob3 = sum_l p_l * sums_for_l
        prob3 = np.dot(p, sums_for_l)
        
        total_prob += p[k] * prob3
        
    return total_prob

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

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using simulated annealing with dynamic support size
    and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin for 600s limit
    
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
    
    # Initialize with a reasonable starting point
    # Start with n=5, geometric-like support
    n_init = 5
    support_init = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    weights_init = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    prob_init = _calc_prob_fast(support_init, weights_init)
    try_update(support_init, weights_init, prob_init)
    
    # Simulated annealing with dynamic support size
    current_support = support_init.copy()
    current_weights = weights_init.copy()
    current_prob = prob_init
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    # Phase 1: Simulated annealing with dynamic n (first 40% of time)
    phase1_deadline = start_time + (deadline - start_time) * 0.4
    T = T_start
    
    while time.time() < phase1_deadline:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.5:
            # Perturb support and weights
            n = len(current_support)
            # Random perturbation to support (in log space)
            log_s = np.log(np.maximum(current_support, 1e-10))
            log_s += np.random.normal(0, 0.3, size=n)
            new_support = np.exp(log_s)
            new_support = np.maximum(new_support, 0.0)
            
            # Random perturbation to weights
            new_weights = current_weights * np.exp(np.random.normal(0, 0.2, size=n))
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        elif move_type < 0.75 and len(current_support) < 10:
            # Add a support point
            n = len(current_support)
            new_support = current_support.copy()
            new_weights = current_weights.copy()
            
            # Insert new point at random position
            insert_idx = random.randint(0, n)
            # New support value: interpolate or random
            if insert_idx == 0:
                new_val = current_support[0] * np.exp(np.random.normal(0, 0.5))
            elif insert_idx == n:
                new_val = current_support[-1] * np.exp(np.random.normal(0, 0.5))
            else:
                # Between existing points
                left = current_support[insert_idx - 1]
                right = current_support[insert_idx]
                new_val = np.exp(np.log(left) + np.random.uniform(0, 1) * (np.log(right) - np.log(left)))
            
            new_support = np.insert(new_support, insert_idx, max(new_val, 0.0))
            # Split weight from neighbors
            if insert_idx == 0:
                new_weights[0] *= 0.5
                new_weights = np.insert(new_weights, 0, new_weights[0])
            elif insert_idx == n:
                new_weights[-1] *= 0.5
                new_weights = np.insert(new_weights, -1, new_weights[-1])
            else:
                w_split = (new_weights[insert_idx - 1] + new_weights[insert_idx]) * 0.25
                new_weights[insert_idx - 1] -= w_split
                new_weights[insert_idx] -= w_split
                new_weights = np.insert(new_weights, insert_idx, w_split)
            
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        elif move_type < 0.9 and len(current_support) > 2:
            # Remove a support point
            n = len(current_support)
            remove_idx = random.randint(0, n - 1)
            
            new_support = np.delete(current_support, remove_idx)
            new_weights = np.delete(current_weights, remove_idx)
            
            # Redistribute weight to neighbors
            if len(new_weights) > 0:
                if remove_idx == 0 and len(new_weights) > 0:
                    new_weights[0] += current_weights[remove_idx]
                elif remove_idx == n - 1 and len(new_weights) > 0:
                    new_weights[-1] += current_weights[remove_idx]
                elif len(new_weights) > 0:
                    # Split to left and right
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
        n_local = 5
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
                    options={'maxfev': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                
                final_params = res.x
                support, weights = _params_to_measure(final_params, n_best)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
                    
            except Exception:
                pass
    
    # Phase 3: Differential evolution for global search (remaining time)
    if time.time() < deadline - 20:
        # Pick a promising n
        n_de = min(7, max(4, len(best_support) if best_support is not None else 5))
        
        bounds = []
        for i in range(n_de):
            bounds.append((np.log(1e-2), np.log(100.0)))  # log support
        for i in range(n_de):
            bounds.append((-10.0, 10.0))  # log weight raw
        
        def de_obj(p):
            return _objective(p, n_de)
            
        try:
            res_de = differential_evolution(
                de_obj, 
                bounds, 
                maxiter=100, 
                popsize=20,
                seed=42,
                tol=1e-8,
                polish=True
            )
            final_params = res_de.x
            support, weights = _params_to_measure(final_params, n_de)
            
            if _validate(support, weights):
                prob = _calc_prob_fast(support, weights)
                try_update(support, weights, prob)
        except Exception:
            pass
    
    # Final high-precision polish
    if best_support is not None and time.time() < deadline - 5:
        n_best = len(best_support)
        log_s = np.log(np.maximum(best_support, 1e-10))
        log_w = np.log(best_weights)
        params_best = np.concatenate([log_s, log_w])
        
        def obj_func(p):
            return _objective(p, n_best)
        
        try:
            res = minimize(
                obj_func,
                params_best,
                method='Nelder-Mead',
                options={'maxfev': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            final_params = res.x
            support, weights = _params_to_measure(final_params, n_best)
            
            if _validate(support, weights):
                prob = _calc_prob_fast(support, weights)
                try_update(support, weights, prob)
                
        except Exception:
            pass
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        # Ensure validity one last time
        if not _validate(best_support, best_weights):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
            
        # Normalize weights
        w_sum = np.sum(best_weights)
        if w_sum > 1e-9:
            best_weights = best_weights / w_sum
        else:
            best_weights = np.ones(len(best_support)) / len(best_support)
            
        # Ensure support >= 0
        best_support = np.maximum(best_support, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
