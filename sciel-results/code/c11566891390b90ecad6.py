# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    def time_left():
        return deadline - time.time()
    
    # Analytic 2-point optimum: support {0, 1}, weight p* on 1 where p* = (1+sqrt(17))/8
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # Try structured 3-point and 4-point configurations
    # These are fast to evaluate and may find better local optima
    
    # 3-point: {0, a, 1} with varying a
    for a in np.linspace(0.1, 0.9, 17):
        if time_left() < 30:
            break
        support = np.array([0.0, a, 1.0])
        # Try a few weight configurations
        for w1 in np.linspace(0.1, 0.8, 8):
            for w2 in np.linspace(0.1, 0.8, 8):
                w0 = 1.0 - w1 - w2
                if w0 < 0.01:
                    continue
                weights = np.array([w0, w1, w2])
                score = _evaluate_measure(support, weights)
                update_best(support, weights, score)
    
    # 4-point: {0, a, b, 1} with structured choices
    for a in [0.2, 0.3, 0.4, 0.5]:
        for b in [0.5, 0.6, 0.7, 0.8]:
            if a >= b:
                continue
            if time_left() < 20:
                break
            support = np.array([0.0, a, b, 1.0])
            # Try geometric and uniform weights
            for w1 in [0.15, 0.2, 0.25, 0.3]:
                for w2 in [0.15, 0.2, 0.25, 0.3]:
                    w0 = 0.3
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < 0.05:
                        continue
                    weights = np.array([w0, w1, w2, w3])
                    score = _evaluate_measure(support, weights)
                    update_best(support, weights, score)
    
    # Now do a more thorough search using Nelder-Mead on the best configuration
    if best_support is not None and time_left() > 120:
        n_best = len(best_support)
        
        def nm_objective(params):
            support = params[:n_best]
            log_w = params[n_best:]
            
            support = np.maximum(support, 0.0)
            support = np.sort(support)
            
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0
            p = w / w_sum
            
            try:
                P = _evaluate_measure(support, p)
            except Exception:
                return 1.0
            
            return -P
        
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        x_best = np.concatenate([best_support, log_w_best])
        
        from scipy.optimize import minimize
        result_final = minimize(
            nm_objective,
            x_best,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        x_final = result_final.x
        support_final = np.maximum(x_final[:n_best], 0.0)
        support_final = np.sort(support_final)
        log_w_final = x_final[n_best:]
        log_w_shifted_final = log_w_final - np.max(log_w_final)
        w_final = np.exp(log_w_shifted_final)
        w_sum_final = np.sum(w_final)
        if w_sum_final >= 1e-9:
            p_final = w_final / w_sum_final
            score_final = _evaluate_measure(support_final, p_final)
            update_best(support_final, p_final, score_final)
    
    # Try adding a 5th point to the best solution if time permits
    if best_support is not None and time_left() > 90 and len(best_support) == 4:
        n_best = len(best_support)
        
        # Add a point in the middle of the largest gap
        gaps = np.diff(best_support)
        max_gap_idx = np.argmax(gaps)
        new_point = best_support[max_gap_idx] + gaps[max_gap_idx] / 2.0
        support_5 = np.insert(best_support, max_gap_idx + 1, new_point)
        support_5 = np.sort(support_5)
        
        # Initialize weights: split the weight of the surrounding points
        w_new = np.zeros(5)
        w_new[0] = best_weights[0]
        w_new[max_gap_idx + 1] = best_weights[max_gap_idx] * 0.5
        w_new[max_gap_idx + 2] = best_weights[max_gap_idx + 1] * 0.5
        if max_gap_idx + 1 < len(best_weights):
            w_new[max_gap_idx + 2] += best_weights[max_gap_idx + 1] * 0.5
        # Adjust remaining weights
        for i in range(3, 5):
            if i - 1 < len(best_weights) and i - 1 != max_gap_idx + 1:
                w_new[i] = best_weights[min(i - 1, len(best_weights) - 1)]
        
        # Simpler: just distribute evenly with a small weight for the new point
        w_new = np.array([best_weights[0], best_weights[1] * 0.7, 0.1, best_weights[2] * 0.7, best_weights[3]])
        w_new = np.maximum(w_new, 0.01)
        w_new = w_new / np.sum(w_new)
        
        def nm_objective_5(params):
            support = params[:5]
            log_w = params[5:]
            
            support = np.maximum(support, 0.0)
            support = np.sort(support)
            
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0
            p = w / w_sum
            
            try:
                P = _evaluate_measure(support, p)
            except Exception:
                return 1.0
            
            return -P
        
        log_w_5 = np.log(np.maximum(w_new, 1e-15))
        x_5 = np.concatenate([support_5, log_w_5])
        
        from scipy.optimize import minimize
        result_5 = minimize(
            nm_objective_5,
            x_5,
            method='Nelder-Mead',
            options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        x_final_5 = result_5.x
        support_final_5 = np.maximum(x_final_5[:5], 0.0)
        support_final_5 = np.sort(support_final_5)
        log_w_final_5 = x_final_5[5:]
        log_w_shifted_final_5 = log_w_final_5 - np.max(log_w_final_5)
        w_final_5 = np.exp(log_w_shifted_final_5)
        w_sum_final_5 = np.sum(w_final_5)
        if w_sum_final_5 >= 1e-9:
            p_final_5 = w_final_5 / w_sum_final_5
            score_final_5 = _evaluate_measure(support_final_5, p_final_5)
            update_best(support_final_5, p_final_5, score_final_5)
    
    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity: no NaN, no inf, non-negative, positive sum
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=0.0, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=0.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    w_sum = np.sum(best_weights)
    if w_sum < 1e-9:
        best_support = support_two
        best_weights = weights_two
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
