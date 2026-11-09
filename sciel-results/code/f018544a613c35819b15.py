# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

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
        
    # 1. Analytic 2-point solution
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # 2. Strategy: Search over 3-point support {0, 1, a}
    # We fix the first two points to 0 and 1 (scale invariance allows setting one point to 1, 
    # and 0 is almost always optimal as the "low" value).
    # We vary 'a' and optimize the weights (w0, w1, w2) for each 'a'.
    
    def optimize_weights_for_a(a):
        """
        For a fixed support [0, 1, a], find optimal weights.
        We use a 2D search over (w0, w1) since w2 = 1 - w0 - w1.
        """
        if a <= 1.0:
            a = 1.0 + 1e-9 # Ensure distinct points, though optimizer might handle it
            
        support = np.array([0.0, 1.0, a])
        
        def neg_P(weights_2d):
            w0, w1 = weights_2d
            # Constraints: w0 >= 0, w1 >= 0, w0 + w1 <= 1
            if w0 < 0 or w1 < 0 or w0 + w1 > 1:
                return 1.0 # Penalty
            w2 = 1.0 - w0 - w1
            weights = np.array([w0, w1, w2])
            try:
                return -_evaluate_measure(support, weights)
            except Exception:
                return 1.0
        
        # Initial guess: uniform or near the 2-point solution
        # If a is large, w2 should be small. If a is close to 1, w2 might be significant.
        # Let's do a coarse grid search first to find a good starting point for local optimization.
        best_val = 1.0
        best_w = (0.3, 0.3)
        
        # Grid search
        for w0 in np.linspace(0.1, 0.9, 10):
            for w1 in np.linspace(0.1, 0.9 - w0, 10):
                if w0 + w1 > 1.0:
                    continue
                val = neg_P((w0, w1))
                if val < best_val:
                    best_val = val
                    best_w = (w0, w1)
        
        # Local polish with Nelder-Mead
        try:
            res = minimize(
                neg_P, 
                best_w, 
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_val:
                best_w = res.x
                best_val = res.fun
        except Exception:
            pass
        
        w0, w1 = best_w
        w2 = 1.0 - w0 - w1
        weights = np.array([w0, w1, w2])
        
        # Clean up small negatives from numerical error
        weights = np.maximum(weights, 0.0)
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            return support, np.array([1/3, 1/3, 1/3]), -1.0
        weights = weights / w_sum
        
        score = _evaluate_measure(support, weights)
        return support, weights, score

    # Search over 'a'
    # We expect 'a' to be somewhat larger than 1 to provide a "large" value that rarely triggers the inequality 
    # but allows the "medium" value (1) to beat the "small" value (0) in the sum comparison.
    # Actually, let's think about the inequality: X1+X2+X3 < 2*X4.
    # If X4 is large (a), RHS is 2a. LHS is sum of three.
    # If X4 is small (0), RHS is 0. LHS must be < 0. Impossible since support >= 0.
    # So if X4=0, probability is 0.
    # If X4=1, RHS=2. LHS < 2.
    # If X4=a, RHS=2a. LHS < 2a.
    
    # Let's scan a from 1.01 to 5.0
    a_grid = np.linspace(1.01, 5.0, 50)
    
    for a in a_grid:
        if time_left() < 10:
            break
        support, weights, score = optimize_weights_for_a(a)
        update_best(support, weights, score)
        
    # Refine around the best 'a' found
    if best_support is not None and len(best_support) == 3:
        best_a = best_support[2]
        # Local search on 'a'
        def score_for_a(a):
            if a <= 1.0:
                return -1.0
            support, weights, score = optimize_weights_for_a(a)
            return score
        
        # Use minimize_scalar to find the peak in a
        try:
            res_a = minimize_scalar(
                lambda x: -score_for_a(x), 
                bounds=(max(1.01, best_a - 0.5), min(10.0, best_a + 0.5)), 
                method='bounded',
                options={'xatol': 1e-4}
            )
            if res_a.success:
                a_opt = res_a.x
                support, weights, score = optimize_weights_for_a(a_opt)
                update_best(support, weights, score)
        except Exception:
            pass

    # Try a 4-point structure {0, 1, a, b} briefly if time permits?
    # Given the 600s limit and that 3-point optimization is fast, we might try a few specific 4-point configs.
    # But 3-point is likely sufficient for a jump from 1.36 if the weights are optimized perfectly.
    # The current DE approach likely didn't converge to the exact weight optimum for any specific support structure.
    
    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity
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
