# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Generate all pair sums and their probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # (n, n)
    pair_probs = np.outer(w, w) # (n, n)
    
    # Flatten
    sums_flat = pair_sums.flatten() # (n^2,)
    probs_flat = pair_probs.flatten() # (n^2,)
    
    # Sort sums and compute cumulative probabilities
    sort_idx = np.argsort(sums_flat)
    sorted_sums = sums_flat[sort_idx]
    sorted_probs = probs_flat[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    # Ensure cum_probs ends at 1.0 (floating point safety)
    if cum_probs[-1] < 1.0:
        if np.isclose(cum_probs[-1], 1.0, atol=1e-9):
             cum_probs[-1] = 1.0
    
    # Compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    thresholds = 2.0 * s_row - s_col  # (n, n)
    
    # For each threshold, find the count of pair_sums < threshold
    # searchsorted with side='left' gives the index of the first element >= threshold.
    # We want sum of probs for elements < threshold.
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights w_k w_l
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def _objective(params, n):
    """
    Objective function for Nelder-Mead.
    params: [log_s_1, ..., log_s_n, logit_w_1, ..., logit_w_n]
    We parameterize support in log-space to ensure positivity and smoothness.
    We parameterize weights in logit-space to ensure positivity and sum-to-1 (via softmax).
    Note: We use softmax for weights to handle the constraint naturally.
    """
    # Unpack
    log_s = params[:n]
    log_w = params[n:]
    
    # Decode support
    # Ensure support is non-negative. 
    # log_s can be any real number, so s = exp(log_s) is always > 0.
    # However, we might want mass at 0. 
    # Strategy: Allow s_1 to be close to 0 by letting log_s_1 go to -inf? 
    # Better: Use a softplus or just exp, but exp(log_s) never hits 0.
    # To allow mass at 0, we can treat the first point specially or just let it be small.
    # Given the "heavy tail at 0" insight, having s[0] very small is good.
    s = np.exp(log_s)
    
    # Decode weights using softmax
    # Subtract max for numerical stability
    log_w_stable = log_w - np.max(log_w)
    w_unnorm = np.exp(log_w_stable)
    w_sum = np.sum(w_unnorm)
    if w_sum < 1e-15:
        return -1.0 # Invalid
    w = w_unnorm / w_sum
    
    # Ensure valid support (non-negative)
    if np.any(s < 0):
        return -1.0
    
    # Compute P
    P = _compute_P_fast(s, w)
    
    # We want to maximize P, so minimize -P
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using Multi-Start Nelder-Mead Optimization.
    """
    t_start = time.time()
    deadline = t_start + 580.0 # Safety margin
    
    n = 6 # Fixed number of atoms. 6 is a good balance between expressiveness and optimization speed.
    
    best_params = None
    best_score = -1.0
    best_s = None
    best_w = None
    
    # Initial guesses
    # We want a distribution with significant mass at 0 (or near 0) and a tail.
    # Let's try a few different initial structures.
    
    inits = []
    
    # Init 1: Geometric-like in log space
    # s: 0.01, 0.1, 1, 10, 100, 1000 (log10: -2, -1, 0, 1, 2, 3)
    # w: heavy at 0, decaying.
    # logit(w) for w=[0.5, 0.25, 0.1, 0.05, 0.05, 0.05] -> log(w/(1-w))
    def make_init(s_vals, w_vals):
        log_s = np.log(np.maximum(s_vals, 1e-6))
        # For weights, use log(w) directly in the parameter vector, 
        # but the objective uses softmax. 
        # If we want initial weights w_init, we can set log_w_init = log(w_init).
        # Softmax(log(w_init)) will normalize them.
        log_w = np.log(np.maximum(w_vals, 1e-6))
        return np.concatenate([log_s, log_w])
    
    # Candidate 1: Standard geometric
    s1 = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    w1 = np.array([0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
    inits.append(make_init(s1, w1))
    
    # Candidate 2: More mass at 0
    s2 = np.array([1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    w2 = np.array([0.8, 0.1, 0.05, 0.03, 0.01, 0.01])
    inits.append(make_init(s2, w2))
    
    # Candidate 3: Wider tail
    s3 = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 1000.0])
    w3 = np.array([0.5, 0.25, 0.15, 0.05, 0.03, 0.02])
    inits.append(make_init(s3, w3))
    
    # Candidate 4: Randomized perturbations of Init 1
    for _ in range(3):
        s_rand = s1 * np.exp(np.random.normal(0, 0.5, 6))
        w_rand = w1 * np.exp(np.random.normal(0, 0.5, 6))
        inits.append(make_init(s_rand, w_rand))
        
    # Candidate 5: Uniform-like but shifted
    s5 = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    w5 = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    inits.append(make_init(s5, w5))

    # Optimization Settings
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 60: # Leave time for finalization
            break
            
        try:
            res = minimize(_objective, x0, args=(n,), method='Nelder-Mead', options=options)
            
            if res.success or res.fun < best_score:
                # Decode best params
                log_s = res.x[:n]
                log_w = res.x[n:]
                
                s = np.exp(log_s)
                log_w_stable = log_w - np.max(log_w)
                w_unnorm = np.exp(log_w_stable)
                w_sum = np.sum(w_unnorm)
                w = w_unnorm / w_sum
                
                score = -res.fun
                
                if score > best_score:
                    best_score = score
                    best_s = s
                    best_w = w
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass
    
    # If optimization didn't find anything good, use a fallback
    if best_s is None:
        # Fallback to a known good configuration
        best_s = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
        best_w = np.array([0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
        best_w = best_w / np.sum(best_w)
        
    # Final Validation and Normalization
    # Ensure support is non-negative and finite
    best_s = np.maximum(best_s, 0.0)
    best_s = best_s[np.isfinite(best_s)]
    best_w = best_w[np.isfinite(best_w)]
    
    # If we lost some elements due to NaN/Inf, we need to handle it
    if len(best_s) != n:
        # Reconstruct from best_params if possible, or fallback
        if best_params is not None:
            log_s = best_params[:n]
            log_w = best_params[n:]
            best_s = np.exp(log_s)
            log_w_stable = log_w - np.max(log_w)
            w_unnorm = np.exp(log_w_stable)
            w_sum = np.sum(w_unnorm)
            best_w = w_unnorm / w_sum
        else:
            # Hard fallback
            best_s = np.array([0.0, 1.0])
            best_w = np.array([0.618, 0.382])
            return best_s, best_w

    # Ensure lengths match
    if len(best_s) != len(best_w):
         # Pad or truncate? Just use the first min(len) if they mismatch due to weird errors
         min_len = min(len(best_s), len(best_w))
         best_s = best_s[:min_len]
         best_w = best_w[:min_len]
         
    # Ensure weights sum to 1 and are positive
    best_w = np.maximum(best_w, 1e-9)
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
    best_w = best_w / w_sum
    
    # Sort by support
    sort_idx = np.argsort(best_s)
    best_s = best_s[sort_idx]
    best_w = best_w[sort_idx]
    
    # Final check for validity
    if not (np.all(np.isfinite(best_s)) and np.all(np.isfinite(best_w))):
        return np.array([0.0, 1.0]), np.array([0.618, 0.382])
        
    return best_s, best_w

# EVOLVE-BLOCK-END
