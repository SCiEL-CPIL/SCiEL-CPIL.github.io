# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently.
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
    
    # Compute P = sum_{k,l} w_k w_l * CDF(2*s_l - s_k)
    # Thresholds: (n, n)
    thresholds = 2.0 * s_row - s_col
    
    # Find indices for CDF
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get CDF values
    cum_probs_with_zero = np.concatenate(([0.0], cum_probs))
    cdf_values = cum_probs_with_zero[indices]
    
    # Sum over k, l with weights
    w_k = w.reshape(-1, 1)
    w_l = w.reshape(1, -1)
    
    P = np.sum(w_k * w_l * cdf_values)
    
    return float(P)

def _optimize_weights_for_support(s, n_restarts=3):
    """
    Given a fixed support s, find optimal weights w by optimizing v in softmax(v).
    """
    n = len(s)
    if n == 0:
        return np.array([])
    
    def softmax(v):
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        return exp_v / np.sum(exp_v)
    
    def neg_P_v(v):
        w = softmax(v)
        return -_compute_P_fast(s, w)
    
    best_w = None
    best_neg_P = np.inf
    
    # Generate random initializations
    # 1. Uniform
    v0_list = [np.zeros(n)]
    # 2. Random
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        v0_list.append(rng.normal(0, 1, n))
    
    for v0 in v0_list:
        try:
            res = minimize(neg_P_v, v0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_neg_P:
                best_neg_P = res.fun
                best_w = softmax(res.x)
        except Exception:
            continue
            
    if best_w is None:
        return np.ones(n) / n
        
    return best_w

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game.
    Strategy: Global search on supports (DE), local optimization on weights (Nelder-Mead on softmax params).
    """
    t_start = time.time()
    deadline = t_start + 590.0 # Safety margin
    
    best_P_global = -1.0
    best_support_global = None
    best_weights_global = None
    
    # Fallback
    s_base = np.array([0.0, 1.0, 10.0])
    w_base = _optimize_weights_for_support(s_base)
    P_base = _compute_P_fast(s_base, w_base)
    best_P_global = P_base
    best_support_global = s_base.copy()
    best_weights_global = w_base.copy()
    
    n_candidates = [3, 4, 5, 6]
    
    for n in n_candidates:
        if time.time() > deadline - 60: # Leave 60s for larger n or final checks
            break
            
        def decode_supports(x):
            # x: (n,) raw values
            # Map to positive values via exp
            # To allow 0, we can't use pure exp. 
            # But s=0 is often optimal or near-optimal.
            # Let's use s = abs(exp(x)) or just exp(x) and hope optimization finds small x.
            # Actually, if s=0 is optimal, exp(x) -> 0 requires x -> -inf.
            # Let's use a mapping that allows 0.
            # s = np.exp(x) is strictly positive.
            # If the optimum is at s=0, the gradient will push x to -inf.
            # DE bounds: [-10, 10]. exp(-10) ~ 4.5e-5. Close to 0.
            # Let's try s = np.exp(x) * 10.0 to scale? No, just exp(x).
            s = np.exp(x)
            # Sort
            sort_idx = np.argsort(s)
            return s[sort_idx]
        
        def objective_supports(x):
            s = decode_supports(x)
            # Check for validity (should be fine with exp)
            w = _optimize_weights_for_support(s)
            P = _compute_P_fast(s, w)
            return -P
        
        bounds = [(-10, 10)] * n
        
        try:
            res = differential_evolution(
                objective_supports, 
                bounds, 
                maxiter=150, 
                popsize=15,
                tol=1e-5,
                seed=42,
                polish=True # Polish with L-BFGS-B on the support space
            )
            
            if res.fun < -best_P_global: # res.fun is -P
                s_opt = decode_supports(res.x)
                w_opt = _optimize_weights_for_support(s_opt)
                P_check = _compute_P_fast(s_opt, w_opt)
                
                if P_check > best_P_global:
                    best_P_global = P_check
                    best_support_global = s_opt.copy()
                    best_weights_global = w_opt.copy()
                    
        except Exception as e:
            print(f"DE failed for n={n}: {e}")
            continue
            
    # Final cleanup
    if best_support_global is None:
        return best_support_global, best_weights_global
        
    # Ensure non-negative
    best_support_global = np.maximum(best_support_global, 0.0)
    best_weights_global = np.maximum(best_weights_global, 0.0)
    
    # Normalize
    w_sum = np.sum(best_weights_global)
    if w_sum > 1e-9:
        best_weights_global = best_weights_global / w_sum
    else:
        best_weights_global = np.ones_like(best_weights_global) / len(best_weights_global)
        
    # Check finite
    if not (np.all(np.isfinite(best_support_global)) and np.all(np.isfinite(best_weights_global))):
        return best_support_global, best_weights_global
        
    return best_support_global, best_weights_global

# EVOLVE-BLOCK-END
