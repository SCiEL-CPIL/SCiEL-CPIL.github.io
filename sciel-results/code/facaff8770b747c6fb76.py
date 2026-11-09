# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] efficiently."""
    n = len(support)
    if n == 0:
        return 0.0
    s = support.astype(np.float64)
    w = weights.astype(np.float64)
    total_w = w.sum()
    if total_w <= 1e-9:
        return 0.0
    p = w / total_w
    
    # Pairwise sums and probabilities
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]
    
    ss_flat = ss.ravel()
    pp_flat = pp.ravel()
    
    order = np.argsort(ss_flat, kind='mergesort')
    ss_sorted = ss_flat[order]
    pp_sorted = pp_flat[order]
    pp_cumsum = np.cumsum(pp_sorted)
    
    # Vectorize the sum over k, l
    # Thresholds T[k, l] = 2*s_l - s_k
    # We need sum_{i,j} pp[i,j] where ss[i,j] < T[k,l]
    # This is equivalent to: for each (k,l), find index in ss_sorted and get cumsum value.
    
    # Create matrix of thresholds
    # T[k, l] = 2*s_l - s_k
    # s_k is shape (n, 1), 2*s_l is shape (1, n)
    thresholds = 2.0 * s[None, :] - s[:, None] # Shape (n, n), rows are k, cols are l
    
    # Searchsorted on 2D array against 1D sorted array
    # Returns indices (n, n)
    idxs = np.searchsorted(ss_sorted, thresholds, side='left')
    
    # We need pp_cumsum[idx-1] for idx>0, else 0.
    # Create a safe index array
    safe_idxs = np.where(idxs > 0, idxs - 1, 0)
    
    # Get the cumulative probabilities
    pair_probs = pp_cumsum[safe_idxs]
    # Zero out where idx was 0
    pair_probs = np.where(idxs > 0, pair_probs, 0.0)
    
    # P = sum_{k,l} p_k * p_l * pair_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * pair_probs)
    
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580
    
    best_p = 0.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts):
        nonlocal best_p, best_support, best_weights
        if len(sup) != len(wts) or len(sup) == 0:
            return
        if np.any(np.isnan(sup)) or np.any(np.isinf(sup)):
            return
        if np.any(np.isnan(wts)) or np.any(np.isinf(wts)):
            return
        if np.any(sup < -1e-9):
            return
        if np.any(wts < -1e-9):
            return
        if wts.sum() <= 1e-9:
            return
        
        # Clean up for computation
        sup_clean = np.maximum(sup, 0.0)
        wts_clean = np.maximum(wts, 0.0)
        tw = wts_clean.sum()
        if tw <= 1e-9:
            return
        wts_norm = wts_clean / tw
        
        p_val = _compute_P(sup_clean, wts_norm)
        if p_val > best_p:
            best_p = p_val
            best_support = sup_clean.copy()
            best_weights = wts_norm.copy()

    # Baseline
    n_base = 10
    sup_base = np.linspace(0, 10, n_base)
    w_base = np.ones(n_base) / n_base
    update_best(sup_base, w_base)
    
    # Differential Evolution Strategy
    # We optimize over support points (n) and weights (n).
    # To handle the simplex constraint on weights and non-negativity on support,
    # we map parameters:
    # s_i = max(0, param_i)
    # w_i = exp(param_{n+i}) / sum(exp(params))
    
    def objective(params):
        n = len(params) // 2
        s_raw = params[:n]
        w_raw = params[n:]
        
        s = np.maximum(s_raw, 0.0)
        
        # Softmax for weights to ensure positive and sum to 1
        # Subtract max for numerical stability
        w_max = np.max(w_raw)
        exp_w = np.exp(w_raw - w_max)
        w = exp_w / np.sum(exp_w)
        
        # Check for time limit
        if time.time() > deadline - 10:
            return 0.0
            
        p_val = _compute_P(s, w)
        return -p_val

    # Try different n
    # DE is expensive, so we start with smaller n and larger bounds
    # Bounds for s: [0, 100]
    # Bounds for w_raw (log-probs): [-10, 10]
    
    n_candidates = [4, 6, 8]
    
    for n in n_candidates:
        if time.time() > deadline - 60:
            break
            
        bounds = [(0, 100)] * n + [(-10, 10)] * n
        
        # Use DE
        try:
            # maxiter is limited to ensure we don't run out of time
            # popsize is 15 by default, which is 15*n evaluations per generation
            # We limit maxiter to ~20-50 depending on n
            maxiter = 50 if n <= 6 else 20
            
            res = differential_evolution(
                objective, bounds, 
                maxiter=maxiter, 
                popsize=10, 
                tol=1e-4, 
                seed=42,
                polish=True, # Use Nelder-Mead at the end
                init='latinhypercube'
            )
            
            if res.success:
                s_opt = np.maximum(res.x[:n], 0.0)
                w_raw = res.x[n:]
                w_max = np.max(w_raw)
                exp_w = np.exp(w_raw - w_max)
                w_opt = exp_w / np.sum(exp_w)
                
                update_best(s_opt, w_opt)
                
        except Exception as e:
            pass

    # If time remains, do local refinement on the best found so far
    if best_support is not None and time.time() < deadline - 30:
        n = len(best_support)
        s0 = best_support
        w0 = best_weights
        
        # Map back to raw parameters for Nelder-Mead
        # s_raw = s0
        # w_raw = log(w0) (approx, or just use w0 directly with constraints)
        # It's easier to just run Nelder-Mead on (s, w) with projection
        
        def nm_objective(params):
            n = len(params) // 2
            s = params[:n]
            w = params[n:]
            s = np.maximum(s, 0.0)
            w = np.maximum(w, 0.0)
            tw = w.sum()
            if tw <= 1e-9:
                return 0.0
            w_norm = w / tw
            return -_compute_P(s, w_norm)
        
        x0 = np.concatenate([s0, w0])
        
        try:
            res_nm = minimize(
                nm_objective, x0, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < -best_p + 1e-8: # Improved
                s_opt = np.maximum(res_nm.x[:n], 0.0)
                w_opt = np.maximum(res_nm.x[n:], 0.0)
                update_best(s_opt, w_opt)
        except Exception:
            pass

    if best_support is not None and best_weights is not None:
        return best_support.astype(np.float64), best_weights.astype(np.float64)
    
    return sup_base, w_base

# EVOLVE-BLOCK-END
