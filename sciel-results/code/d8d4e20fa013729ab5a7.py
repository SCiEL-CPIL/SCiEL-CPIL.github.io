# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 590.0  # Safety margin: 630s limit, leave 40s buffer

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using sorted pair sums.
    P = sum_{k,l} w_k w_l * Prob[X1+X2 < 2*s_l - s_k]
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights just in case, though they should be normalized
    total_w = np.sum(weights)
    if total_w < 1e-9:
        return 0.0
    w = weights / total_w
    
    # Compute all pairwise sums s_i + s_j and their corresponding probability weights w_i * w_j
    pair_sums = support[:, None] + support[None, :]
    pair_probs = w[:, None] * w[None, :]
    
    # Flatten and sort by sum
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    P = 0.0
    # Iterate over all (k, l) pairs
    # We need sum_{i,j} w_i w_j [s_i + s_j < 2 s_l - s_k]
    for k in range(n):
        for l in range(n):
            thresh = 2.0 * support[l] - support[k]
            # Find number of pair sums strictly less than thresh
            idx = np.searchsorted(sorted_sums, thresh, side='left')
            if idx > 0:
                inner_prob = cum_probs[idx - 1]
            else:
                inner_prob = 0.0
            P += w[k] * w[l] * inner_prob
            
    return P

def _objective(params, n):
    """Objective to minimize: -P. params = [s_0,...,s_{n-1}, w_0,...,w_{n-1}]"""
    s = params[:n]
    w = params[n:]
    # Enforce s >= 0, w >= 0
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-9:
        return 1e10
    P = _compute_P_fast(s, w)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Start with the 2-point analytical optimum
    p_opt = (15 - np.sqrt(33)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    P2 = _compute_P_fast(s2, w2)
    best_P = P2
    best_support = s2.copy()
    best_weights = w2.copy()
    
    # Try various n values with local optimization
    # We can now afford larger n and more restarts due to fast evaluation
    for n in [3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if _time_left() < 60:
            break
        
        # Number of restarts scales with remaining time and n
        # Larger n is slower per eval, so fewer restarts
        max_restarts = max(10, int(300 * (1.0 / np.sqrt(n))))
        
        for restart in range(max_restarts):
            if _time_left() < 30:
                break
            
            # Random initial point
            # Support points in [0, 10] roughly, weights Dirichlet
            np.random.seed(42 + restart * 1000 + n)
            s0 = np.sort(np.random.exponential(1.0, size=n))
            w0 = np.random.dirichlet(np.ones(n))
            params0 = np.concatenate([s0, w0])
            
            # Use Powell (doesn't require gradients)
            try:
                res = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 200, 'maxfev': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < -best_P:
                    params = res.x
                    s = np.maximum(params[:n], 0.0)
                    w = np.maximum(params[n:], 0.0)
                    if w.sum() > 1e-9:
                        P = _compute_P_fast(s, w)
                        if P > best_P:
                            best_P = P
                            best_support = s.copy()
                            best_weights = w.copy()
            except Exception:
                pass
    
    # Try differential evolution for small/medium n if time permits
    for n in [4, 6, 8]:
        if _time_left() < 120:
            break
        
        try:
            bounds = [(0, 5.0)] * n + [(1e-6, 1.0)] * n
            res = differential_evolution(_objective, bounds, args=(n,),
                                          maxiter=50, popsize=10, seed=42,
                                          tol=1e-6, polish=True)
            if res.fun < -best_P:
                params = res.x
                s = np.maximum(params[:n], 0.0)
                w = np.maximum(params[n:], 0.0)
                if w.sum() > 1e-9:
                    P = _compute_P_fast(s, w)
                    if P > best_P:
                        best_P = P
                        best_support = s.copy()
                        best_weights = w.copy()
        except Exception:
            pass
    
    # Final polish on best result
    if best_support is not None and _time_left() > 30:
        n = len(best_support)
        params0 = np.concatenate([best_support, best_weights])
        try:
            res = minimize(_objective, params0, args=(n,), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < -best_P:
                params = res.x
                s = np.maximum(params[:n], 0.0)
                w = np.maximum(params[n:], 0.0)
                if w.sum() > 1e-9:
                    P = _compute_P_fast(s, w)
                    if P > best_P:
                        best_P = P
                        best_support = s.copy()
                        best_weights = w.copy()
        except Exception:
            pass
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p_opt, 1.0 - p_opt])
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
