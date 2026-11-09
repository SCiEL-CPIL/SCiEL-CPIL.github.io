# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 595.0  # Safety margin: 630s limit, leave 35s buffer

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
    
    # Vectorize over (k, l) pairs
    k_idx = np.repeat(np.arange(n), n)
    l_idx = np.tile(np.arange(n), n)
    thresh = 2.0 * support[l_idx] - support[k_idx]
    
    # Find number of pair sums strictly less than thresh for each (k,l)
    idxs = np.searchsorted(sorted_sums, thresh, side='left')
    inner_probs = np.where(idxs > 0, cum_probs[np.maximum(idxs - 1, 0)], 0.0)
    
    P = np.sum(w[k_idx] * w[l_idx] * inner_probs)
    
    return float(P)

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

def _geo_init(n, r, p):
    """Create geometric progression support with geometrically decaying weights."""
    s = np.array([r**i for i in range(n)])
    # Weights decay geometrically
    w_raw = np.array([p**i for i in range(n)])
    w = w_raw / w_raw.sum()
    return s, w

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
    
    # Try geometric progression initializations
    for r in [0.3, 0.5, 0.7, 0.8, 0.9]:
        for p in [0.3, 0.5, 0.7]:
            if _time_left() < 30:
                break
            for n in [4, 6, 8, 10, 12]:
                if _time_left() < 15:
                    break
                s, w = _geo_init(n, r, p)
                P = _compute_P_fast(s, w)
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
    
    # Try various n values with local optimization
    for n in [3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if _time_left() < 60:
            break
        
        # Number of restarts scales with remaining time and n
        max_restarts = max(5, int(200 * (1.0 / np.sqrt(n))))
        
        for restart in range(max_restarts):
            if _time_left() < 30:
                break
            
            # Multiple initialization strategies
            if restart % 3 == 0:
                # Exponential init
                np.random.seed(42 + restart * 1000 + n)
                s0 = np.sort(np.random.exponential(1.0, size=n))
                w0 = np.random.dirichlet(np.ones(n))
            elif restart % 3 == 1:
                # Geometric init
                r = np.random.uniform(0.3, 0.9, size=1)[0]
                p = np.random.uniform(0.3, 0.7, size=1)[0]
                s0, w0 = _geo_init(n, r, p)
            else:
                # Uniform init
                np.random.seed(42 + restart * 1000 + n + 100)
                s0 = np.sort(np.random.uniform(0, 2.0, size=n))
                w0 = np.random.dirichlet(np.ones(n) * 0.5)
            
            params0 = np.concatenate([s0, w0])
            
            # Use Powell (doesn't require gradients)
            try:
                res = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 150, 'maxfev': 800, 'xatol': 1e-8, 'fatol': 1e-8})
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
                                          maxiter=40, popsize=10, seed=42,
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
