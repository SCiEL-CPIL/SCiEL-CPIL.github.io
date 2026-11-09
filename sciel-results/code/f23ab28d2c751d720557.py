# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_start_time = time.time()
_DEADLINE = 595.0

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _compute_P_fast(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] efficiently using sorted pair sums.
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
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    if w.sum() < 1e-9:
        return 1e10
    P = _compute_P_fast(s, w)
    return -P

def _geo_init(n, r, p):
    """Create geometric progression support with geometrically decaying weights."""
    s = np.array([r**i for i in range(n)])
    w_raw = np.array([p**i for i in range(n)])
    w = w_raw / w_raw.sum()
    return s, w

def _two_point_P(p):
    """
    Exact P for 2-point measure {0 with prob p, 1 with prob 1-p}.
    P[X1+X2+X3 < 2*X4]
    
    X4=1 (prob 1-p): need X1+X2+X3 < 2, always true (max sum=3, but <2 means sum<=1)
      Actually X1+X2+X3 can be 0,1,2,3. Need <2, so sum=0 or sum=1.
      P(sum=0) = p^3
      P(sum=1) = 3*p^2*(1-p)
      So P(X1+X2+X3<2 | X4=1) = p^3 + 3*p^2*(1-p) = p^2*(p + 3 - 3p) = p^2*(3-2p)
    
    X4=0 (prob p): need X1+X2+X3 < 0, impossible.
    
    So P = (1-p) * p^2 * (3-2p)
    """
    return (1.0 - p) * p**2 * (3.0 - 2.0 * p)

def _three_point_P(p0, p1, p2, s0, s1, s2):
    """Compute P for a 3-point measure."""
    support = np.array([s0, s1, s2])
    weights = np.array([p0, p1, p2])
    return _compute_P_fast(support, weights)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    best_support = None
    best_weights = None
    best_P = -1.0
    
    # Start with the 2-point analytical optimum
    # Maximize (1-p)*p^2*(3-2p)
    # f(p) = (1-p)*p^2*(3-2p) = p^2*(1-p)*(3-2p)
    # f'(p) = 2p(1-p)(3-2p) + p^2*(-1)(3-2p) + p^2(1-p)(-2)
    #       = p[2(1-p)(3-2p) - p(3-2p) - 2p(1-p)]
    #       = p[(1-p)(3-2p)*2 - p(3-2p) - 2p(1-p)]
    #       = p[(1-p)(3-2p)(2-1) ... let me just solve numerically
    # f(p) = p^2 - p^3)(3-2p) = 3p^2 - 2p^3 - 3p^3 + 2p^4 = 3p^2 - 5p^3 + 2p^4
    # f'(p) = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    # 8p^2 - 15p + 6 = 0
    # p = (15 ± sqrt(225 - 192))/16 = (15 ± sqrt(33))/16
    # sqrt(33) ≈ 5.7446
    # p = (15 - 5.7446)/16 ≈ 0.5785 or p = (15+5.7446)/16 ≈ 1.30 (invalid)
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    P2 = _compute_P_fast(s2, w2)
    best_P = P2
    best_support = s2.copy()
    best_weights = w2.copy()
    
    # Try geometric progression initializations with 2-param search
    for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if _time_left() < 30:
            break
        for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            if _time_left() < 15:
                break
            for n in [3, 4, 5, 6, 8, 10, 12, 16]:
                if _time_left() < 10:
                    break
                s, w = _geo_init(n, r, p)
                P = _compute_P_fast(s, w)
                if P > best_P:
                    best_P = P
                    best_support = s.copy()
                    best_weights = w.copy()
    
    # 2-parameter geometric search: optimize r and alpha for fixed n
    for n in [3, 4, 5, 6, 8, 10]:
        if _time_left() < 60:
            break
        
        def _geo_obj(params, n=n):
            r, alpha = params
            r = np.clip(r, 0.01, 0.99)
            alpha = np.clip(alpha, 0.01, 0.99)
            s, w = _geo_init(n, r, alpha)
            P = _compute_P_fast(s, w)
            return -P
        
        # Try multiple starting points
        for r0 in [0.3, 0.5, 0.7, 0.85]:
            for a0 in [0.3, 0.5, 0.7, 0.85]:
                if _time_left() < 20:
                    break
                try:
                    res = minimize(_geo_obj, [r0, a0], method='Nelder-Mead',
                                  options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res.fun < -best_P:
                        r_opt, a_opt = res.x
                        r_opt = np.clip(r_opt, 0.01, 0.99)
                        a_opt = np.clip(a_opt, 0.01, 0.99)
                        s, w = _geo_init(n, r_opt, a_opt)
                        P = _compute_P_fast(s, w)
                        if P > best_P:
                            best_P = P
                            best_support = s.copy()
                            best_weights = w.copy()
                except Exception:
                    pass
    
    # Try various n values with local optimization
    for n in [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]:
        if _time_left() < 60:
            break
        
        max_restarts = max(5, int(150 * (1.0 / np.sqrt(n))))
        
        for restart in range(max_restarts):
            if _time_left() < 30:
                break
            
            if restart % 4 == 0:
                np.random.seed(42 + restart * 1000 + n)
                s0 = np.sort(np.random.exponential(1.0, size=n))
                w0 = np.random.dirichlet(np.ones(n))
            elif restart % 4 == 1:
                r = np.random.uniform(0.3, 0.9, size=1)[0]
                p = np.random.uniform(0.3, 0.7, size=1)[0]
                s0, w0 = _geo_init(n, r, p)
            elif restart % 4 == 2:
                np.random.seed(42 + restart * 1000 + n + 100)
                s0 = np.sort(np.random.uniform(0, 2.0, size=n))
                w0 = np.random.dirichlet(np.ones(n) * 0.5)
            else:
                # Concentrated at 0 and a large value
                np.random.seed(42 + restart * 1000 + n + 200)
                large_val = np.random.uniform(1.0, 5.0)
                s0 = np.zeros(n)
                s0[-1] = large_val
                s0[-2] = large_val * np.random.uniform(0.3, 0.7)
                w0 = np.zeros(n)
                w0[0] = np.random.uniform(0.3, 0.7)
                w0[-1] = 1.0 - w0[0]
                w0 = w0 / w0.sum()
            
            params0 = np.concatenate([s0, w0])
            
            try:
                res = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 100, 'maxfev': 500, 'xatol': 1e-8, 'fatol': 1e-8})
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
    
    # Differential evolution for small/medium n
    for n in [4, 6, 8, 10]:
        if _time_left() < 120:
            break
        
        try:
            bounds = [(0, 5.0)] * n + [(1e-6, 1.0)] * n
            res = differential_evolution(_objective, bounds, args=(n,),
                                          maxiter=30, popsize=8, seed=42,
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
    
    # Final high-precision polish
    if best_support is not None and _time_left() > 40:
        n = len(best_support)
        params0 = np.concatenate([best_support, best_weights])
        try:
            res = minimize(_objective, params0, args=(n,), method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
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
        
        # Second polish with Powell
        if _time_left() > 40:
            try:
                res = minimize(_objective, params0, args=(n,), method='Powell',
                              options={'maxiter': 200, 'maxfev': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
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
