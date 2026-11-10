# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _compute_score_fast(support, weights):
    """Fast computation using sorting and vectorization."""
    n = len(support)
    if n == 0:
        return 0.0
    w = weights.astype(np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    w = w / total
    s = support.astype(np.float64)
    
    # Compute all pairwise sums
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # For each (k, l), threshold = 2*s_l - s_k
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # Use searchsorted
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k * w_l * probs[k,l]
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580.0
    
    best_score = 0.0
    best_support = None
    best_weights = None
    
    def update_best(s, w, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = s.copy()
            best_weights = w.copy()
    
    # Phase 1: Evaluate diverse structured candidates
    candidates = []
    
    # Two-point distributions
    for M in [1, 2, 5, 10, 50]:
        for p in np.linspace(0.3, 0.8, 51):
            candidates.append((np.array([0.0, M]), np.array([p, 1-p])))
    
    # Three-point: 0, a, b
    for a in [0.5, 1, 2, 5]:
        for b in [2, 5, 10, 20, 50]:
            if b <= a:
                continue
            for p1 in np.linspace(0.2, 0.7, 11):
                for p2 in np.linspace(0.1, 0.6, 11):
                    p3 = 1 - p1 - p2
                    if p3 < 0.05:
                        continue
                    candidates.append((np.array([0.0, a, b]), np.array([p1, p2, p3])))
    
    # Four-point: 0, a, b, c
    for a in [0.5, 1, 2]:
        for b in [2, 5, 10]:
            for c in [10, 20, 50]:
                if not (a < b < c):
                    continue
                for p1 in np.linspace(0.2, 0.5, 6):
                    for p2 in np.linspace(0.1, 0.4, 6):
                        for p3 in np.linspace(0.1, 0.3, 6):
                            p4 = 1 - p1 - p2 - p3
                            if p4 < 0.05:
                                continue
                            candidates.append((np.array([0.0, a, b, c]), np.array([p1, p2, p3, p4])))
    
    # Geometric distributions
    for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for n_pts in [5, 8, 12, 16, 24]:
            support = np.arange(n_pts, dtype=np.float64)
            weights = r ** support
            weights /= weights.sum()
            candidates.append((support, weights))
    
    # Evaluate candidates
    for s, w in candidates:
        if time.time() > deadline - 120.0:
            break
        try:
            score = _compute_score_fast(s, w)
            update_best(s, w, score)
        except Exception:
            continue
    
    # Phase 2: Local optimization
    if time.time() < deadline - 60.0 and best_support is not None:
        n = len(best_support)
        
        # Optimize weights only
        def neg_score_weights(w_params):
            w = np.abs(w_params)
            total = w.sum()
            if total < 1e-9:
                return 0.0
            w = w / total
            return -_compute_score_fast(best_support, w)
        
        w0 = best_weights.copy()
        try:
            res = minimize(neg_score_weights, w0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() < deadline - 30.0:
                w_opt = np.abs(res.x)
                w_opt = w_opt / w_opt.sum()
                score = _compute_score_fast(best_support, w_opt)
                update_best(best_support, w_opt, score)
        except Exception:
            pass
        
        # Optimize support only
        def neg_score_support(s_params):
            s = np.maximum(s_params, 0.0)
            return -_compute_score_fast(s, best_weights)
        
        s0 = best_support.copy()
        try:
            res = minimize(neg_score_support, s0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() < deadline - 30.0:
                s_opt = np.maximum(res.x, 0.0)
                score = _compute_score_fast(s_opt, best_weights)
                update_best(s_opt, best_weights, score)
        except Exception:
            pass
    
    # Phase 3: Try different numbers of points
    if time.time() < deadline - 30.0:
        for n_pts in [3, 4, 5, 6]:
            if time.time() > deadline - 30.0:
                break
            
            s_init = np.linspace(0, 10, n_pts)
            w_init = np.ones(n_pts) / n_pts
            
            def neg_score_joint(params):
                s = np.maximum(params[:n_pts], 0.0)
                w = np.abs(params[n_pts:])
                total = w.sum()
                if total < 1e-9:
                    return 0.0
                w = w / total
                return -_compute_score_fast(s, w)
            
            params0 = np.concatenate([s_init, w_init])
            try:
                res = minimize(neg_score_joint, params0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
                s_opt = np.maximum(res.x[:n_pts], 0.0)
                w_opt = np.abs(res.x[n_pts:])
                w_opt = w_opt / w_opt.sum()
                score = _compute_score_fast(s_opt, w_opt)
                update_best(s_opt, w_opt, score)
            except Exception:
                continue
    
    # Final validation
    if best_support is None or len(best_support) < 1:
        p = (15 - np.sqrt(33)) / 16
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([p, 1-p])
    
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    else:
        best_weights = best_weights / total_w
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
