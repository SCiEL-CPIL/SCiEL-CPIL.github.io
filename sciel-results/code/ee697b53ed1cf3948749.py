# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure with given support and weights.
    Uses O(n^2 log n) vectorized approach.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w = w / w.sum()
    
    s_pair = s[:, None] + s[None, :]  # (n, n)
    w_pair = w[:, None] * w[None, :]  # (n, n)
    
    flat_s = s_pair.ravel()
    flat_w = w_pair.ravel()
    
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    targets = 2.0 * s[None, :] - s[:, None]  # (n, n), targets[k, l]
    flat_targets = targets.ravel()
    
    idx = np.searchsorted(sorted_s, flat_targets, side='left')
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    
    return P

def _optimize_weights(support, initial_weights, time_budget=10.0):
    """Optimize weights for fixed support using L-BFGS-B."""
    start_time = time.time()
    n = len(support)
    
    w0 = np.asarray(initial_weights, dtype=np.float64)
    w0 = np.abs(w0)
    if w0.sum() <= 1e-12:
        w0 = np.ones(n) / n
    else:
        w0 = w0 / w0.sum()
    
    best_P = _compute_P(support, w0)
    best_w = w0.copy()
    
    def _log_w_to_w(x):
        x_shifted = x - x.max()
        e = np.exp(x_shifted)
        return e / e.sum()
    
    def _neg_P_log(x):
        w = _log_w_to_w(x)
        return -_compute_P(support, w)
    
    x0 = np.log(w0 + 1e-300)
    
    try:
        result = minimize(_neg_P_log, x0, method='L-BFGS-B', 
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
        
        w_opt = _log_w_to_w(result.x)
        P_opt = _compute_P(support, w_opt)
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt.copy()
    except Exception:
        pass
    
    if time.time() - start_time < time_budget:
        try:
            def _neg_P_nm(w):
                w = np.abs(w)
                s = w.sum()
                if s <= 1e-12:
                    return 0.0
                return -_compute_P(support, w / s)
            
            result2 = minimize(_neg_P_nm, w0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14})
            w_opt = np.abs(result2.x)
            s = w_opt.sum()
            if s > 1e-12:
                w_opt = w_opt / s
                P_opt = _compute_P(support, w_opt)
                if P_opt > best_P:
                    best_P = P_opt
                    best_w = w_opt.copy()
        except Exception:
            pass
    
    return best_w, best_P

def _try_config(support, weights, deadline):
    """Try a configuration and optimize weights. Returns (support, weights, P)."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    if len(s) != len(w) or len(s) < 1:
        return None, None, -1.0
    
    s = np.maximum(s, 0.0)
    w = np.abs(w)
    if w.sum() <= 1e-12:
        w = np.ones(len(s)) / len(s)
    else:
        w = w / w.sum()
    
    P = _compute_P(s, w)
    
    remaining = deadline - time.time()
    if remaining > 2.0:
        time_budget = min(15.0, remaining * 0.4)
        w_opt, P_opt = _optimize_weights(s, w, time_budget=time_budget)
        if P_opt > P:
            P = P_opt
            w = w_opt
    
    return s.copy(), w.copy(), P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 610.0
    
    best_support = None
    best_weights = None
    best_P = -1.0
    
    def update_best(s, w, P):
        nonlocal best_support, best_weights, best_P
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()
    
    # Phase 1: Quick scan of structured supports
    # Two-point: {0, a}
    for a in [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
        support = np.array([0.0, a])
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            weights = np.array([p, 1.0 - p])
            s, w, P = _try_config(support, weights, deadline)
            if s is not None:
                update_best(s, w, P)
    
    # Three-point: {0, a, b} with b >> a
    for a in [1.0, 2.0, 5.0]:
        for b in [5.0, 10.0, 20.0, 50.0, 100.0]:
            if time.time() > deadline:
                break
            support = np.array([0.0, a, b])
            # Try various weight distributions
            for p0 in [0.3, 0.5, 0.7]:
                for p1 in [0.1, 0.2, 0.3]:
                    p2 = 1.0 - p0 - p1
                    if p2 < 0.01:
                        continue
                    weights = np.array([p0, p1, p2])
                    s, w, P = _try_config(support, weights, deadline)
                    if s is not None:
                        update_best(s, w, P)
    
    # Four-point: {0, a, b, c}
    for a in [1.0, 2.0]:
        for b in [3.0, 5.0, 10.0]:
            for c in [10.0, 20.0, 50.0, 100.0]:
                if time.time() > deadline:
                    break
                support = np.array([0.0, a, b, c])
                # Most mass on 0 and c
                for p0 in [0.4, 0.5, 0.6]:
                    for pc in [0.3, 0.4, 0.5]:
                        p1 = (1.0 - p0 - pc) * 0.3
                        p2 = (1.0 - p0 - pc) * 0.7
                        if p1 < 0.001 or p2 < 0.001:
                            continue
                        weights = np.array([p0, p1, p2, pc])
                        s, w, P = _try_config(support, weights, deadline)
                        if s is not None:
                            update_best(s, w, P)
    
    # Phase 2: Optimize best configurations found
    if time.time() < deadline - 30.0 and best_support is not None:
        # Re-optimize the best with more time
        remaining = deadline - time.time() - 10.0
        if remaining > 5.0:
            s, w, P = _try_config(best_support, best_weights, deadline)
            if s is not None:
                update_best(s, w, P)
        
        # Try perturbed versions of best support
        n = len(best_support)
        for trial in range(5):
            if time.time() > deadline:
                break
            # Add a new point at a random large value
            new_val = best_support[-1] * (2.0 + trial * 0.5)
            support = np.append(best_support, new_val)
            weights = np.append(best_weights, 0.1)
            weights = weights / weights.sum()
            s, w, P = _try_config(support, weights, deadline)
            if s is not None:
                update_best(s, w, P)
    
    # Phase 3: Try larger structured supports
    for n in [5, 6, 8]:
        if time.time() > deadline:
            break
        # Support: 0, 1, ..., n-2, large_value
        for large_val in [10.0, 20.0, 50.0, 100.0]:
            support = np.append(np.arange(n - 1, dtype=np.float64), large_val)
            # Weights: more mass on 0 and large_val
            w_small = 0.6 / (n - 1)
            w_large = 0.4
            weights = np.full(n - 1, w_small)
            weights[-1] = w_large
            weights = weights / weights.sum()
            s, w, P = _try_config(support, weights, deadline)
            if s is not None:
                update_best(s, w, P)
    
    # Final: ensure valid output
    if best_support is None or best_weights is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    best_support = np.abs(best_support)
    best_weights = np.abs(best_weights)
    
    mask = best_weights > 1e-15
    if np.sum(mask) < len(best_weights):
        best_support = best_support[mask]
        best_weights = best_weights[mask]
    
    if len(best_support) < 1:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    if len(best_support) != len(best_weights):
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
