# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure.
    P = sum_{k,l} w_k*w_l * sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i*w_j
    """
    n = len(support)
    w = weights / weights.sum()
    
    pair_sums = support[:, None] + support[None, :]
    pair_weights = w[:, None] * w[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            mask = pair_sums < threshold
            inner = np.sum(pair_weights * mask)
            P += w[k] * w[l] * inner
    return P


def _compute_P_fast(support, weights):
    """
    Fast computation using sorted support and prefix sums.
    F(t) = P[X1+X2 < t] = sum_{i,j: s_i+s_j < t} w_i*w_j
    For sorted support, for each i, find max j where s_i+s_j < t.
    """
    n = len(support)
    w = weights / weights.sum()
    w_prefix = np.concatenate([[0.0], np.cumsum(w)])
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            if threshold <= 0:
                continue
            inner = 0.0
            for i in range(n):
                target = threshold - support[i]
                if target <= 0:
                    break
                j_max = np.searchsorted(support, target, side='left') - 1
                if j_max >= 0:
                    inner += w[i] * w_prefix[j_max + 1]
            P += w[k] * w[l] * inner
    return P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize a discrete measure to maximize P[X1+X2+X3 < 2*X4].
    
    Strategy:
    1. Analytical 2-point solution
    2. 3-point {0, a, 1} with optimized weights
    3. 4-point {0, a, b, 1} with optimized weights
    4. 5-point with geometric spacing
    5. 6-point with more refined spacing
    """
    deadline = time.time() + 590
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def _update_best(P, support, weights):
        nonlocal best_score, best_support, best_weights
        if P > best_score:
            best_score = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # ===== Two-point: {0, 1}, P(X=1)=q =====
    q_star = (1.0 + np.sqrt(33.0)) / 16.0
    sup2 = np.array([0.0, 1.0])
    wt2 = np.array([1 - q_star, q_star])
    P2 = _compute_P(sup2, wt2)
    _update_best(P2, sup2, wt2)
    
    # ===== Three-point: {0, a, 1} =====
    def _P_three(ratio, p1, p2):
        if p1 < 0 or p2 < 0 or p1 + p2 > 1:
            return -1.0
        p0 = 1.0 - p1 - p2
        if p0 < 0:
            return -1.0
        sup = np.array([0.0, ratio, 1.0])
        wt = np.array([p0, p1, p2])
        return _compute_P_fast(sup, wt)
    
    for ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if time.time() > deadline - 60:
            break
        best_local = -1.0
        best_p = None
        for p1_0 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            for p2_0 in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                if p1_0 + p2_0 > 0.95:
                    continue
                x0 = np.array([p1_0, p2_0])
                try:
                    res = minimize(
                        lambda x: -_P_three(ratio, x[0], x[1]),
                        x0, method='L-BFGS-B',
                        bounds=[(0.0, 0.95), (0.0, 0.95)],
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    P_val = -res.fun
                    if P_val > best_local:
                        best_local = P_val
                        best_p = res.x
                except Exception:
                    continue
        
        if best_local > -0.5 and best_p is not None:
            p0 = 1.0 - best_p[0] - best_p[1]
            sup = np.array([0.0, ratio, 1.0])
            wt = np.array([max(p0, 0), max(best_p[0], 0), max(best_p[1], 0)])
            wt = wt / wt.sum()
            P_val = _compute_P_fast(sup, wt)
            _update_best(P_val, sup, wt)
    
    # ===== Four-point: {0, a, b, 1} =====
    def _P_four(a, b, p1, p2, p3):
        if p1 < 0 or p2 < 0 or p3 < 0 or p1 + p2 + p3 > 1:
            return -1.0
        p0 = 1.0 - p1 - p2 - p3
        if p0 < 0:
            return -1.0
        sup = np.array([0.0, a, b, 1.0])
        wt = np.array([p0, p1, p2, p3])
        return _compute_P_fast(sup, wt)
    
    for a in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        if time.time() > deadline - 45:
            break
        for b in [0.5, 0.6, 0.7, 0.8, 0.9]:
            if time.time() > deadline - 45:
                break
            if a >= b:
                continue
            best_local = -1.0
            best_p = None
            for p1_0 in [0.1, 0.2, 0.3]:
                for p2_0 in [0.1, 0.2, 0.3]:
                    for p3_0 in [0.1, 0.2, 0.3]:
                        if p1_0 + p2_0 + p3_0 > 0.9:
                            continue
                        x0 = np.array([p1_0, p2_0, p3_0])
                        try:
                            res = minimize(
                                lambda x: -_P_four(a, b, x[0], x[1], x[2]),
                                x0, method='L-BFGS-B',
                                bounds=[(0.0, 0.9)] * 3,
                                options={'maxiter': 80, 'ftol': 1e-12}
                            )
                            P_val = -res.fun
                            if P_val > best_local:
                                best_local = P_val
                                best_p = res.x
                        except Exception:
                            continue
            
            if best_local > -0.5 and best_p is not None:
                p0 = 1.0 - best_p[0] - best_p[1] - best_p[2]
                sup = np.array([0.0, a, b, 1.0])
                wt = np.array([max(p0, 0), max(best_p[0], 0), max(best_p[1], 0), max(best_p[2], 0)])
                wt = wt / wt.sum()
                P_val = _compute_P_fast(sup, wt)
                _update_best(P_val, sup, wt)
    
    # ===== Five-point: {0, 1/r, 1/r^2, 1/r^3, 1} =====
    for r in [2.0, 2.5, 3.0, 3.5, 4.0]:
        if time.time() > deadline - 30:
            break
        sup = np.array([0.0, 1.0/r, 1.0/r**2, 1.0/r**3, 1.0])
        sup = np.sort(sup)
        
        best_local = -1.0
        best_wt = None
        for init in [np.array([0.3, 0.2, 0.15, 0.1, 0.25]),
                     np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                     np.array([0.4, 0.1, 0.1, 0.1, 0.3]),
                     np.array([0.1, 0.1, 0.1, 0.3, 0.4]),
                     np.array([0.5, 0.1, 0.1, 0.05, 0.25])]:
            try:
                res = minimize(
                    lambda x: -_compute_P_fast(sup, np.exp(x)),
                    np.log(init + 1e-10), method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                P_val = -res.fun
                if P_val > best_local:
                    best_local = P_val
                    best_wt = np.exp(res.x)
            except Exception:
                continue
        
        if best_local > -0.5 and best_wt is not None:
            best_wt = best_wt / best_wt.sum()
            P_val = _compute_P_fast(sup, best_wt)
            _update_best(P_val, sup, best_wt)
    
    # ===== Six-point: {0, 1/r, 1/r^2, 1/r^3, 1/r^4, 1} =====
    for r in [2.0, 2.5, 3.0]:
        if time.time() > deadline - 15:
            break
        sup = np.array([0.0, 1.0/r, 1.0/r**2, 1.0/r**3, 1.0/r**4, 1.0])
        sup = np.sort(sup)
        
        best_local = -1.0
        best_wt = None
        for init in [np.array([0.25, 0.15, 0.1, 0.1, 0.1, 0.3]),
                     np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.3]),
                     np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])]:
            try:
                res = minimize(
                    lambda x: -_compute_P_fast(sup, np.exp(x)),
                    np.log(init + 1e-10), method='L-BFGS-B',
                    options={'maxiter': 120, 'ftol': 1e-12}
                )
                P_val = -res.fun
                if P_val > best_local:
                    best_local = P_val
                    best_wt = np.exp(res.x)
            except Exception:
                continue
        
        if best_local > -0.5 and best_wt is not None:
            best_wt = best_wt / best_wt.sum()
            P_val = _compute_P_fast(sup, best_wt)
            _update_best(P_val, sup, best_wt)
    
    # ===== Final validation =====
    if best_support is None or best_weights is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([1 - q_star, q_star], dtype=np.float64)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
