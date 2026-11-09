# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P(s, w):
    """Exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <)."""
    n = s.size
    if n <= 1:
        return 0.0
    # Build all n^2 pairwise sums (for X1+X2)
    pair_sums = s[:, None] + s[None, :]
    pair_weights = w[:, None] * w[None, :]
    # Sort pair sums
    flat_s = pair_sums.ravel()
    flat_w = pair_weights.ravel()
    order = np.argsort(flat_s)
    sorted_sums = flat_s[order]
    sorted_ws = flat_w[order]
    cumsum = np.cumsum(sorted_ws)
    # For each l, k: need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # Total = sum_k w_k * sum_l w_l * H(2*s_l - s_k)
    # where H(t) = sum of pair weights with pair_sum < t
    # Use searchsorted for efficiency
    total = 0.0
    for k in range(n):
        thresholds = 2.0 * s - s[k]
        # For each l, count pair sums < 2*s_l - s_k
        idx = np.searchsorted(sorted_sums, thresholds, side='left')
        H = np.zeros(n)
        mask = idx > 0
        H[mask] = cumsum[idx[mask] - 1]
        total += w[k] * np.sum(w * H)
    return float(total)


def _compute_P_fast(s, w):
    """Faster version using vectorized approach."""
    n = s.size
    if n <= 1:
        return 0.0
    # Pair sums: n x n matrix
    pair_sums = s[:, None] + s[None, :]  # n x n
    pair_w = w[:, None] * w[None, :]     # n x n
    
    # Flatten and sort
    flat_sums = pair_sums.ravel()
    flat_w = pair_w.ravel()
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    cumsum = np.cumsum(sorted_w)
    
    # For each (k, l): threshold = 2*s[l] - s[k]
    # Build all thresholds as n x n
    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]  # shape (n, n), [k, l]
    
    # Searchsorted for each threshold
    # searchsorted expects 1D, so ravel
    idx = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    H = np.zeros(n * n)
    mask = idx > 0
    H[mask] = cumsum[idx[mask] - 1]
    H = H.reshape(n, n)  # H[k, l] = P[X1+X2 < 2*s_l - s_k]
    
    # Total = sum_k sum_l w[k]*w[l]*H[k,l]
    total = np.sum(w[:, None] * w[None, :] * H)
    return float(total)


def _negP(params):
    """Objective: minimize -P. params = [s_1,...,s_n, logw_1,...,logw_n]"""
    n = len(params) // 2
    raw_s = params[:n]
    raw_w = params[n:]
    # s must be >= 0; use softplus-like or just exp
    s = np.exp(raw_s)  # ensures positive
    # Normalize weights
    w = np.exp(raw_w - np.max(raw_w))
    w = w / w.sum()
    P = _compute_P_fast(s, w)
    return -P


def _negP_constrained(params):
    """For scipy minimize with bounds: params directly encode s>=0 and log-weights."""
    n = len(params) // 2
    s = params[:n]  # already >= 0 due to bounds
    raw_w = params[n:]
    w = np.exp(raw_w - np.max(raw_w))
    w = w / w.sum()
    P = _compute_P_fast(s, w)
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 580.0  # safety margin
    rng = np.random.default_rng(42)
    
    best_P = -1.0
    best_s = None
    best_w = None
    
    def consider(s, w):
        nonlocal best_P, best_s, best_w
        if s.size < 1:
            return
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        if np.any(s < -1e-12) or np.any(w < -1e-12):
            return
        if w.sum() <= 1e-12:
            return
        w = w / w.sum()
        P = _compute_P_fast(s, w)
        if P > best_P:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
    
    # --- Analytical 2-point structure ---
    # Two points: s = [a, b] with a << b, weights [p, 1-p]
    # For a < b/2: P = 3p^2 - 5p^3 + 2p^4
    # dP/dp = 6p - 15p^2 + 8p^3 = p(6 - 15p + 8p^2)
    # 8p^2 - 15p + 6 = 0 -> p = (15 ± sqrt(225-192))/16 = (15 ± sqrt(33))/16
    # p* = (15 - sqrt(33))/16 ≈ 0.5785
    p_opt = (15.0 - np.sqrt(33.0)) / 16.0
    
    # Use a very small a to ensure a < b/2 robustly, and optimize b
    # P is independent of b as long as a < b/2, so any b > 2a works
    # Use a = 1e-10, b = 1.0 (well within constraint)
    s2 = np.array([1e-10, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    consider(s2, w2)
    
    # Also try a slightly larger a to check robustness
    for a_test in [1e-8, 1e-6, 1e-4, 1e-3]:
        s2t = np.array([a_test, 1.0])
        w2t = np.array([p_opt, 1.0 - p_opt])
        consider(s2t, w2t)
    
    # Also try the other regime: a >= b/2, P = (1-p)*p^3
    # dP/dp = 3p^2(1-p) - p^3(1-p)... wait P = p^3(1-p), dP/dp = 3p^2(1-p) - p^3 = p^2(3-4p)
    # p* = 3/4, P = (27/64)(1/4) = 27/256 ≈ 0.1055
    # vs previous: 3*(0.5785)^2 - 5*(0.5785)^3 + 2*(0.5785)^4
    # = 3*0.3347 - 5*0.1936 + 2*0.1120 = 1.0040 - 0.9680 + 0.2240 = 0.2600
    # So first regime is much better.
    
    # Try various 2-point configs to confirm
    for a in [1e-6, 1e-4, 1e-3, 0.01, 0.1]:
        for b in [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]:
            if a < b/2:
                P_est = (1-p_opt)*(p_opt**3 + 3*p_opt**2*(1-p_opt))
            else:
                P_est = (1-p_opt)*p_opt**3
            s_test = np.array([a, b])
            w_test = np.array([p_opt, 1-p_opt])
            consider(s_test, w_test)
    
    # --- Now optimize 3-point and 4-point structures ---
    # Strategy: fix the "structure" (which points are small, which large) and optimize
    
    def optimize_n(n, s0, w0, bounds_s=None, maxiter=5000):
        """Optimize n-point measure."""
        # Parameterize: s_i >= 0 (use log for optimization), weights via softmax
        # params: [log(s_1),...,log(s_n), logw_1,...,logw_n]
        def objective(params):
            raw_s = params[:n]
            raw_w = params[n:]
            s = np.exp(raw_s)
            w = np.exp(raw_w - np.max(raw_w))
            w = w / w.sum()
            return _compute_P_fast(s, w)
        
        def neg_obj(params):
            return -objective(params)
        
        log_s0 = np.log(np.maximum(s0, 1e-12))
        log_w0 = np.log(np.maximum(w0, 1e-12))
        x0 = np.concatenate([log_s0, log_w0])
        
        # Bounds: log_s in [-25, 25], log_w in [-25, 25]
        bounds = [(-25.0, 25.0)] * (2 * n)
        
        try:
            res = optimize.minimize(neg_obj, x0, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9})
            if res.x is not None and np.all(np.isfinite(res.x)):
                s = np.exp(res.x[:n])
                raw_w = res.x[n:]
                w = np.exp(raw_w - np.max(raw_w))
                w = w / w.sum()
                consider(s, w)
                return s, w
        except Exception:
            pass
        
        # Try Nelder-Mead on raw params (s >= 0, weights normalized)
        def neg_obj_raw(params):
            s = np.maximum(params[:n], 0.0)
            w = np.maximum(params[n:], 0.0)
            if w.sum() < 1e-15:
                return 1.0
            w = w / w.sum()
            return -_compute_P_fast(s, w)
        
        x0_raw = np.concatenate([s0, w0])
        try:
            res2 = optimize.minimize(neg_obj_raw, x0_raw, method='Nelder-Mead',
                                     options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-11})
            if res2.x is not None and np.all(np.isfinite(res2.x)):
                s = np.maximum(res2.x[:n], 0.0)
                w = np.maximum(res2.x[n:], 0.0)
                if w.sum() > 1e-15:
                    w = w / w.sum()
                    consider(s, w)
                    return s, w
        except Exception:
            pass
        return s0, w0
    
    # 3-point: [small, medium, large]
    for trial in range(30):
        if time.time() - t0 > deadline:
            break
        a = rng.uniform(1e-6, 0.1)
        b = rng.uniform(0.5, 2.0)
        c = rng.uniform(5.0, 50.0)
        pa = rng.uniform(0.3, 0.7)
        pb = rng.uniform(0.1, 0.4)
        pc = 1.0 - pa - pb
        if pc < 0.01:
            pc = 0.01
            pa = (1.0 - pc - pb) / 2
            pb = (1.0 - pc - pa)
        s3 = np.array([a, b, c])
        w3 = np.array([pa, pb, pc])
        w3 = w3 / w3.sum()
        consider(s3, w3)
        optimize_n(3, s3, w3)
        
        # Also try 3-point with one very small point (near 2-point structure)
        a_tiny = rng.uniform(1e-8, 1e-4)
        b_med = rng.uniform(0.1, 1.0)
        c_large = rng.uniform(2.0, 20.0)
        pa_t = rng.uniform(0.4, 0.65)
        pb_t = rng.uniform(0.1, 0.35)
        pc_t = 1.0 - pa_t - pb_t
        if pc_t < 0.01:
            pc_t = 0.01
            pa_t = (1.0 - pc_t - pb_t) / 2
            pb_t = 1.0 - pc_t - pa_t
        s3b = np.array([a_tiny, b_med, c_large])
        w3b = np.array([pa_t, pb_t, pc_t])
        w3b = w3b / w3b.sum()
        consider(s3b, w3b)
        optimize_n(3, s3b, w3b)
    
    # 4-point: [tiny, small, medium, large]
    for trial in range(20):
        if time.time() - t0 > deadline:
            break
        a = rng.uniform(1e-7, 1e-3)
        b = rng.uniform(0.05, 0.5)
        c = rng.uniform(1.0, 5.0)
        d = rng.uniform(10.0, 100.0)
        wa = rng.uniform(0.3, 0.6)
        wb = rng.uniform(0.15, 0.35)
        wc = rng.uniform(0.05, 0.2)
        wd = 1.0 - wa - wb - wc
        if wd < 0.01:
            wd = 0.01
            wa = (1.0 - wd - wb - wc) / 3
            wb = (1.0 - wd - wa - wc) / 2
            wc = 1.0 - wd - wa - wb
        s4 = np.array([a, b, c, d])
        w4 = np.array([wa, wb, wc, wd])
        w4 = w4 / w4.sum()
        consider(s4, w4)
        optimize_n(4, s4, w4)
        
        # Also try 4-point with two tiny points
        a1 = rng.uniform(1e-8, 1e-5)
        a2 = rng.uniform(1e-5, 1e-3)
        c_mid = rng.uniform(0.5, 3.0)
        d_large = rng.uniform(5.0, 50.0)
        w1 = rng.uniform(0.3, 0.5)
        w2 = rng.uniform(0.1, 0.3)
        w3 = rng.uniform(0.1, 0.3)
        w4v = 1.0 - w1 - w2 - w3
        if w4v < 0.01:
            w4v = 0.01
            w1 = (1.0 - w4v - w2 - w3) / 1.0
            w2 = (1.0 - w4v - w1 - w3) / 2.0
            w3 = 1.0 - w4v - w1 - w2
        s4b = np.array([a1, a2, c_mid, d_large])
        w4b = np.array([w1, w2, w3, w4v])
        w4b = w4b / w4b.sum()
        consider(s4b, w4b)
        optimize_n(4, s4b, w4b)
    
    # 5-point
    for trial in range(15):
        if time.time() - t0 > deadline:
            break
        s5 = np.sort(rng.uniform(1e-5, 20.0, 5))
        w5 = rng.dirichlet(np.ones(5) * 0.5)
        consider(s5, w5)
        optimize_n(5, s5, w5)
        
        # Also try 5-point with hierarchical structure
        s5b = np.array([1e-8, 1e-5, 0.1, 1.0, 10.0])
        w5b = rng.dirichlet(np.array([2.0, 1.5, 1.0, 0.8, 0.5]))
        consider(s5b, w5b)
        optimize_n(5, s5b, w5b)
    
    # Final polish: take best found and do fine optimization
    if best_s is not None:
        n_best = best_s.size
        if time.time() - t0 < deadline - 30:
            # Fine-tune with high-precision Nelder-Mead
            def neg_obj_final(params):
                s = np.maximum(params[:n_best], 0.0)
                w = np.maximum(params[n_best:], 0.0)
                if w.sum() < 1e-15:
                    return 1.0
                w = w / w.sum()
                return -_compute_P_fast(s, w)
            
            x0 = np.concatenate([best_s, best_w])
            try:
                resf = optimize.minimize(neg_obj_final, x0, method='Nelder-Mead',
                                         options={'maxiter': 200000, 'xatol': 1e-14, 'fatol': 1e-15})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    s = np.maximum(resf.x[:n_best], 0.0)
                    w = np.maximum(resf.x[n_best:], 0.0)
                    if w.sum() > 1e-15:
                        w = w / w.sum()
                        consider(s, w)
            except Exception:
                pass
            
            # Second polish pass with even tighter tolerance
            if time.time() - t0 < deadline - 15:
                try:
                    resf2 = optimize.minimize(neg_obj_final, x0, method='Nelder-Mead',
                                             options={'maxiter': 300000, 'xatol': 1e-15, 'fatol': 1e-16})
                    if resf2.x is not None and np.all(np.isfinite(resf2.x)):
                        s = np.maximum(resf2.x[:n_best], 0.0)
                        w = np.maximum(resf2.x[n_best:], 0.0)
                        if w.sum() > 1e-15:
                            w = w / w.sum()
                            consider(s, w)
                except Exception:
                    pass
    
    # Prepare final answer
    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    
    # Remove near-zero weights to reduce n if beneficial
    min_w = 1e-15
    mask = w > min_w
    if mask.sum() < s.size:
        s = s[mask]
        w = w[mask]
        w = w / w.sum()
    
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END
