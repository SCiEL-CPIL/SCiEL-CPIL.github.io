<<<<<<< SEARCH
import time
import numpy as np
from scipy import optimize
=======
import time
import numpy as np
from scipy import optimize
from scipy.optimize import differential_evolution
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======

>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======

>>>>>>> REPLACE
<<<<<<< SEARCH
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
    for trial in range(20):
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
    
    # 4-point: [tiny, small, medium, large]
    for trial in range(15):
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
    
    # 5-point
    for trial in range(10):
        if time.time() - t0 > deadline:
            break
        s5 = np.sort(rng.uniform(1e-5, 20.0, 5))
        w5 = rng.dirichlet(np.ones(5) * 0.5)
        consider(s5, w5)
        optimize_n(5, s5, w5)
    
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
                                         options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-13})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    s = np.maximum(resf.x[:n_best], 0.0)
                    w = np.maximum(resf.x[n_best:], 0.0)
                    if w.sum() > 1e-15:
                        w = w / w.sum()
                        consider(s, w)
            except Exception:
                pass
=======
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

    def _global_search(n, time_budget):
        """Global search using differential evolution for n points."""
        if time.time() - t0 > deadline:
            return
        
        # Parameters: log_s (n) + log_w (n)
        # log_s in [-30, 15] (s in [~1e-13, ~3e6])
        # log_w in [-30, 5] (w can be tiny or up to ~148)
        bounds = [(-30.0, 15.0)] * n + [(-30.0, 5.0)] * n
        
        def neg_obj(params):
            raw_s = params[:n]
            raw_w = params[n:]
            s = np.exp(raw_s)
            w = np.exp(raw_w - np.max(raw_w))
            w = w / w.sum()
            return -_compute_P_fast(s, w)
        
        try:
            start_time = time.time()
            res = differential_evolution(
                neg_obj, bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.2),
                recombination=0.7,
                polish=True,
                init='sobol',
                updating='immediate',
                seed=42,
                workers=1
            )
            elapsed = time.time() - start_time
            if res.x is not None and np.all(np.isfinite(res.x)):
                s = np.exp(res.x[:n])
                raw_w = res.x[n:]
                w = np.exp(raw_w - np.max(raw_w))
                w = w / w.sum()
                consider(s, w)
        except Exception:
            pass

    # 3-point: [small, medium, large]
    for trial in range(20):
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
    
    # 4-point: [tiny, small, medium, large]
    for trial in range(15):
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
    
    # 5-point
    for trial in range(10):
        if time.time() - t0 > deadline:
            break
        s5 = np.sort(rng.uniform(1e-5, 20.0, 5))
        w5 = rng.dirichlet(np.ones(5) * 0.5)
        consider(s5, w5)
        optimize_n(5, s5, w5)
    
    # Global search for 3 and 4 points if time permits
    if time.time() - t0 < deadline - 120:
        _global_search(3, 60)
    
    if time.time() - t0 < deadline - 60:
        _global_search(4, 60)
    
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
                                         options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-13})
                if resf.x is not None and np.all(np.isfinite(resf.x)):
                    s = np.maximum(resf.x[:n_best], 0.0)
                    w = np.maximum(resf.x[n_best:], 0.0)
                    if w.sum() > 1e-15:
                        w = w / w.sum()
                        consider(s, w)
            except Exception:
                pass
>>>>>>> REPLACE
