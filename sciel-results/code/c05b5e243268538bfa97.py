import numpy as np
from scipy.optimize import minimize
import time

N = 50
W = 1.0 / N

def _autoconv_sup(h):
    n = len(h)
    max_val = 0.0
    for m in range(2 * n):
        val = 0.0
        for i in range(n):
            hi_i = i + 1
            lo_i = i
            for j in range(n):
                k = m - j
                lo = max(lo_i, k - 1)
                hi = min(hi_i, k)
                if hi > lo:
                    val += h[i] * h[j] * (hi - lo)
        val *= W
        if val > max_val:
            max_val = val
    return max_val

def _compute_ratio(h):
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0
    l2_sq = np.dot(h, h) * W
    l1 = s * W
    sup_c = _autoconv_sup(h)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    best_h = np.ones(N)
    best_val = _compute_ratio(best_h)
    
    # Evaluate structured patterns
    candidates = [np.ones(N)]
    # Symmetric bumps of various widths
    for width in range(1, N + 1):
        p = np.zeros(N)
        c = N // 2
        lo = max(0, c - width // 2)
        hi = min(N, lo + width)
        p[lo:hi] = 1.0
        candidates.append(p)
    # Symmetric triangles
    for peak in range(1, N):
        p = np.zeros(N)
        c = N // 2
        for i in range(N):
            p[i] = max(0.0, 1.0 - abs(i - c) / peak)
        candidates.append(p)
    
    for p in candidates:
        if time.time() > deadline:
            break
        v = _compute_ratio(p)
        if v > best_val:
            best_val = v
            best_h = p.copy()
    
    # Local optimization
    def neg_ratio(x):
        x = np.maximum(x, 0)
        s = np.sum(x)
        if s < 1e-15:
            return 0.0
        h = x / s
        return -_compute_ratio(h)
    
    x0 = best_h.copy()
    s0 = np.sum(x0)
    if s0 > 0:
        x0 = x0 / s0
    
    try:
        res = minimize(neg_ratio, x0, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
        if time.time() < deadline:
            h_opt = np.maximum(res.x, 0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt /= s
                v = _compute_ratio(h_opt)
                if v > best_val:
                    best_val = v
                    best_h = h_opt.copy()
    except:
        pass
    
    # A couple more starts
    for trial in range(2):
        if time.time() > deadline:
            break
        x = np.random.rand(N)
        x = (x + x[::-1]) / 2.0  # symmetric
        s = np.sum(x)
        if s > 0:
            x /= s
            try:
                res = minimize(neg_ratio, x, method='Powell', options={'maxiter': 5000})
                if time.time() < deadline:
                    h_opt = np.maximum(res.x, 0)
                    s = np.sum(h_opt)
                    if s > 1e-15:
                        h_opt /= s
                        v = _compute_ratio(h_opt)
                        if v > best_val:
                            best_val = v
                            best_h = h_opt.copy()
            except:
                pass
    
    return list(best_h)
