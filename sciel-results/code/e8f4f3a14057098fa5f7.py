import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # (N-1)//2

_DEADLINE = 150.0
_SAFETY = 15.0
_start = 0.0


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same N-grid."""
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    return ac


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return 0.0
    h_norm = h / l1
    l2sq_norm = float(np.dot(h_norm, h_norm))
    if l2sq_norm <= 1e-300:
        return 0.0
    ac_norm = _autoconv(h_norm)
    acmax = float(np.max(ac_norm))
    if acmax <= 1e-300:
        return 0.0
    return l2sq_norm / acmax


def _logR_and_grad(h, beta=50.0):
    """Compute log R(h) and its gradient w.r.t. h, using smooth max."""
    h = np.maximum(h, 1e-300)
    l1 = float(np.sum(h))
    if l1 <= 1e-300:
        return -1e30, np.zeros(N)
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-300:
        return -1e30, np.zeros(N)
    ac = _autoconv(h)
    ac = np.maximum(ac, 1e-300)
    ac_max = float(np.max(ac))
    if ac_max <= 1e-300:
        return -1e30, np.zeros(N)
    
    # Smooth max via log-sum-exp
    inv_beta = 1.0 / beta
    shifted = ac - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp <= 1e-300:
        return -1e30, np.zeros(N)
    log_acmax = ac_max + inv_beta * math.log(sum_exp)
    softmax_w = exp_shifted / sum_exp
    
    logR = math.log(l2sq) - math.log(l1) - log_acmax
    
    # Gradient
    # d logR / dh_k = 2*h_k/l2sq - 1/l1 - d log_acmax / dh_k
    # d log_acmax / dh_k = (1/beta) * sum_j softmax_w[j] * dac_j/dh_k
    # dac_j/dh_k = 2 * DH * h_{j-k} if 0 <= j-k < N, else 0
    # For fixed k, sum_j softmax_w[j] * dac_j/dh_k = 2*DH * sum_{m=0}^{N-1-k} softmax_w[m+k] * h[m]
    
    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        if N - k > 0:
            dlog_acmax_dh[k] = (2.0 * DH / beta) * float(np.dot(softmax_w[k:], h[:N-k]))
    
    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh
    
    # Zero out gradient where h is at boundary (h <= 0)
    dlogR_dh[h <= 1e-15] = 0.0
    
    if not np.all(np.isfinite(dlogR_dh)):
        dlogR_dh = np.zeros(N)
    
    return logR, dlogR_dh


def _neg_logR(h):
    v, _ = _logR_and_grad(h)
    return -v


def _neg_logR_jac(h):
    v, g = _logR_and_grad(h)
    return -v, -g


def _local_optimize(h0, budget, beta=50.0):
    """Local L-BFGS-B optimization from h0."""
    t0 = time.time()
    bounds = [(0.0, None)] * N
    
    def obj(x):
        return _neg_logR(x)
    
    def jac(x):
        return _neg_logR_jac(x)
    
    try:
        res = minimize(
            obj, h0.copy(),
            method='L-BFGS-B',
            jac=jac,
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            h_opt = np.maximum(res.x, 0.0)
            R_opt = _ratio_from_h(h_opt)
            return h_opt, R_opt
    except Exception:
        pass
    return h0.copy(), _ratio_from_h(h0)


def _generate_starts():
    """Generate diverse starting points."""
    rng = np.random.default_rng(42)
    starts = []
    
    # Uniform
    starts.append(np.ones(N))
    
    # Gaussian bumps at various positions and widths
    for center in range(0, N, 5):
        for width in [2, 5, 10, 15, 20]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
            starts.append(h)
    
    # Multi-bump
    for n_bumps in [2, 3, 4, 5]:
        for _ in range(10):
            h = np.zeros(N)
            centers = rng.choice(N, size=n_bumps, replace=False)
            widths = rng.uniform(2, 10, n_bumps)
            for c, w in zip(centers, widths):
                h += np.exp(-0.5 * ((np.arange(N) - c) / w) ** 2)
            starts.append(h)
    
    # Random smooth
    for _ in range(30):
        h = rng.uniform(0.1, 1.0, N)
        # Smooth it
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        h = np.convolve(h, kernel, mode='same')
        starts.append(h)
    
    # Sparse
    for _ in range(20):
        h = np.zeros(N)
        n_nonzero = rng.integers(3, 10)
        indices = rng.choice(N, size=n_nonzero, replace=False)
        h[indices] = rng.uniform(0.5, 2.0, n_nonzero)
        starts.append(h)
    
    # Power laws
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x = np.linspace(0.1, 1.0, N)
        h = x ** alpha
        starts.append(h)
    
    # Lorentzians
    for center in range(0, N, 5):
        for width in [2, 5, 10]:
            x = np.arange(N)
            h = 1.0 / (1.0 + ((x - center) / width) ** 2)
            starts.append(h)
    
    return starts


def construct():
    global _start
    _start = time.time()
    
    best_h = np.ones(N)
    best_ratio = _ratio_from_h(best_h)
    
    starts = _generate_starts()
    
    # Try each start with L-BFGS-B
    for i, h0 in enumerate(starts):
        if _time_left() < 2.0:
            break
        
        # Allocate remaining time to this and subsequent starts
        n_remaining = len(starts) - i
        budget = min(3.0, _time_left() / max(n_remaining, 1) - 0.5)
        if budget < 0.5:
            budget = 0.5
        
        h_opt, R_opt = _local_optimize(h0, budget)
        
        if R_opt > best_ratio:
            best_ratio = R_opt
            best_h = h_opt.copy()
    
    # Final polish with tighter tolerance and more iterations
    if _time_left() > 5.0:
        budget = min(10.0, _time_left() - 1.0)
        h_polished, R_polished = _local_optimize(best_h, budget, beta=100.0)
        
        if R_polished > best_ratio:
            best_ratio = R_polished
            best_h = h_polished.copy()
    
    # Coordinate descent refinement
    if _time_left() > 3.0:
        h = best_h.copy()
        for _ in range(50):
            if _time_left() < 1.0:
                break
            
            improved = False
            for i in range(N):
                old_val = h[i]
                # Try a range of values around current
                candidates = [old_val * 0.5, old_val * 0.75, old_val * 1.25, old_val * 1.5, old_val + 0.1, old_val - 0.1]
                candidates = [max(0, c) for c in candidates]
                
                best_local = h.copy()
                best_local_ratio = _ratio_from_h(h)
                
                for c in candidates:
                    test_h = h.copy()
                    test_h[i] = c
                    test_ratio = _ratio_from_h(test_h)
                    if test_ratio > best_local_ratio:
                        best_local_ratio = test_ratio
                        best_local = test_h.copy()
                
                if best_local_ratio > _ratio_from_h(h) + 1e-10:
                    h = best_local
                    improved = True
            
            if not improved:
                break
        
        final_ratio = _ratio_from_h(h)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_h = h
    
    return list(best_h)
