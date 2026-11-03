import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50

# EVOLVE-BLOCK-START
def _autoconv(h):
    """Compute full autoconvolution (correlation) of h with itself."""
    n = len(h)
    conv = np.zeros(2 * n - 1)
    for k in range(2 * n - 1):
        if k < n:
            i = np.arange(k, n)
            j = np.arange(0, n - k)
        else:
            i = np.arange(0, 2 * n - 1 - k)
            j = np.arange(k - n + 1, n)
        conv[k] = np.dot(h[i], h[j])
    return conv

def _score(h):
    """Compute the score: (sum h^2) / ((sum h) * max_k conv_k)."""
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0, None)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 == 0 or l2_sq == 0:
        return 0.0
    conv = _autoconv(h)
    max_conv = np.max(conv)
    if max_conv == 0:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _score_neg(h):
    """Negative score for minimization."""
    return -_score(h)

def _score_neg_smooth(h, eps=1e-8):
    """Smooth surrogate using log-sum-exp for max."""
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0, None)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h) + eps
    if l1 < eps:
        return 0.0
    conv = _autoconv(h)
    # Soft max via log-sum-exp
    scale = 100.0
    shifted = scale * conv
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_max = log_sum_exp / scale
    if soft_max < eps:
        return 0.0
    return -l2_sq / (l1 * soft_max)

def _subgradient_step(h, lr, n_iter=100):
    """Subgradient ascent on the score."""
    h = np.array(h, dtype=float)
    best_h = h.copy()
    best_score = _score(h)
    for it in range(n_iter):
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        # Numerical gradient
        grad = np.zeros_like(h)
        for i in range(len(h)):
            h_plus = h.copy()
            h_plus[i] += 1e-5
            grad[i] = (_score(h_plus) - s) / 1e-5
        h = h + lr * grad
        h = np.clip(h, 0, None)
        # Normalize to keep scale reasonable
        if np.sum(h) > 0:
            h = h / np.sum(h) * 1.0
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start from the known good point
    h0 = np.zeros(N)
    h0[0] = 2.0 / 3.0
    h0[1] = 1.0 / 3.0
    best_h = h0.copy()
    best_score = _score(h0)
    
    # Try scipy optimization with smooth surrogate
    try:
        res = minimize(
            _score_neg_smooth,
            h0,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline:
            h_scipy = np.clip(res.x, 0, None)
            if np.sum(h_scipy) > 0:
                h_scipy = h_scipy / np.sum(h_scipy)
            s = _score(h_scipy)
            if s > best_score:
                best_score = s
                best_h = h_scipy.copy()
    except Exception:
        pass
    
    # Try subgradient ascent from multiple starting points
    start_points = [
        np.array([2/3, 1/3] + [0]*(N-2)),
        np.array([1/2, 1/2] + [0]*(N-2)),
        np.array([0.6, 0.3, 0.1] + [0]*(N-3)),
        np.array([0.5, 0.25, 0.25] + [0]*(N-3)),
        np.array([0.7, 0.2, 0.1] + [0]*(N-3)),
        np.array([0.4, 0.4, 0.2] + [0]*(N-3)),
        np.array([0.5, 0.3, 0.2] + [0]*(N-3)),
        np.array([0.6, 0.4] + [0]*(N-2)),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        h_opt, s = _subgradient_step(sp, lr=0.01, n_iter=50)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()
    
    # Local polish with scipy from the best point
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                _score_neg,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if time.time() < deadline:
                h_nm = np.clip(res2.x, 0, None)
                if np.sum(h_nm) > 0:
                    h_nm = h_nm / np.sum(h_nm)
                s = _score(h_nm)
                if s > best_score:
                    best_score = s
                    best_h = h_nm.copy()
        except Exception:
            pass
    
    # Final cleanup: ensure nonnegative and reasonable
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
