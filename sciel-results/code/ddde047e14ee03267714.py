"""Second autocorrelation inequality -- direct height optimization.

Discrete setting: 50 nonnegative step heights on equal intervals of [-1/4, 1/4],
dh = 1/20.  The discrete autoconvolution sampled on the same grid is
    (h*h)(x_i) = dh * sum_{j+k = i + OFFSET} h_j h_k,   OFFSET = 10.

Objective (scale-invariant):  R = ||h||_2^2 / (||h||_1 * ||h*h||_inf).

We optimize the 50 heights directly using L-BFGS-B with a smooth approximation
to the max (via log-sum-exp) and analytic gradients.  Multiple diverse starts
are tried; the best result is returned.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

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
    # d log_acmax / dh_k = sum_i softmax_w[i] * (1/ac_i) * d ac_i / dh_k
    # But we're using the smooth max, so:
    # d log_acmax / dh_k = sum_i softmax_w[i] * d ac_i / dh_k / ac_max_smoothed
    # Actually, let's be more careful.
    # log_acmax = ac_max + inv_beta * log(sum_exp)
    # d log_acmax / dh_k = d ac_max / dh_k + inv_beta * (1/sum_exp) * sum_i d exp_shifted_i / dh_k
    # This is getting complicated. Let me use the simpler approach:
    # The gradient of the log-sum-exp smooth max is:
    # d log_acmax / dh_k = sum_i (softmax_w[i] / ac_i) * d ac_i / dh_k
    # Wait, that's not right either. Let me think again.
    # 
    # Let M = log(sum_j exp(ac_j / beta))
    # dM/dh_k = sum_j (exp(ac_j/beta) / sum_i exp(ac_i/beta)) * (1/beta) * dac_j/dh_k
    # = sum_j softmax_w_j * (1/beta) * dac_j/dh_k
    # where softmax_w_j = exp(ac_j/beta) / sum_i exp(ac_i/beta)
    # 
    # But we have log_acmax = ac_max + inv_beta * log(sum_exp)
    # = ac_max + inv_beta * (log(sum_j exp((ac_j - ac_max)/beta)))
    # The second term is log(sum_j softmax_w_j) = log(1) = 0? No.
    # Actually, softmax_w_j = exp((ac_j - ac_max)/beta) / sum_i exp((ac_i - ac_max)/beta)
    # So sum_j softmax_w_j = 1.
    # And log_acmax = ac_max + inv_beta * log(sum_j exp((ac_j-ac_max)/beta))
    # The gradient of this w.r.t. h_k is:
    # d ac_max/dh_k + inv_beta * (1/sum_exp) * sum_j d exp((ac_j-ac_max)/beta)/dh_k
    # This is messy. Let me just use the standard log-sum-exp gradient.
    # 
    # Let LSE = log(sum_j exp(ac_j / beta))
    # dLSE/dh_k = sum_j (exp(ac_j/beta) / sum_i exp(ac_i/beta)) * (1/beta) * dac_j/dh_k
    # = (1/beta) * sum_j w_j * dac_j/dh_k
    # where w_j = exp(ac_j/beta) / sum_i exp(ac_i/beta)
    #
    # So d log_acmax / dh_k = (1/beta) * sum_j w_j * dac_j/dh_k
    #
    # dac_j/dh_k = 2 * DH * h_{j-k} if 0 <= j-k < N, else 0
    #
    # So d log_acmax / dh_k = (2*DH/beta) * sum_j w_j * h_{j-k} (with bounds)
    
    dlog_acmax_dh = np.zeros(N)
    # dac_j/dh_k = 2*DH*h_{j-k} for j-k in [0,N)
    # sum_j w_j * dac_j/dh_k = 2*DH * sum_j w_j * h_{j-k}
    # For fixed k, sum_j w_j * h_{j-k} = sum_{m=0}^{N-1} w_{m+k} * h_m (with bounds on m+k)
    # Let's compute this efficiently.
    # For each k, we need sum over j such that j-k is in [0,N-1] of w_j * h_{j-k}
    # = sum over m=0 to N-1 of w_{m+k} * h_m where m+k < N
    for k in range(N):
        # j ranges from k to N-1, m = j-k ranges from 0 to N-1-k
        if k < N:
            dlog_acmax_dh[k] = (2.0 * DH / beta) * float(np.dot(w_j if False else softmax_w[k:], h[:N-k]))
    
    # Wait, I need to be more careful. Let me recompute.
    # dlog_acmax_dh[k] = (2*DH/beta) * sum_{j=k}^{N-1} softmax_w[j] * h[j-k]
    # Let m = j-k, so m ranges from 0 to N-1-k
    # = (2*DH/beta) * sum_{m=0}^{N-1-k} softmax_w[m+k] * h[m]
    # = (2*DH/beta) * dot(softmax_w[k:N], h[0:N-k])
    
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


def _local_optimize(h0, budget):
    """Local L-BFGS-B optimization from h0."""
    t0 = time.time()
    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            _neg_logR, h0.copy(),
            method='L-BFGS-B',
            jac=_neg_logR_jac,
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
        idx = rng.choice(N, size=n_nonzero, replace=False)
        h[idx] = rng.uniform(0.5, 2.0, n_nonzero)
        starts.append(h)
    
    # Triangle / tent functions
    for peak in range(5, N-5, 5):
        h = np.maximum(0, 1 - np.abs(np.arange(N) - peak) / 10.0)
        starts.append(h)
    
    # Cosine bumps
    for center in range(0, N, 5):
        for half_width in [5, 10, 15, 20]:
            x = np.arange(N)
            dist = np.abs(x - center)
            h = np.where(dist < half_width, 0.5 * (1 + np.cos(np.pi * dist / half_width)), 0.0)
            starts.append(h)
    
    return starts


def construct():
    global _start
    _start = time.time()
    
    best_h = np.ones(N)
    best_R = _ratio_from_h(best_h)
    
    starts = _generate_starts()
    
    # Score all starts
    scored = []
    for h in starts:
        h = np.maximum(h, 0.0)
        R = _ratio_from_h(h)
        scored.append((R, h.copy()))
    scored.sort(key=lambda t: -t[0])
    
    # Polish top starts
    top_k = min(len(scored), 40)
    for k in range(top_k):
        R0, h0 = scored[k]
        if _time_left() <= 5.0:
            break
        if R0 < best_R * 0.5 and best_R > 0:
            continue
        budget = min(2.0, max(0.3, _time_left() - 8.0))
        try:
            h_opt, R_opt = _local_optimize(h0, budget=budget)
            if R_opt > best_R:
                best_R = R_opt
                best_h = h_opt.copy()
        except Exception:
            pass
        if _time_left() <= 20.0:
            break
    
    # Final polish on best
    if _time_left() > 10.0:
        try:
            h_final, R_final = _local_optimize(best_h, budget=max(1.0, _time_left() - 12.0))
            if R_final > best_R:
                best_R = R_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_h)) or float(np.sum(best_h)) <= 0.0:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
