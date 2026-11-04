"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Discrete formulation on 50 equal intervals of [-1/4, 1/4], spacing dh = 1/20.
The continuous autoconvolution sampled at the grid is approximated by the
discrete convolution  (h*h)(x_i) = dh * sum_j h_j * h_k  over j+k = i + OFFSET,
where OFFSET = 1/2/dh = 10  (because x_j + x_k = x_i  <=>  j + k = i + 10).

The objective is scale-invariant, so we fix ||h||_2 = 1 and minimize
||h*h||_inf  (equivalently maximize  1 / (||h||_1 * ||h*h||_inf)).
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    # (h*h)(x_i) uses j+k = i + OFFSET  ->  full[i + OFFSET]
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _ratio_neg(h):
    return -_ratio(h)


def _ratio_and_grad_smooth(h):
    """
    Computes the ratio R and the gradient of log(R) w.r.t h.
    Uses the active set approximation for the max term in the denominator.
    """
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0, np.zeros_like(h)
    
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0, np.zeros_like(h)
    
    # Find index of max in ac
    idx_max = int(np.argmax(ac))
    
    # Corresponding index in the full convolution
    M = OFFSET + idx_max
    
    # Gradient of log(R)
    # log R = log(l2sq) - log(l1) - log(acmax)
    # grad log(l2sq) = 2*h / l2sq
    # grad log(l1) = ones / l1
    # grad log(acmax) = grad(acmax) / acmax
    # grad(acmax) = 2 * h_shifted, where h_shifted[i] = h[M-i]
    
    grad_l2 = 2.0 * h / l2sq
    grad_l1 = np.ones_like(h) / l1
    
    # Construct h_shifted
    h_shifted = np.zeros_like(h)
    i_start = max(0, M - (N - 1))
    i_end = min(N, M + 1)
    
    if i_start < i_end:
        i_range = np.arange(i_start, i_end)
        j_range = M - i_range
        h_shifted[i_range] = h[j_range]
        
    grad_ac = 2.0 * h_shifted / acmax
    
    grad_logR = grad_l2 - grad_l1 - grad_ac
    
    val = l2sq / (l1 * acmax)
    return val, grad_logR


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0:
        return h
    return h / s


def _make_gaussian_lobes(centers, widths, heights):
    """Create a signal from Gaussian lobes at given centers with given widths and heights."""
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N)
    for c, w, ht in zip(centers, widths, heights):
        if w <= 0: continue
        h += ht * np.exp(-((x - c) ** 2) / (2 * w ** 2))
    return h


def _local_optimize(h0, time_budget):
    """Run L-BFGS-B on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def neg_logR(x):
        x = np.maximum(x, 0.0)
        v, _ = _ratio_and_grad_smooth(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        v, g = _ratio_and_grad_smooth(x)
        if v <= 0.0:
            return 1e12, np.zeros_like(x)
        return -math.log(v), -g

    bounds = [(0.0, None)] * N
    
    # Phase 1: L-BFGS-B with analytic subgradient
    try:
        res = minimize(
            neg_logR, h0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget * 0.6 else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead polish (robust to non-smoothness)
    try:
        if _time_left() > 2.0:
            # Use a smaller time budget for NM as it can be slow
            nm_budget = max(1.0, time_budget * 0.4)
            t_nm = time.time()
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t_nm) > nm_budget else x),
            )
            if res2.x is not None:
                rx = np.maximum(res2.x, 0.0)
                rv = _ratio(rx)
                if rv > best_v:
                    best_v = rv
                    best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Parametric Geometric Search
    # We search over configurations of Gaussian lobes.
    # This avoids the ruggedness of raw coefficient space.
    
    candidates = []
    
    # 1. Symmetric and structured lobes
    # Try 1, 2, 3, 4 lobes
    for n_lobe in [1, 2, 3, 4]:
        for trial in range(10):
            if n_lobe == 1:
                centers = [0.0]
                widths = [0.02 + 0.01 * (trial % 5)]
                heights = [1.0]
            else:
                if trial == 0:
                    # Symmetric
                    centers = np.linspace(-0.2, 0.2, n_lobe)
                else:
                    # Random sorted
                    centers = np.sort(rng.uniform(-0.22, 0.22, n_lobe))
                
                widths = rng.uniform(0.01, 0.04, n_lobe)
                heights = rng.uniform(0.5, 1.5, n_lobe)
            
            h = _make_gaussian_lobes(centers, widths, heights)
            if np.sum(h) > 0:
                h = _normalize(h)
                v = _ratio(h)
                if v > 0:
                    candidates.append((v, h))
    
    # 2. Random smooth signals (low-pass filtered noise)
    for trial in range(50):
        h_raw = rng.random(N)
        # Simple smoothing
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h_smooth = np.convolve(h_raw, kernel, mode='same')
        h = np.maximum(h_smooth, 0.0)
        if np.sum(h) > 0:
            h = _normalize(h)
            v = _ratio(h)
            if v > 0:
                candidates.append((v, h))
                
    # 3. Sparse spikes (from current code, kept as backup)
    for pos in range(0, N, 5):
        s = np.zeros(N); s[pos] = 1.0
        candidates.append((_ratio(s), s))
        
    # Sort candidates by score
    candidates.sort(key=lambda x: -x[0])
    
    # Keep top K candidates for local optimization
    top_k = 20
    if len(candidates) > top_k:
        candidates = candidates[:top_k]
    
    # Update global best from candidates
    if candidates:
        best_v = candidates[0][0]
        best_h = candidates[0][1].copy()
    
    # Phase 2: Local Optimization on top candidates
    # Budget allocation: ~100 seconds for optimization
    total_time_budget = 90.0
    time_per_candidate = total_time_budget / len(candidates) if candidates else 0
    
    for v0, h0 in candidates:
        if _time_left() < time_per_candidate + 5.0:
            break
            
        # If the candidate is already very good, give it more time?
        # Or just fixed time. Fixed time is safer.
        t_budget = time_per_candidate
        
        h_opt, v_opt = _local_optimize(h0, time_budget=t_budget)
        
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
            
    # Phase 3: Final Polish on the global best
    if best_h is not None and _time_left() > 10.0:
        try:
            h_pol, v_pol = _local_optimize(best_h, time_budget=_time_left() - 12.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    
    return [float(x) for x in best_h]


if __name__ == '__main__':
    # quick self-test / timing
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
