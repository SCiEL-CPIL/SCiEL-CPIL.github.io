"""Rewritten third autocorrelation inequality optimizer.

Strategy (same family as parent, refined):
  * Represent h as a constant baseline plus a zero-mean oscillatory part.
  * Use an FFT-based soft-max surrogate of the signed max of the autoconvolution
    with an exact analytic gradient computed via FFT correlation.
  * Optimize the full 400-dim vector with L-BFGS-B using an annealed beta schedule.
  * Polish with a coordinate / subgradient step on the hard objective.
  * Add diverse starting points (DCT-restricted, sparse, cosine, random) to escape
    local minima.
  * Enforce a strict wall-clock deadline guard with safety margin.
"""
N = 400
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fftpack import dct, idct


def _autoconv(h):
    """Full autoconvolution via FFT (c[k] for lag k = -(N-1)..(N-1))."""
    n = len(h)
    f = np.fft.rfft(h, n=2 * n)
    c = np.fft.irfft(f * f, n=2 * n)
    return c


def _soft_max_conv(h, beta):
    """Return (soft_max_value, gradient w.r.t. h) using FFT-based gradient.
    
    The gradient is computed in O(n log n) using FFT cross-correlation:
    grad_i = sum_k w_k (h[k-i] + h[i-k])
    where w is the softmax weights over the autoconvolution entries.
    """
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z
    
    # Gradient via FFT:
    # grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    # The second term is (w conv h)[i] for i in 0..n-1
    # The first term is (w_rev conv h)[i] where w_rev is reversed w
    # Actually, let's use:
    # sum_k w_k h[i-k] = (w * h)[i] where * is convolution
    # sum_k w_k h[k-i] = (flip(w) * h)[i] where flip(w)_k = w_{L-1-k}
    
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # term2: sum_k w_k h[i-k] = ifft(W * H)[i] for i in 0..n-1
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]
    
    # term1: sum_k w_k h[k-i]
    # Let's compute flip(w) and convolve with h
    w_flip = w[::-1]
    WF = np.fft.rfft(w_flip, n=2 * n)
    conv_wf_h = np.fft.irfft(WF * H, n=2 * n)
    term1 = conv_wf_h[0:n]
    
    grad = term1 + term2
    return sm, grad


def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _param_to_h(p):
    """Map parameter vector to a 400-length step-height vector.
    p = [baseline, dct_coeff_1, dct_coeff_2, ..., dct_coeff_m]
    """
    baseline = p[0]
    m = len(p) - 1
    coeffs = np.zeros(N)
    coeffs[0] = 0.0
    coeffs[1:m+1] = p[1:m+1]
    h_osc = idct(coeffs, type=2, norm='ortho')
    h_osc = h_osc - h_osc.mean()
    h = baseline + h_osc
    return h


def _param_obj(p):
    h = _param_to_h(p)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    DEADLINE = 140.0  # wall-clock budget; keep margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(12345)

    # Number of DCT coefficients to use for the oscillatory part
    m_dct = 30

    # ---- Phase 1: optimize the DCT parametric family with diverse starts ----
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Try different baseline values and random DCT coefficient patterns
    baselines = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 0.0]
    for baseline in baselines:
        if time_left() < 20.0:
            break
        for seed_idx in range(3):
            if time_left() < 15.0:
                break
            # Random DCT coefficients with varying scale
            scale = 0.3 + 0.2 * seed_idx
            coeffs = rng.normal(0, scale, size=m_dct)
            p0 = np.concatenate([[baseline], coeffs])
            
            try:
                res = minimize(_param_obj, p0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                        'maxfun': 3000})
                h = _param_to_h(res.x)
                v = _hard(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
            except Exception:
                pass

    # ---- Phase 1b: diverse non-DCT starting points ----
    diverse_starts = []
    
    # Cosine wave
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h_cos = 1.0 + 0.5 * np.cos(t)
    diverse_starts.append(h_cos)
    
    # Sparse spikes
    h_sparse = np.ones(N) * 0.5
    for _ in range(5):
        idx = rng.integers(0, N)
        h_sparse[idx] += 2.0
    diverse_starts.append(h_sparse)
    
    # Alternating
    h_alt = np.ones(N) * 1.0
    h_alt[::2] += 0.5
    h_alt[1::2] -= 0.5
    diverse_starts.append(h_alt)
    
    # Random
    h_rand = rng.normal(1.0, 0.5, size=N)
    diverse_starts.append(h_rand)
    
    for h_start in diverse_starts:
        if time_left() < 15.0:
            break
        h_norm = h_start / h_start.sum() * 1.0  # normalize sum to 1
        try:
            # Quick L-BFGS-B from this start
            def obj_quick(x):
                return _hard(x)
            res = minimize(obj_quick, h_norm, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'maxfun': 2000})
            v = _hard(res.x)
            if v < best_val:
                best_val = v
                best_h = res.x.copy()
        except Exception:
            pass

    # ---- Phase 2: full 400-dim polish with smooth surrogate (FFT gradient) ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0]
    for beta in beta_seq:
        if time_left() < 15.0:
            break
        try:
            def obj(x, b=beta):
                val, _ = _obj_and_grad(x, b)
                return val
            def gr(x, b=beta):
                _, g = _obj_and_grad(x, b)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 5000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # ---- Phase 3: coordinate / subgradient refinement on hard objective ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    iters = 0
    while time_left() > 10.0 and iters < 80:
        iters += 1
        conv = _autoconv(h_curr)
        # signed max index (lag)
        kmax = int(np.argmax(conv))
        # subgradient of max(conv) w.r.t. h at the maximizing lag
        n = N
        sg = np.zeros(n)
        lag = kmax - (n - 1)
        # d c[lag] / d h_i = h[lag - i] + h[i - lag]  (zero outside)
        for i in range(n):
            a = lag - i
            b = i - lag
            if 0 <= a < n:
                sg[i] += h_curr[a]
            if 0 <= b < n:
                sg[i] += h_curr[b]
        # current sum
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        # We want to reduce maxc relative to ss^2.
        # Take a step that lowers maxc while keeping sum positive.
        sg_centered = sg - sg.mean()
        improved = False
        for alpha in [1e-3, 1e-4, 1e-5, 1e-2, 5e-4]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                improved = True
                break

    # ---- Phase 4: Final high-beta polish ----
    if time_left() > 8.0:
        for beta in [128.0, 256.0, 512.0]:
            if time_left() < 5.0:
                break
            try:
                def obj(x, b=beta):
                    val, _ = _obj_and_grad(x, b)
                    return val
                def gr(x, b=beta):
                    _, g = _obj_and_grad(x, b)
                    return g
                
                h_start = best_h.copy()
                s = h_start.sum()
                if abs(s) > 1e-9:
                    h_start = h_start / s
                
                res = minimize(obj, h_start, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 6000})
                h_new = res.x
                v = _hard(h_new)
                if v < best_val:
                    best_val = v
                    best_h = h_new.copy()
            except Exception:
                pass

    # ---- Sanity / validity ----
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        best_val = _hard(best_h)
    if abs(best_h.sum()) < 1e-10:
        best_h = np.ones(N)
        best_val = _hard(best_h)

    return list(best_h)
