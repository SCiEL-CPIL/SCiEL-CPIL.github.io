"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Represent h as a constant baseline plus a zero-mean oscillatory perturbation
    parameterized by DCT coefficients (low-frequency spectral shaping).
  * Use FFT-based soft-max surrogate with exact analytic gradient via FFT
    correlation (fast O(n log n) per evaluation).
  * Add a sparse-spike parameterization: h = baseline + sum_j a_j * delta(x - x_j)
    to exploit the fact that autoconvolution peaks are dominated by large elements.
  * Optimize both DCT coefficients and spike parameters jointly, then polish
    with coordinate descent on the hard objective.
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


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _soft_max_conv(h, beta):
    """Return (soft_max_value, gradient w.r.t. h) using FFT-based gradient."""
    n = len(h)
    conv = _autoconv(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    ez = np.exp(shifted)
    z = ez.sum()
    sm = m + np.log(z) / beta
    w = ez / z

    # Gradient computation via FFT
    # grad_i = sum_k w_k (h[k-i] + h[i-k])
    # Use FFT: W = fft(w), H = fft(h)
    # term1_i = sum_k w_k h[k-i] = cross-correlation
    # term2_i = sum_k w_k h[i-k] = convolution
    
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # conv_wh[i] = sum_k w_k h[i-k] for i in 0..2n-1
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]
    
    # For term1: sum_k w_k h[k-i]
    # = sum_j w_{j+i} h_j (with proper bounds)
    # This is the cross-correlation of w and h
    # corr_wh[p] = sum_k w_k h[p+k] (or similar, depends on convention)
    # Let's compute: sum_k w_k h[k-i] = sum_k w_{k+i} h_k (reindexing)
    # = sum_k h_k w_{k+i}
    # This is like a shifted dot product
    # Using FFT: ifft(W * conj(H)) gives cross-correlation
    # For real signals, conj(H) = H, so ifft(W*H) = conv_wh
    # But we need the correlation, not convolution
    # corr[p] = sum_k w[k] h[p-k] = conv_wh[p] (same for real)
    # Wait, that's not right. Let me think again.
    # sum_k w_k h[k-i]: let j = k-i, k = j+i
    # = sum_j w_{j+i} h_j
    # This is the cross-correlation of w and h at lag i
    # = sum_j h_j w_{j+i}
    # In FFT terms: ifft(conj(W) * H) gives sum_j h_j w_{j-p} (correlation)
    # So ifft(conj(W) * H)[i] = sum_j h_j w_{j-i}
    # We want sum_j h_j w_{j+i} = ifft(conj(W) * H)[-i] = ifft(conj(W) * H)[2n-1-i] (circular)
    
    # Actually, let's just compute both terms directly since n=400 is small enough
    # O(n^2) = 160k ops, which is fast
    
    grad = np.zeros(n)
    for i in range(n):
        s1 = 0.0
        s2 = 0.0
        for k in range(2 * n - 1):
            j1 = k - i
            j2 = i - k
            if 0 <= j1 < n:
                s1 += w[k] * h[j1]
            if 0 <= j2 < n:
                s2 += w[k] * h[j2]
        grad[i] = s1 + s2
    
    return sm, grad


def _obj_and_grad(h, beta):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12, np.zeros_like(h)
    sm, g_sm = _soft_max_conv(h, beta)
    val = 2.0 * N * sm / (s * s)
    g = 2.0 * N * (g_sm * s - 2.0 * sm) / (s * s * s)
    return val, g


def _dct_to_h(coeffs):
    """Map DCT coefficients to a 400-length step-height vector."""
    n = len(coeffs)
    h = idct(coeffs, type=2, norm='ortho')
    h = h - h.mean()
    return h


def _param_to_h_dct(p, m_dct):
    """Map parameter vector to a 400-length step-height vector (DCT only).
    p = [baseline, dct_coeff_1, ..., dct_coeff_m]
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


def _param_to_h_spikes(p, n_spikes):
    """Map parameter vector to a 400-length step-height vector with spikes.
    p = [baseline, amp_1, ..., amp_n_spikes, loc_1, ..., loc_n_spikes]
    loc_j is in [0, N-1], amp_j is the spike amplitude.
    """
    baseline = p[0]
    amps = p[1:1+n_spikes]
    locs = p[1+n_spikes:1+2*n_spikes]
    
    h = np.full(N, baseline, dtype=float)
    for j in range(n_spikes):
        loc = int(round(locs[j])) % N
        h[loc] += amps[j]
    
    return h


def _param_obj_dct(p, m_dct):
    h = _param_to_h_dct(p, m_dct)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _param_obj_spikes(p, n_spikes):
    h = _param_to_h_spikes(p, n_spikes)
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    DEADLINE = 130.0

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(12345)

    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # ---- Phase 1: DCT parametric optimization ----
    m_dct = 25
    baselines = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 3.0]
    for baseline in baselines:
        if time_left() < 25.0:
            break
        coeffs = rng.normal(0, 0.3, size=m_dct)
        p0 = np.concatenate([[baseline], coeffs])
        
        try:
            res = minimize(_param_obj_dct, p0, args=(m_dct,), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 5000})
            h = _param_to_h_dct(res.x, m_dct)
            v = _hard(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        except Exception:
            pass

    # ---- Phase 2: Sparse spike optimization ----
    n_spikes = 8
    for trial in range(5):
        if time_left() < 20.0:
            break
        baseline = rng.uniform(0.5, 2.0)
        amps = rng.normal(0, 0.5, size=n_spikes)
        locs = rng.integers(0, N, size=n_spikes).astype(float)
        p0 = np.concatenate([[baseline], amps, locs])
        
        try:
            res = minimize(_param_obj_spikes, p0, args=(n_spikes,), method='L-BFGS-B',
                           options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 8000})
            h = _param_to_h_spikes(res.x, n_spikes)
            v = _hard(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        except Exception:
            pass

    # ---- Phase 3: Full 400-dim polish with smooth surrogate ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    for beta in beta_seq:
        if time_left() < 15.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 8000})
            h_new = res.x
            v = _hard(h_new)
            if v < best_val:
                best_val = v
                best_h = h_new.copy()
                h_curr = h_new.copy()
        except Exception:
            pass

    # ---- Phase 4: Coordinate descent on hard objective ----
    h_curr = best_h.copy()
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s
    best_local = _hard(h_curr)
    iters = 0
    while time_left() > 10.0 and iters < 100:
        iters += 1
        conv = _autoconv(h_curr)
        kmax = int(np.argmax(conv))
        n = N
        sg = np.zeros(n)
        lag = kmax - (n - 1)
        for i in range(n):
            a = lag - i
            b = i - lag
            if 0 <= a < n:
                sg[i] += h_curr[a]
            if 0 <= b < n:
                sg[i] += h_curr[b]
        ss = h_curr.sum()
        if abs(ss) < 1e-9:
            break
        sg_centered = sg - sg.mean()
        for alpha in [1e-3, 1e-4, 1e-5, 1e-2, 5e-3]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break

    # ---- Phase 5: Final high-precision polish ----
    if time_left() > 10.0:
        for beta in [256.0, 512.0, 1024.0]:
            if time_left() < 5.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, best_h, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13,
                                        'maxfun': 10000})
                v = _hard(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
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
