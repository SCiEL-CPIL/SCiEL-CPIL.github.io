"""Model-and-optimize program for third autocorrelation inequality.

Strategy:
  * Represent h as a constant baseline plus a zero-mean oscillatory perturbation.
  * Use the Discrete Cosine Transform (DCT) to enforce a specific low-frequency
    spectral shape that minimizes the peak-to-mean ratio of the autoconvolution.
  * Use an FFT-based soft-max surrogate of the signed max of the autoconvolution
    with an exact analytic gradient computed via FFT correlation.
  * Optimize the DCT coefficients with L-BFGS-B, then polish with a
    coordinate / subgradient step on the full 400-dim vector against the
    hard objective.
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
    
    # Gradient of soft-max w.r.t. h_i = sum_k w_k * d(conv_k)/d(h_i)
    # d(conv_k)/d(h_i) = h[k-i] + h[i-k] (zero outside 0..n-1)
    # This equals 2 * (w conv h)[i] where conv is full convolution
    # Actually: sum_k w_k h[k-i] + sum_k w_k h[i-k]
    # = (w flipped conv h) + (w conv h)
    # Let's compute via FFT:
    # grad_i = sum_k w_k (h[k-i] + h[i-k])
    # = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    # First term: let j = k-i, sum_j w_{j+i} h_j = cross-correlation of w and h
    # Second term: sum_k w_k h[i-k] = convolution of w and h at index i
    
    # Use FFT for efficiency:
    W = np.fft.rfft(w, n=2 * n)
    H = np.fft.rfft(h, n=2 * n)
    
    # conv_wh = ifft( W * H ) gives sum_k w_k h[i-k] for i in 0..2n-1
    # We want indices 0..n-1 for the second term
    conv_wh = np.fft.irfft(W * H, n=2 * n)
    term2 = conv_wh[0:n]  # sum_k w_k h[i-k]
    
    # For the first term: sum_k w_k h[k-i]
    # = sum_k w_{k-i} h_k (by change of variable, but careful with bounds)
    # Actually: sum_k w_k h[k-i] = sum_j w_{j+i} h_j
    # This is the cross-correlation of w and h at lag i
    # = ifft( W * conj(H) ) at appropriate indices?
    # Let's use: corr_wh = ifft( W * conj(H) )
    # corr_wh[p] = sum_k w_k conj(h)_{p-k} for real h = sum_k w_k h[p-k]
    # That's the same as conv_wh!
    # Wait, for real signals: ifft( W * conj(H) ) = cross-correlation
    # = sum_k w_k h[p-k] which is the same as conv_wh[p]
    # So both terms are the same? No.
    
    # Let me reconsider:
    # conv_k = sum_i h_i h_{k-i} (with proper bounds)
    # d conv_k / d h_i = h_{k-i} + h_{i-k} (by symmetry of autoconvolution)
    # So grad_i = sum_k w_k (h_{k-i} + h_{i-k})
    # = sum_k w_k h_{k-i} + sum_k w_k h_{i-k}
    
    # First term: sum_k w_k h_{k-i}
    # Let j = k-i => k = j+i, sum_j w_{j+i} h_j
    # This is the cross-correlation of w and h, but with w shifted.
    # = sum_j h_j w_{j+i}
    # This is (h conv w)[i] if we define conv properly.
    
    # Second term: sum_k w_k h_{i-k}
    # Let j = i-k => k = i-j, sum_j w_{i-j} h_j
    # = sum_j h_j w_{i-j}
    # This is (h conv w)[i] as well? No.
    
    # Actually, for real sequences:
    # sum_k w_k h_{k-i} = (w * h)[i] where * is cross-correlation
    # sum_k w_k h_{i-k} = (w * h)[i] where * is convolution
    
    # Let's just compute both via direct convolution for clarity, then optimize:
    # But n=400, so O(n^2) = 160k operations, which is fine.
    
    # First term: sum_k w_k h_{k-i} for i=0..n-1
    # For fixed i, k ranges such that 0 <= k-i < n => i <= k < i+n
    # = sum_{k=i}^{i+n-1} w_k h_{k-i} (with w_k=0 for k>=n, h_j=0 for j<0 or j>=n)
    # Actually w is defined on 0..2n-2 (length 2n-1 from conv), but we only use w[0..2n-2]
    # Wait, w has length 2n-1 (from autoconv of length 2n-1)
    # Let's just use np.convolve for correctness:
    
    # grad_i = sum_k w_k h[k-i] + sum_k w_k h[i-k]
    # = (w conv h)[i] + (w conv h)[i] ? No.
    
    # Let's use the fact that autoconvolution is symmetric:
    # conv_k = sum_i h_i h_{k-i}
    # The gradient of sum_k w_k conv_k w.r.t. h_i is:
    # sum_k w_k d/dh_i (sum_j h_j h_{k-j}) = sum_k w_k (h_{k-i} + h_{i-k})
    # (where terms are zero if indices out of range)
    
    # This is equal to:
    # sum_k w_k h_{k-i} + sum_k w_k h_{i-k}
    # = sum_k w_{k-i} h_k + sum_k w_{i-k} h_k  (by reindexing, with proper bounds)
    # = (w shifted by -i) dot h + (w shifted by i) dot h
    
    # The sum of these two is: 2 * (w conv h)[i] if w is symmetric, but w is not necessarily symmetric.
    # Actually, since conv is autoconvolution, w is the softmax of conv, which is symmetric.
    # conv_k = conv_{2n-2-k} (by symmetry of autoconvolution)
    # So w_k = w_{2n-2-k}
    # Therefore sum_k w_k h_{k-i} = sum_k w_{2n-2-k} h_{k-i} = sum_k w_k h_{2n-2-k-i}
    # This is getting complicated. Let's just compute it directly for n=400.
    
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


def _hard(h):
    s = h.sum()
    if abs(s) < 1e-9:
        return 1e12
    conv = _autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _dct_to_h(coeffs):
    """Map DCT coefficients to a 400-length step-height vector.
    coeffs[0] is the DC component, coeffs[1:] are AC components.
    We enforce zero-mean by subtracting the mean, then adding a baseline.
    """
    n = len(coeffs)
    h = idct(coeffs, type=2, norm='ortho')
    # h has length n
    # Enforce zero-mean for the oscillatory part
    h = h - h.mean()
    return h


def _param_to_h(p):
    """Map parameter vector to a 400-length step-height vector.
    p = [baseline, dct_coeff_1, dct_coeff_2, ..., dct_coeff_m]
    where m is the number of AC DCT coefficients to use.
    """
    baseline = p[0]
    m = len(p) - 1
    # Create full DCT coefficient vector of length N
    coeffs = np.zeros(N)
    coeffs[0] = 0.0  # DC is handled by baseline
    coeffs[1:m+1] = p[1:m+1]
    # Inverse DCT to get oscillatory part
    h_osc = idct(coeffs, type=2, norm='ortho')
    # h_osc should be zero-mean if coeffs[0]=0, but let's enforce it
    h_osc = h_osc - h_osc.mean()
    # Add baseline
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
    DEADLINE = 130.0  # wall-clock budget; keep margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    rng = np.random.default_rng(12345)

    # Number of DCT coefficients to use for the oscillatory part
    m_dct = 20

    # ---- Phase 1: optimize the DCT parametric family ----
    # Initial guesses: various baseline values and random DCT coefficients
    best_h = np.ones(N, dtype=float)
    best_val = _hard(best_h)

    # Try different baseline values and random DCT coefficient patterns
    baselines = [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]
    for baseline in baselines:
        if time_left() < 30.0:
            break
        # Random DCT coefficients
        coeffs = rng.normal(0, 0.5, size=m_dct)
        p0 = np.concatenate([[baseline], coeffs])
        
        try:
            res = minimize(_param_obj, p0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 4000})
            h = _param_to_h(res.x)
            v = _hard(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        except Exception:
            pass

    # ---- Phase 2: full 400-dim polish with smooth surrogate (FFT gradient) ----
    h_curr = best_h.copy()
    # normalize sum to 1 to keep objective well-scaled
    s = h_curr.sum()
    if abs(s) > 1e-9:
        h_curr = h_curr / s

    beta_seq = [8.0, 16.0, 32.0, 64.0, 128.0]
    for beta in beta_seq:
        if time_left() < 25.0:
            break
        try:
            def obj(x):
                val, _ = _obj_and_grad(x, beta)
                return val
            def gr(x):
                _, g = _obj_and_grad(x, beta)
                return g
            res = minimize(obj, h_curr, jac=gr, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12,
                                    'maxfun': 6000})
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
    while time_left() > 8.0 and iters < 60:
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
        # value = 2N*maxc / ss^2 ; gradient direction (for maxc, ignore abs sign flip)
        # We want to reduce maxc relative to ss^2. Take a step that lowers maxc
        # while keeping sum positive: step = -alpha*(sg - (ss/n)*ones) to keep sum const.
        sg_centered = sg - sg.mean()
        for alpha in [1e-3, 1e-4, 1e-5, 1e-2]:
            cand = h_curr - alpha * sg_centered
            vv = _hard(cand)
            if vv < best_local:
                best_local = vv
                h_curr = cand
                if vv < best_val:
                    best_val = vv
                    best_h = cand.copy()
                break

    # ---- Final: a few L-BFGS-B runs from best with high beta for crisp max ----
    if time_left() > 10.0:
        for beta in [128.0, 256.0, 512.0]:
            if time_left() < 6.0:
                break
            try:
                def obj(x):
                    val, _ = _obj_and_grad(x, beta)
                    return val
                def gr(x):
                    _, g = _obj_and_grad(x, beta)
                    return g
                res = minimize(obj, best_h, jac=gr, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxfun': 8000})
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
