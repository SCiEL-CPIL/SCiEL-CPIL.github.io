"""Rewritten program: Thue-Morse-derived smooth signed sequence + smooth min-max polish.

Objective (lower is better):  C3' = max_k |autoconv(x)[k]| / (sum x)^2
Hard lower bound is 1/N (attained only by the constant sequence).  We build a
smooth, near-constant signed sequence from a morphic (Thue-Morse) low-discrepancy
source, then run a smooth min-max (log-sum-exp / epigraph) local polish to reduce
the residual autoconvolution peak while preserving the integral.
"""

import time
import numpy as np

N = 400
# Wall-clock guard with a safety margin (evaluator kills at 150 s).
_DEADLINE = time.time() + 140.0
_EPS = 1e-12


def _remaining():
    return _DEADLINE - time.time()


def _autoconv(x):
    """Circular autoconvolution via FFT (real, symmetric)."""
    n = x.size
    X = np.fft.rfft(x)
    r = np.fft.irfft(X * np.conj(X), n=n)
    return r


def _integral(x):
    return float(np.sum(x))


def _score(x):
    """C3' = max |autoconv| / (sum x)^2  (lower is better)."""
    s = _integral(x)
    if abs(s) < _EPS:
        return np.inf
    r = _autoconv(x)
    return float(np.max(np.abs(r))) / (s * s)


def _thue_morse(m):
    """Thue-Morse bits of length m: t_k = parity(popcount(k))."""
    k = np.arange(m, dtype=np.int64)
    # parity of popcount via bit tricks
    k = k ^ (k >> 1)
    k = (k & 0x5555555555555555) + ((k >> 1) & 0x5555555555555555)
    k = (k & 0x3333333333333333) + ((k >> 2) & 0x3333333333333333)
    k = (k & 0x0F0F0F0F0F0F0F0F) + ((k >> 4) & 0x0F0F0F0F0F0F0F0F)
    k = (k & 0x00FF00FF00FF00FF) + ((k >> 8) & 0x00FF00FF00FF00FF)
    k = (k & 0x0000FFFF0000FFFF) + ((k >> 16) & 0x0000FFFF0000FFFF)
    k = (k & 0x00000000FFFFFFFF) + ((k >> 32) & 0x00000000FFFFFFFF)
    return (k & 1).astype(np.float64)


def _build_smooth_signed():
    """Build a smooth signed near-constant sequence from a morphic source.

    We take the Thue-Morse bit pattern, map {0,1}->{-1,+1}, then apply a strong
    low-pass (Gaussian) filter so the result is a smooth, low-frequency signed
    signal.  Low-frequency energy => slowly varying, low-peak autoconvolution.
    We then re-center/normalize so the integral is positive and O(N).
    """
    tm = _thue_morse(N)
    # signed: 0 -> -1, 1 -> +1
    sig = 2.0 * tm - 1.0
    # Gaussian low-pass in the Fourier domain keeps only low modes
    f = np.fft.rfftfreq(N)
    sigma = max(2.0, N / 12.0)          # width of the low-pass window
    w = np.exp(-(f * N) ** 2 / (2.0 * sigma ** 2))
    w[0] = 0.0                          # remove DC so the signal is signed/zero-mean-ish
    X = np.fft.rfft(sig) * w
    x = np.fft.irfft(X, n=N)
    # Make it near-constant positive with a positive integral:
    # add a constant offset so mean is +1 (integral = N), preserving smoothness.
    x = x - x.mean() + 1.0
    # Clip tiny negatives to keep a clean signed step profile but allow signs.
    return x


def _smooth_minmax_polish(x, budget_s):
    """Smooth min-max (log-sum-exp) local polish.

    Minimize  L = logsumexp(|r|/tau) + beta*log(sum x^2)  (proxy for max|r|/S^2),
    with r = autoconv(x).  Gradient via FFT.  Project onto the affine set
    {sum x = const} to keep the integral fixed (so S^2 is constant and we
    directly minimize the peak).  Uses a decreasing tau (soft->hard max) and a
    small adaptive step.  Returns the best (lowest raw C3') iterate seen.
    """
    best_x = x.copy()
    best_val = _score(best_x)
    s0 = _integral(x)

    def grad_L(xk, tau, beta):
        r = _autoconv(xk)
        a = np.abs(r)
        # stable logsumexp of a/tau
        m = a.max()
        z = np.exp((a - m) / tau)
        lse = m / tau + np.log(z.sum())
        # d/dx of logsumexp(|r|/tau):  r is symmetric real, dr_i/dx_j = x_{i-j}
        w = z / z.sum() * (1.0 / tau)
        sign_r = np.sign(r)
        # grad_i = sum_j w_j sign(r_j) (dr_j/dx_i) ; dr_j/dx_i = x_{j-i} (circular)
        # Build via FFT: grad = ifft( conj(FFT(w*sign)) * ... ) -- use direct correlation
        # grad_i = sum_j g_j * x_{j - i}  where g_j = w_j*sign(r_j)
        # This is a circular cross-correlation.  Use FFT.
        G = np.fft.fft(w * sign_r)
        Xf = np.fft.fft(xk)
        grad = np.real(np.fft.ifft(G * Xf))
        # beta term: d/dx beta*log(sum x^2) = 2 beta x / sum x^2
        ssq = float(np.sum(xk * xk)) + _EPS
        grad = grad + (2.0 * beta * xk) / ssq
        return grad, lse

    tau = max(0.05, 0.2 * (np.max(np.abs(_autoconv(x))) + _EPS))
    beta = 1e-4
    step = 0.02
    it = 0
    while _remaining() > budget_s:
        it += 1
        g, _ = grad_L(x, tau, beta)
        # project gradient onto tangent of {sum x = const}: remove mean
        g = g - g.mean()
        x_new = x - step * g
        # re-center to keep integral exactly s0
        x_new = x_new + (s0 - _integral(x_new)) / N
        # light amplitude guard to avoid blow-up
        if not np.all(np.isfinite(x_new)):
            break
        v = _score(x_new)
        if v < best_val - 1e-12:
            best_x = x_new.copy()
            best_val = v
            x = x_new
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.8
            if step < 1e-5:
                # anneal tau to sharpen the max
                tau *= 0.85
                step = 0.05
        if tau < 1e-3:
            break
    return best_x, best_val, it


def construct():
    t_start = time.time()
    # 1) deterministic smooth signed base from Thue-Morse morphic source
    x = _build_smooth_signed()
    best_x = x.copy()
    best_val = _score(best_x)

    # 2) smooth min-max polish, leaving a hard safety margin
    polish_budget = 10.0  # seconds
    if _remaining() > polish_budget + 2.0:
        px, pv, _ = _smooth_minmax_polish(x, polish_budget)
        if pv < best_val:
            best_x, best_val = px, pv

    # 3) final exact re-centering + a couple of tiny gradient steps if time remains
    if _remaining() > 3.0:
        s0 = _integral(best_x)
        best_x = best_x + (s0 - _integral(best_x)) / N

    # Guard: if anything is non-finite or integral ~0, fall back to constant.
    if (not np.all(np.isfinite(best_x))) or abs(_integral(best_x)) < _EPS:
        return np.ones(N)

    return best_x
