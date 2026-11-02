import time
import numpy as np
from scipy.optimize import minimize

N = 600
HALF = N // 2          # 300, cosine basis size
DEADLINE = 130.0       # safety margin under the 150 s kill

# Precompute the cosine (DCT-II) basis so that f = A @ coeffs.
# DCT-II row i, col k: cos(pi*(2*i+1)*k/(2*M))
_ii = np.arange(N)
_kk = np.arange(HALF)
_COS = np.cos(np.pi * np.outer(2.0 * _ii + 1.0, _kk) / (2.0 * HALF))
_NORM = np.sqrt(2.0 / N)
_COS *= _NORM
# Make the transform orthonormal: f = _COS @ coeffs.
COS = _COS

def _autoconv_peak(f):
    """Peak of the linear (full) autoconvolution of f, via FFT."""
    n = N
    m = 2 * n - 1
    size = 1 << (m - 1).bit_length()
    F = np.fft.rfft(f, size)
    auto = np.fft.irfft(F * F, size)[:m]
    return float(auto.max())

def _objective(c, lam):
    f = COS @ c
    neg = f[f < 0.0]
    peak = _autoconv_peak(f)
    pen = lam * float(np.sum(neg * neg))
    return peak + pen

def _build(c, lam):
    f = COS @ c
    f = np.maximum(f, 0.0)
    s = f.sum()
    if s > 0.0:
        f = f / s
    return f

def _try_start(c0, lam, t0):
    res = minimize(
        _objective, c0, args=(lam,),
        method='L-BFGS-B',
        options={'maxiter': 300, 'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-9},
    )
    c = res.x
    f = _build(c, lam)
    return _autoconv_peak(f), f

def construct():
    t0 = time.time()

    starts = []
    # 1) flat (matches seed)
    starts.append(np.ones(HALF))
    # 2) low-frequency concentration (smooth, positive)
    low = np.zeros(HALF)
    low[:4] = 1.0
    starts.append(low)
    # 3) a couple of random non-negative starts
    rng = np.random.default_rng(12345)
    for _ in range(2):
        starts.append(rng.random(HALF))

    best_val = _autoconv_peak(_build(starts[0], 1.0))
    best_f = _build(starts[0], 1.0)

    for c0 in starts:
        if time.time() - t0 > DEADLINE:
            break
        try:
            val, f = _try_start(c0, 1.0, t0)
        except Exception:
            continue
        if val < best_val:
            best_val = val
            best_f = f

    # Optional: one refinement pass on the best with a larger penalty, if time allows.
    if time.time() - t0 < DEADLINE - 5.0:
        c_best = np.linalg.lstsq(COS, best_f, rcond=None)[0]
        try:
            val, f = _try_start(c_best, 5.0, t0)
        except Exception:
            val, f = best_val, best_f
        if val < best_val:
            best_val = val
            best_f = f

    return [float(x) for x in best_f]
