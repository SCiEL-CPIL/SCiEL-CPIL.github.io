"""Hermite-Gaussian tuning for the L4 uncertainty constant C4.

We minimize the test-function upper bound
    C4_upper = ||f||_4 * ||F[f]||_4 / ||f||_2
for f(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) e^{-x^2/2},
with c12 fixed by g(0)=0.  Lower is better.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

_DEADLINE = 180.0 - 30.0  # safety margin below the 210s kill / 180s limit

# Precompute a fine integration grid for spatial domain
def _build_grid(n=8192, half=16.0):
    x = np.linspace(-half, half, n)
    dx = x[1] - x[0]
    return x, dx

_X, _DX = _build_grid()
_GAUSS = np.exp(-0.5 * _X * _X)
# Hermite values on the grid
_H = np.stack([eval_hermite(n, _X) for n in (0, 4, 8, 12)], axis=1)  # (N,4)
_H00 = np.array([eval_hermite(0, 0.0), eval_hermite(4, 0.0),
                 eval_hermite(8, 0.0), eval_hermite(12, 0.0)])

def _coeffs12(c0, c4, c8):
    c12 = -(c0 * _H00[0] + c4 * _H00[1] + c8 * _H00[2]) / _H00[3]
    return c12

def _compute_c4_upper(c):
    """Compute the upper bound C4 = ||f||_4 * ||F[f]||_4 / ||f||_2"""
    c0, c4, c8 = c
    c12 = _coeffs12(c0, c4, c8)
    
    # f(x) = poly(x) * exp(-x^2/2)
    poly = c0 * _H[:, 0] + c4 * _H[:, 1] + c8 * _H[:, 2] + c12 * _H[:, 3]
    f = poly * _GAUSS
    
    # ||f||_2^2 = ∫|f|² dx
    n2_sq = np.sum(f**2) * _DX
    if n2_sq <= 0 or not np.isfinite(n2_sq):
        return 1e300
    n2 = np.sqrt(n2_sq)
    
    # ||f||_4^4 = ∫|f|⁴ dx
    n4_4 = np.sum(f**4) * _DX
    if n4_4 <= 0 or not np.isfinite(n4_4):
        return 1e300
    n4 = n4_4 ** 0.25
    
    # Compute Fourier transform using FFT
    # F[f](ξ) = ∫ f(x) e^{-2πixξ} dx
    # Using FFT: need proper scaling
    n = len(_X)
    dx = _DX
    F_f = np.fft.fftshift(np.fft.fft(f)) * dx
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    
    # ||F[f]||_4^4 = ∫|F[f]|⁴ dξ
    dxi = freqs[1] - freqs[0] if n > 1 else 1.0
    n4_freq_4 = np.sum(np.abs(F_f)**4) * dxi
    if n4_freq_4 <= 0 or not np.isfinite(n4_freq_4):
        return 1e300
    n4_freq = n4_freq_4 ** 0.25
    
    # C4_upper = ||f||_4 * ||F[f]||_4 / ||f||_2
    C4 = (n4 * n4_freq) / n2
    return C4

def _safe_objective(c):
    try:
        v = _compute_c4_upper(c)
        if not np.isfinite(v):
            return 1e300
        return v
    except Exception:
        return 1e300

def construct():
    t_start = time.time()
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_v = _safe_objective(best_c)
    
    def remaining():
        return _DEADLINE - (time.time() - t_start)
    
    # 1) Global search with differential evolution
    if remaining() > 30:
        try:
            bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
            res = differential_evolution(
                _safe_objective, bounds,
                seed=42, maxiter=80, popsize=15,
                tol=1e-10, polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res.x is not None and np.isfinite(res.fun) and res.fun < best_v:
                best_v = float(res.fun)
                best_c = np.asarray(res.x, dtype=float)
        except Exception:
            pass
    
    # 2) Local polish from multiple starts
    starts = [best_c.copy(),
              np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0]),
              np.array([-1.0, 0.05, 0.001]),
              np.array([-1.5, 0.08, 0.0005]),
              np.array([-0.5, 0.02, 0.002]),
              np.array([-2.0, 0.1, 0.0003])]
    
    for s in starts:
        if remaining() < 10:
            break
        try:
            r = minimize(_safe_objective, s, method='Nelder-Mead',
                         options={'xatol': 1e-12, 'fatol': 1e-12,
                                  'maxiter': 500, 'adaptive': True})
            if r.x is not None and np.isfinite(r.fun) and r.fun < best_v:
                best_v = float(r.fun)
                best_c = np.asarray(r.x, dtype=float)
        except Exception:
            pass
    
    # 3) Final fine polish
    if remaining() > 5:
        try:
            r = minimize(_safe_objective, best_c, method='Nelder-Mead',
                         options={'xatol': 1e-14, 'fatol': 1e-14,
                                  'maxiter': 1000, 'adaptive': True})
            if r.x is not None and np.isfinite(r.fun) and r.fun < best_v:
                best_v = float(r.fun)
                best_c = np.asarray(r.x, dtype=float)
        except Exception:
            pass
    
    return np.array([best_c[0], best_c[1], best_c[2]])
# EVOLVE-BLOCK-END
