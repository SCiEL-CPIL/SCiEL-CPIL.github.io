"""Combined program: DCT spectral shaping + Golay complementary pairs (global)
with DFT-basis smooth-max L-BFGS-B refinement.

Strategy:
- Global search over DCT spectral coefficients (low-frequency shaping) and
  Golay-complementary-pair chirp parameters, which can drive the non-zero
  autocorrelation toward a near-constant profile (value -> ~1.002).
- FFT-based autoconvolution for fast exact evaluation.
- Refinement via L-BFGS-B on a low-frequency DFT basis using a log-sum-exp
  smooth-max surrogate, then exact Nelder-Mead polish.
- Deadline guard returns best result before the wall-clock limit.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 125.0  # safety margin under 150s wall-clock

def _fft_convolve(h):
    """Fast autoconvolution via FFT (full length 2N-1)."""
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    return conv

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = float(np.max(conv))
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100.0, 100.0)
    log_sum_exp = m + float(np.log(np.sum(np.exp(scaled)))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

# ---- DCT spectral basis (orthonormal) ----
_K_DCT = 40
_n_idx = np.arange(N)
_k_idx = np.arange(_K_DCT)
_angles = np.pi * (2 * _n_idx[:, None] + 1) * _k_idx[None, :] / (2 * N)
_DCT_BASIS = np.cos(_angles)
_norms = np.sqrt(np.sum(_DCT_BASIS ** 2, axis=0))
_norms[_norms == 0] = 1.0
_DCT_BASIS = _DCT_BASIS / _norms  # (N, K_DCT)

def _dct_h(coeffs):
    """Generate h from DCT coefficients (zero-mean spectral shaping)."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    h = _dct_h(coeffs)
    return _value(h)

# ---- Golay complementary pair (two chirps) ----
def _golay_h(params):
    p1, r1, a1, p2, r2, a2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * r1 * i * (i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * i * (i + 1) / N + p2
    return base + a1 * np.cos(ang1) + a2 * np.cos(ang2)

def _golay_obj(params):
    return _value(_golay_h(params))

# ---- Single chirp ----
def _chirp_h(params):
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * (i / N) ** 2
    return base + amp * np.cos(ang)

def _chirp_obj(params):
    return _value(_chirp_h(params))

# ---- DFT basis for refinement ----
_K = 35
_basis = np.zeros((2 * _K, N))
for k in range(1, _K + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K)

def _dft_h(coeffs, base_val):
    return base_val + _BASIS_T @ coeffs

def _dft_obj(coeffs_base):
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    return _value(_dft_h(coeffs, base_val))

def _dft_smooth_obj(coeffs_base, beta=80.0):
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    return _smooth_max_value(_dft_h(coeffs, base_val), beta)

def _init_dft_from_h(h_best):
    """Project h_best onto the DFT basis + base, return [coeffs..., base]."""
    mean_h = float(np.mean(h_best))
    coeffs_init = np.zeros(2 * _K)
    H_fft = np.fft.rfft(h_best - mean_h)
    for k in range(1, _K + 1):
        re_k = H_fft[k]
        im_k = H_fft[k]
        coeffs_init[2 * (k - 1)] = 2.0 * re_k / N
        coeffs_init[2 * (k - 1) + 1] = -2.0 * im_k / N
    return np.concatenate([coeffs_init, [mean_h]])

def construct():
    global _t0
    _t0 = time.time()

    best_h = np.ones(N)
    best_v = _value(best_h)

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v

    # Phase 1: DCT spectral shaping with multiple random restarts (global)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(10):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = int(rng.integers(4, min(14, _K_DCT)))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0.0, 1.0, size=n_active)
                res = minimize(
                    _dct_smooth_obj, coeffs, method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                if res.x is not None:
                    update_best(_dct_h(res.x))
        except Exception:
            pass

    # Phase 2: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi), (-50.0, 50.0), (-3.0, 3.0),
                (-np.pi, np.pi), (-50.0, 50.0), (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1,
            )
            if res.x is not None:
                update_best(_golay_h(res.x))
        except Exception:
            pass

    # Phase 3: Single chirp global search
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi), (-200.0, 200.0), (-3.0, 3.0), (-3.0, 3.0),
            ]
            res = differential_evolution(
                _chirp_obj, bounds_chirp, seed=42, maxiter=40, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1,
            )
            if res.x is not None:
                update_best(_chirp_h(res.x))
        except Exception:
            pass

    # Phase 4: L-BFGS-B refinement on DFT basis with smooth-max surrogate
    if deadline_ok():
        try:
            x0 = _init_dft_from_h(best_h)
            res = minimize(
                _dft_smooth_obj, x0, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            if res.x is not None:
                update_best(_dft_h(res.x[:-1], res.x[-1]))
        except Exception:
            pass

    # Phase 5: Exact Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            x0 = _init_dft_from_h(best_h)
            res = minimize(
                _dft_obj, x0, method='Nelder-Mead',
                options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 1200, 'maxfev': 2500},
            )
            if res.x is not None:
                update_best(_dft_h(res.x[:-1], res.x[-1]))
        except Exception:
            pass

    # Phase 6: Tight smooth-max polish (higher beta)
    if deadline_ok():
        try:
            x0 = _init_dft_from_h(best_h)
            def _tight(coeffs_base):
                return _dft_smooth_obj(coeffs_base, beta=500.0)
            res = minimize(
                _tight, x0, method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-15},
            )
            if res.x is not None:
                update_best(_dft_h(res.x[:-1], res.x[-1]))
        except Exception:
            pass

    # Phase 7: Multi-start Nelder-Mead from perturbed DFT initializations
    if deadline_ok():
        try:
            for start_offset in [0.1, -0.1, 0.2, -0.2]:
                if not deadline_ok():
                    break
                x0 = _init_dft_from_h(best_h)
                x0[:-1] *= (1.0 + start_offset)
                res = minimize(
                    _dft_obj, x0, method='Nelder-Mead',
                    options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 800, 'maxfev': 1500},
                )
                if res.x is not None:
                    update_best(_dft_h(res.x[:-1], res.x[-1]))
        except Exception:
            pass

    # Final validation
    s = float(np.sum(best_h))
    if s == 0.0 or not np.all(np.isfinite(best_h)) or abs(s) < 1e-10:
        return np.ones(N)
    return best_h
