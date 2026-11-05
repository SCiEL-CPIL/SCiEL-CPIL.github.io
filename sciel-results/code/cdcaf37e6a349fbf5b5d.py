import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    # Use FFT for speed
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _smooth_max_value(h, alpha=50.0):
    """Smooth surrogate for the max of convolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    scaled = conv * alpha
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / alpha
    return abs(2.0 * N * soft_max / (s * s))

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_h(params):
    """Generate h from Zadoff-Chu-like parameters."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * i * (i - N / 2.0) / N
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    return _value(_chirp_h(params))

def _zadoff_chu_obj(params):
    return _value(_zadoff_chu_h(params))

# Low-frequency DFT basis for refinement
_K = 20
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K, N))
for k in range(1, _K + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T

def _dft_h(coeffs, base_val):
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    # Phase 1: Global search on chirp
    bounds = [
        (-np.pi, np.pi),
        (-200.0, 200.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
    ]
    try:
        if deadline_ok():
            res = differential_evolution(
                _chirp_obj, bounds, seed=42, maxiter=50, tol=1e-6, popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
    except Exception:
        pass

    # Phase 2: Global search on ZC
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_zc = differential_evolution(
                _zadoff_chu_obj, bounds_zc, seed=123, maxiter=40, tol=1e-6, popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res_zc.x is not None:
                h = _zadoff_chu_h(res_zc.x)
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B on DFT basis with smooth max
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                A_k = np.column_stack([np.cos(angle), np.sin(angle)])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            def _dft_smooth_obj(coeffs_base):
                coeffs = coeffs_base[:-1]
                base_val = coeffs_base[-1]
                h = _dft_h(coeffs, base_val)
                return _smooth_max_value(h, alpha=100.0)
            
            res3 = minimize(_dft_smooth_obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res3.x is not None:
                h = _dft_h(res3.x[:-1], res3.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass

    # Phase 4: Nelder-Mead on DFT
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K)
            for k in range(1, _K + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                A_k = np.column_stack([np.cos(angle), np.sin(angle)])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res4 = minimize(_dft_obj, x0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000})
            if res4.x is not None:
                h = _dft_h(res4.x[:-1], res4.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
