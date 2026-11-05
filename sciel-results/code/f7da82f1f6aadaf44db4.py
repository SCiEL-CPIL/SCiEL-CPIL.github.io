"""Combined program: Zadoff-Chu windowed + DCT spectral shaping + chirp + smooth-max refinement.

Strategy:
- Primary: Windowed Zadoff-Chu sequences (smooth window to suppress sidelobes)
  with global optimization over window width, center, rate, amplitude, phase, base.
- Secondary: DCT spectral shaping with projected gradient descent.
- Tertiary: Chirp-based parameterization.
- Refine with L-BFGS-B on DFT basis using smooth max.
- Deadline guard ensures we return before 150s.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_K_DCT = 30  # number of DCT basis vectors for spectral shaping
_K_DFT = 15  # number of DFT frequency components for refinement

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=50.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    # Smooth max via log-sum-exp (scaled)
    m = np.max(conv)
    if m <= 0:
        smooth_m = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    else:
        smooth_m = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
    return abs(2.0 * N * smooth_m / (s * s))

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    # DCT-II: basis[k, n] = cos(pi * (2*n+1) * k / (2*N))
    # We want shape (N, K) so that h = basis @ coeffs
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    # Normalize columns
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)  # (N, K_DCT)
_DCT_BASIS_T = _DCT_BASIS.T  # (K_DCT, N)

def _dct_h(coeffs):
    """Generate h from DCT coefficients. coeffs shape (K_DCT,)"""
    h = _DCT_BASIS @ coeffs
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _value(h)

def _dft_smooth_obj(coeffs_base, beta=50.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu: h[i] = base + amp * window[i] * cos(2*pi*rate*i*(i+1)/N + phase).
    
    params: [phase, rate, amp, base, win_width, win_center, win_type]
    - win_width: relative width of window (0.1 to 1.5), controls Gaussian sigma or Hann extent
    - win_center: center of window in [0, 1] (relative to N)
    - win_type: 0=Hann, 1=Gaussian
    """
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = np.arange(N, dtype=float)
    # Zadoff-Chu quadratic phase
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    
    # Apply smooth window to suppress sidelobes
    rel_pos = i / N  # [0, 1]
    if win_type < 0.5:
        # Hann window centered at win_center with width win_width
        # Hann: 0.5*(1+cos(pi*x)) for x in [-1,1], 0 outside
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        # Gaussian window
        sigma = win_width / 4.0  # FWHM ~ 2.355*sigma
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _zadoff_chu_h(params):
    """Zadoff-Chu inspired sequence: h[i] = base + amp * cos(2*pi*rate*i*(i+1)/N + phase)."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Zadoff-Chu uses quadratic phase with specific structure
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_obj(params):
    """Objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _value(h)

def _zadoff_chu_smooth_obj(params, beta=50.0):
    """Smooth objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase + rate * (i/N)^2
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _chirp_smooth_obj(params, beta=50.0):
    """Smooth objective for chirp parameters."""
    h = _chirp_h(params)
    return _smooth_max_value(h, beta)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K_DFT)

def _dft_h(coeffs, base_val):
    """Generate h from DFT coefficients and a base value."""
    # coeffs shape: (2*K,), base_val is scalar
    h = base_val + _BASIS_T @ coeffs
    return h

def _dft_obj(coeffs_base):
    """Objective for DFT parameters: [coeffs..., base]."""
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
    
    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with projected gradient descent
    # This is the primary structural change: work in DCT domain directly
    if deadline_ok():
        try:
            # Multiple restarts in DCT coefficient space
            rng = np.random.default_rng(123)
            for restart in range(8):
                if not deadline_ok():
                    break
                # Initialize DCT coefficients: sparse random
                coeffs = np.zeros(_K_DCT)
                # Add some random low-frequency energy
                n_active = rng.integers(3, min(12, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                for idx in active_idx:
                    coeffs[idx] = rng.normal(0, 1.0)
                
                # Projected gradient descent with adaptive step size
                lr = 0.1
                best_coeffs = coeffs.copy()
                best_smooth = _dct_smooth_obj(coeffs, beta=50.0)
                
                for step in range(200):
                    if not deadline_ok():
                        break
                    # Compute gradient numerically (central difference for accuracy)
                    eps = 1e-5
                    grad = np.zeros(_K_DCT)
                    for j in range(_K_DCT):
                        cp = coeffs.copy()
                        cm = coeffs.copy()
                        cp[j] += eps
                        cm[j] -= eps
                        grad[j] = (_dct_smooth_obj(cp, beta=50.0) - _dct_smooth_obj(cm, beta=50.0)) / (2 * eps)
                    
                    # Gradient step
                    coeffs_new = coeffs - lr * grad
                    
                    # Project: ensure sum(h) != 0 by adjusting DC component
                    h_new = _dct_h(coeffs_new)
                    s_new = np.sum(h_new)
                    if abs(s_new) < 0.1:
                        # Adjust first DCT coefficient (DC-like) to ensure nonzero sum
                        coeffs_new[0] += (0.5 - s_new) / N
                    
                    h_new = _dct_h(coeffs_new)
                    smooth_val = _dct_smooth_obj(coeffs_new, beta=50.0)
                    
                    if smooth_val < best_smooth:
                        best_smooth = smooth_val
                        best_coeffs = coeffs_new.copy()
                        coeffs = coeffs_new
                        # Decrease learning rate slowly
                        lr *= 0.999
                    else:
                        # Reduce learning rate on failure
                        lr *= 0.9
                        if lr < 1e-6:
                            break
                    
                    # Update best if actual value is good
                    h_check = _dct_h(coeffs_new)
                    if deadline_ok():
                        v_check = _value(h_check)
                        if v_check < best_v:
                            update_best(h_check)
                
                # Final evaluation of best DCT solution
                h_final = _dct_h(best_coeffs)
                update_best(h_final)
                
        except Exception:
            pass
    
    # Phase 2: Global search via differential evolution on chirp parameters
    if deadline_ok():
        try:
            bounds = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _chirp_obj,
                bounds,
                seed=42,
                maxiter=40,
                tol=1e-6,
                popsize=8,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res.x is not None:
                h = _chirp_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu global search (primary new strategy)
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),       # phase
                (-50.0, 50.0),         # rate
                (-5.0, 5.0),           # amp
                (-5.0, 5.0),           # base
                (0.1, 1.5),            # win_width
                (0.2, 0.8),            # win_center
                (0.0, 1.0),            # win_type (0=Hann, 1=Gaussian)
            ]
            res_wzc = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=77,
                maxiter=60,
                tol=1e-6,
                popsize=12,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_wzc.x is not None:
                h = _windowed_zadoff_chu_h(res_wzc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3b: Unwindowed Zadoff-Chu (keep for diversity)
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),
                (-10.0, 10.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
            ]
            res_zc = differential_evolution(
                _zadoff_chu_obj,
                bounds_zc,
                seed=99,
                maxiter=20,
                tol=1e-6,
                popsize=8,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_zc.x is not None:
                h = _zadoff_chu_h(res_zc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: L-BFGS-B refinement on DFT basis with smooth max
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K_DFT)
            for k in range(1, _K_DFT + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                cos_k = np.cos(angle)
                sin_k = np.sin(angle)
                A_k = np.column_stack([cos_k, sin_k])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            # Use smooth objective for L-BFGS-B
            res3 = minimize(
                lambda x: _dft_smooth_obj(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res3.x is not None:
                h = _dft_h(res3.x[:-1], res3.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 4b: L-BFGS-B refinement on windowed Zadoff-Chu parameters
    if deadline_ok():
        try:
            # Initialize from best windowed ZC if available, else reasonable defaults
            x0_wzc = np.array([0.0, 10.0, 1.0, 0.5, 0.8, 0.5, 0.0])
            
            res_wzc_refine = minimize(
                lambda x: _windowed_zc_smooth_obj(x, beta=80.0),
                x0_wzc,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res_wzc_refine.x is not None:
                h = _windowed_zadoff_chu_h(res_wzc_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            # Project best_h onto DCT basis
            h_best = best_h
            # Use scipy's DCT to get coefficients
            dct_coeffs = scipy_dct(h_best, type=2, n=_K_DCT, norm='ortho')
            
            def _dct_obj_wrapper(coeffs):
                h = _dct_h(coeffs)
                return _dft_smooth_obj(np.concatenate([np.zeros(2 * _K_DFT), [0.0]]), beta=50.0) if False else _smooth_max_value(h, beta=50.0)
            
            res5 = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=50.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res5.x is not None:
                h = _dct_h(res5.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: Final Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            h_best = best_h
            mean_h = np.mean(h_best)
            coeffs_init = np.zeros(2 * _K_DFT)
            for k in range(1, _K_DFT + 1):
                angle = 2.0 * np.pi * k * _n_idx / N
                cos_k = np.cos(angle)
                sin_k = np.sin(angle)
                A_k = np.column_stack([cos_k, sin_k])
                target = h_best - mean_h
                try:
                    sol = np.linalg.lstsq(A_k, target, rcond=None)[0]
                    coeffs_init[2 * (k - 1)] = sol[0]
                    coeffs_init[2 * (k - 1) + 1] = sol[1]
                except:
                    pass
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res4 = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 400, 'maxfev': 800},
            )
            if res4.x is not None:
                h = _dft_h(res4.x[:-1], res4.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
