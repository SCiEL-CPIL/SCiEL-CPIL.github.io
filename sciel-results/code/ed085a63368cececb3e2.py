"""Combined program: Windowed Zadoff-Chu + Golay-inspired complementary pair + DCT spectral shaping + refinement.

Strategy:
- Primary: Windowed Zadoff-Chu sequences with global optimization over window params.
- Secondary: Golay complementary pair-inspired construction where h = a + b,
  and a, b are optimized to minimize max of combined autoconvolution.
- Tertiary: DCT spectral shaping with projected gradient descent.
- Refine with L-BFGS-B on DFT basis using smooth max.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock
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
    m = np.max(conv)
    # Numerically stable log-sum-exp
    shifted = beta * (conv - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    smooth_m = m + np.log(np.sum(np.exp(shifted))) / beta
    return abs(2.0 * N * smooth_m / (s * s))

def _dct_basis_matrix(K, N):
    """Build DCT-II basis matrix of shape (N, K) for first K modes."""
    n = np.arange(N)
    k = np.arange(K)
    angles = np.pi * (2 * n[:, None] + 1) * k[None, :] / (2 * N)
    basis = np.cos(angles)
    norms = np.sqrt(np.sum(basis ** 2, axis=0))
    norms[norms == 0] = 1.0
    basis = basis / norms
    return basis

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)  # (N, K_DCT)

def _dct_h(coeffs):
    """Generate h from DCT coefficients. coeffs shape (K_DCT,)"""
    h = _DCT_BASIS @ coeffs
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
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
    
    params: [phase, rate, amp, base, win_width, win_center]
    - win_width: relative width of window (0.1 to 1.5)
    - win_center: center of window in [0, 1]
    """
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    
    rel_pos = i / N
    x = (rel_pos - win_center) / (0.5 * max(win_width, 0.01))
    window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    
    h = base + amp * window * np.cos(ang)
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _golay_pair_h(params):
    """Golay complementary pair inspired: h = a + b where a and b are chirps with
    complementary phase relationships.
    
    params: [phase_a, rate_a, amp_a, phase_b, rate_b, amp_b, base]
    a[i] = amp_a * cos(2*pi*rate_a*i/N + phase_a)
    b[i] = amp_b * cos(2*pi*rate_b*i/N + phase_b)
    h[i] = base + a[i] + b[i]
    
    The complementary property: if a and b form a Golay pair, then
    R_a(lag) + R_b(lag) = 0 for all nonzero lags, which minimizes sidelobes.
    """
    phase_a, rate_a, amp_a, phase_b, rate_b, amp_b, base = params
    i = np.arange(N, dtype=float)
    ang_a = 2.0 * np.pi * rate_a * i / N + phase_a
    ang_b = 2.0 * np.pi * rate_b * i / N + phase_b
    a = amp_a * np.cos(ang_a)
    b = amp_b * np.cos(ang_b)
    h = base + a + b
    return h

def _golay_pair_obj(params):
    """Objective for Golay pair parameters."""
    h = _golay_pair_h(params)
    return _value(h)

def _golay_pair_smooth_obj(params, beta=50.0):
    """Smooth objective for Golay pair parameters."""
    h = _golay_pair_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _zadoff_chu_h(params):
    """Zadoff-Chu inspired sequence."""
    phase, rate, amp, base = params
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    h = base + amp * np.cos(ang)
    return h

def _zadoff_chu_obj(params):
    """Objective for Zadoff-Chu parameters."""
    h = _zadoff_chu_h(params)
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
    
    # Phase 1: Windowed Zadoff-Chu global search (primary, proven effective)
    if deadline_ok():
        try:
            bounds_wz = [
                (-np.pi, np.pi),       # phase
                (-50.0, 50.0),        # rate
                (-3.0, 3.0),          # amp
                (-3.0, 3.0),          # base
                (0.1, 1.5),           # win_width
                (0.2, 0.8),           # win_center
            ]
            res = differential_evolution(
                _windowed_zc_obj,
                bounds_wz,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=10,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res.x is not None:
                h = _windowed_zadoff_chu_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair inspired search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),       # phase_a
                (0.5, 20.0),           # rate_a (positive freq)
                (-3.0, 3.0),           # amp_a
                (-np.pi, np.pi),       # phase_b
                (0.5, 20.0),           # rate_b
                (-3.0, 3.0),           # amp_b
                (-2.0, 2.0),           # base
            ]
            res_g = differential_evolution(
                _golay_pair_obj,
                bounds_golay,
                seed=77,
                maxiter=40,
                tol=1e-6,
                popsize=8,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_g.x is not None:
                h = _golay_pair_h(res_g.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Standard Zadoff-Chu search
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
                maxiter=30,
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
    
    # Phase 4: Chirp search
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_c = differential_evolution(
                _chirp_obj,
                bounds_chirp,
                seed=42,
                maxiter=30,
                tol=1e-6,
                popsize=8,
                init='sobol',
                polish=False,
                update_interval=1,
            )
            if res_c.x is not None:
                h = _chirp_h(res_c.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: DCT spectral shaping with projected gradient descent
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(6):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(3, min(12, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                for idx in active_idx:
                    coeffs[idx] = rng.normal(0, 1.0)
                
                lr = 0.1
                best_coeffs = coeffs.copy()
                best_smooth = _dct_smooth_obj(coeffs, beta=50.0)
                
                for step in range(150):
                    if not deadline_ok():
                        break
                    eps = 1e-5
                    grad = np.zeros(_K_DCT)
                    for j in range(_K_DCT):
                        cp = coeffs.copy()
                        cm = coeffs.copy()
                        cp[j] += eps
                        cm[j] -= eps
                        grad[j] = (_dct_smooth_obj(cp, beta=50.0) - _dct_smooth_obj(cm, beta=50.0)) / (2 * eps)
                    
                    coeffs_new = coeffs - lr * grad
                    h_new = _dct_h(coeffs_new)
                    s_new = np.sum(h_new)
                    if abs(s_new) < 0.1:
                        coeffs_new[0] += (0.5 - s_new) / N
                    h_new = _dct_h(coeffs_new)
                    smooth_val = _dct_smooth_obj(coeffs_new, beta=50.0)
                    
                    if smooth_val < best_smooth:
                        best_smooth = smooth_val
                        best_coeffs = coeffs_new.copy()
                        coeffs = coeffs_new
                        lr *= 0.999
                    else:
                        lr *= 0.9
                        if lr < 1e-6:
                            break
                    
                    h_check = _dct_h(coeffs_new)
                    if deadline_ok():
                        v_check = _value(h_check)
                        if v_check < best_v:
                            update_best(h_check)
                
                h_final = _dct_h(best_coeffs)
                update_best(h_final)
        except Exception:
            pass
    
    # Phase 6: L-BFGS-B refinement on DFT basis with smooth max
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
    
    # Phase 7: DCT-based L-BFGS-B refinement
    if deadline_ok():
        try:
            h_best = best_h
            dct_coeffs = scipy_dct(h_best, type=2, n=_K_DCT, norm='ortho')
            
            res5 = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=50.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res5.x is not None:
                h = _dct_h(res5.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Nelder-Mead polish on DFT basis
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
    
    # Phase 9: Final windowed ZC refinement with L-BFGS-B
    if deadline_ok():
        try:
            # Try to refine the best windowed ZC parameters
            # Use a small local search around the best known parameters
            h_best = best_h
            # Fit a windowed ZC to the best h
            i = np.arange(N, dtype=float)
            # Try a few rate values
            best_params = None
            best_fit = np.inf
            for rate_try in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
                ang = 2.0 * np.pi * rate_try * i * (i + 1) / N
                cos_a = np.cos(ang)
                sin_a = np.sin(ang)
                # Least squares fit: h ≈ base + amp * cos(ang + phase)
                # = base + amp*cos(phase)*cos(ang) - amp*sin(phase)*sin(ang)
                A = np.column_stack([np.ones(N), cos_a, sin_a])
                try:
                    sol = np.linalg.lstsq(A, h_best, rcond=None)[0]
                    base_fit = sol[0]
                    amp_c = sol[1]
                    amp_s = sol[2]
                    amp_fit = np.sqrt(amp_c**2 + amp_s**2)
                    phase_fit = np.arctan2(-amp_s, amp_c)
                    # Compute residual
                    h_fit = base_fit + amp_fit * np.cos(ang + phase_fit)
                    fit_err = np.sum((h_best - h_fit)**2)
                    if fit_err < best_fit:
                        best_fit = fit_err
                        best_params = [phase_fit, rate_try, amp_fit, base_fit, 1.0, 0.5]
                except:
                    pass
            
            if best_params is not None:
                # Refine with L-BFGS-B
                res_wz = minimize(
                    lambda p: _windowed_zc_smooth_obj(p, beta=50.0),
                    best_params,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (-50, 50), (-3, 3), (-3, 3), (0.1, 1.5), (0.2, 0.8)],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                if res_wz.x is not None:
                    h = _windowed_zadoff_chu_h(res_wz.x)
                    update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
