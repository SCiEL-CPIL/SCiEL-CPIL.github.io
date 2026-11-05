"""Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu +
full-dimensional coordinate descent + smooth-max refinement.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional coordinate descent with smooth logsumexp surrogate to escape
  Fourier subspace restriction, allowing high-frequency corrections.
- Refine best candidates with L-BFGS-B on DFT basis and Powell polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_K_DCT = 40
_K_DFT = 20

# Precompute common arrays
_i_arr = np.arange(N, dtype=float)
_n_idx = np.arange(N)

def _value(h):
    """Compute the task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)[:2 * N - 1]
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve(h):
    """Fast autoconvolution via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * n - 1]
    return conv

def _smooth_max_value(h, beta=80.0):
    """Smooth approximation of max of autoconvolution using log-sum-exp."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    return abs(2.0 * N * log_sum_exp / (s * s))

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

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)
_DCT_BASIS_T = _DCT_BASIS.T

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# Low-frequency DFT basis for refinement
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T

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

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _golay_complementary_h(params):
    """
    Golay complementary pair: h = base + amp1*cos(quad_phase1) + amp2*cos(quad_phase2)
    Two chirps with different rates can have complementary autocorrelation properties.
    params: [phase1, rate1, amp1, phase2, rate2, amp2, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, base = params
    i = _i_arr
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    ang2 = 2.0 * np.pi * rate2 * i * (i + 1) / N + phase2
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _golay_obj(params):
    """Objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _value(h)

def _golay_smooth_obj(params, beta=80.0):
    """Smooth objective for Golay complementary parameters."""
    h = _golay_complementary_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu with smooth window to suppress sidelobes."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = _i_arr
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    
    if win_type < 0.5:
        x = (rel_pos - win_center) / (0.5 * win_width)
        window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    else:
        sigma = win_width / 4.0
        window = np.exp(-0.5 * ((rel_pos - win_center) / sigma) ** 2)
    
    h = base + amp * window * np.cos(ang)
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=80.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _chirp_h(params):
    """Generate h from chirp parameters: [phase, rate, amp, base]."""
    phase, rate, amp, base = params
    i = _i_arr
    ang = phase + rate * (i / N) ** 2
    h = base + amp * np.cos(ang)
    return h

def _chirp_obj(params):
    """Objective for chirp parameters."""
    h = _chirp_h(params)
    return _value(h)

def _barker_like_h(params):
    """
    Barker-like sequence from Golay complementary pair.
    h = base + sum_k a_k * cos(2*pi*k*i/N + phi_k) for k in active set
    with optimized amplitudes and phases to minimize peak autocorrelation.
    """
    base = params[0]
    n_harmonics = 8
    amplitudes = params[1:1+n_harmonics]
    phases = params[1+n_harmonics:1+2*n_harmonics]
    i = _i_arr
    h = np.full(N, base)
    for k in range(1, n_harmonics + 1):
        h += amplitudes[k-1] * np.cos(2.0 * np.pi * k * i / N + phases[k-1])
    return h

def _barker_like_obj(params):
    h = _barker_like_h(params)
    return _value(h)

def _barker_like_smooth_obj(params, beta=80.0):
    h = _barker_like_h(params)
    return _smooth_max_value(h, beta)

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    for k in range(1, _K_DFT + 1):
        angle = 2.0 * np.pi * k * _n_idx / N
        cos_k = np.cos(angle)
        sin_k = np.sin(angle)
        A = np.column_stack([cos_k, sin_k])
        try:
            sol = np.linalg.lstsq(A, target, rcond=None)[0]
            coeffs[2 * (k - 1)] = sol[0]
            coeffs[2 * (k - 1) + 1] = sol[1]
        except:
            pass
    return coeffs, mean_h

def _full_dim_smooth_obj(h, beta=80.0):
    """Smooth objective for full-dimensional h."""
    return _smooth_max_value(h, beta)

def _full_dim_coord_descent(h0, max_sweeps=12, beta_init=50.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Escapes Fourier subspace restriction, allowing high-frequency corrections.
    """
    h = h0.copy()
    beta = beta_init
    cur_obj = _smooth_max_value(h, beta)
    step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    rng_cd = np.random.default_rng(99)
    
    n_sweeps = 0
    while (time.time() - _t0) < _DEADLINE and n_sweeps < max_sweeps:
        improved = False
        order = rng_cd.permutation(N)
        for idx in order:
            if (time.time() - _t0) >= _DEADLINE:
                break
            old_val = h[idx]
            best_new_val = old_val
            best_new_obj = cur_obj
            for ss in step_sizes:
                for sign in [1.0, -1.0]:
                    h[idx] = old_val + sign * ss
                    v_try = _smooth_max_value(h, beta)
                    if v_try < best_new_obj:
                        best_new_obj = v_try
                        best_new_val = old_val + sign * ss
                if best_new_val != old_val:
                    break
            if best_new_val != old_val:
                h[idx] = best_new_val
                cur_obj = best_new_obj
                improved = True
        n_sweeps += 1
        if not improved:
            break
        beta = min(beta * 1.4, 200.0)
    
    return h

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
    
    # Phase 1: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.3, 1.5),
                (0.2, 0.8),
                (0.0, 1.0),
            ]
            res = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=45, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _windowed_zadoff_chu_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _golay_complementary_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Barker-like multi-harmonic search
    if deadline_ok():
        try:
            n_harm = 8
            bounds_barker = [(-2.0, 2.0)]  # base
            for _ in range(n_harm):
                bounds_barker.append((-2.0, 2.0))  # amplitude
            for _ in range(n_harm):
                bounds_barker.append((-np.pi, np.pi))  # phase
            
            res = differential_evolution(
                _barker_like_obj, bounds_barker, seed=23, maxiter=35, tol=1e-8,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res.x is not None:
                h = _barker_like_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: DCT spectral shaping with L-BFGS-B (refinement machinery)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(8):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(14, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 5: L-BFGS-B refinement on DFT basis from best solution
    if deadline_ok():
        try:
            h_best = best_h
            coeffs_init, mean_h = _project_dft(h_best)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(
                _dft_smooth_obj, x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 6: DCT L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            h_best = best_h
            dct_coeffs = np.zeros(_K_DCT)
            for k in range(_K_DCT):
                dct_coeffs[k] = np.sum(h_best * _DCT_BASIS[:, k])
            
            res = minimize(
                lambda c: _dct_smooth_obj(c),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                h = _dct_h(res.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Powell polish on DFT basis with exact objective
    if deadline_ok():
        try:
            h_best = best_h
            coeffs_init, mean_h = _project_dft(h_best)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(
                _dft_obj, x0,
                method='Powell',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 400, 'maxfev': 800}
            )
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Full-dimensional coordinate descent with smooth surrogate
    if deadline_ok():
        try:
            h_cd = _full_dim_coord_descent(best_h, max_sweeps=12, beta_init=50.0)
            v_cd = _value(h_cd)
            if v_cd < best_v:
                best_v = v_cd
                best_h = h_cd.copy()
        except Exception:
            pass
    
    # Phase 9: L-BFGS-B on full h vector with smooth surrogate
    if deadline_ok():
        try:
            h_lbfgs = best_h.copy()
            beta_final = 100.0
            
            def _full_smooth_obj(h):
                return _smooth_max_value(h, beta_final)
            
            res = minimize(
                _full_smooth_obj, h_lbfgs,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                h = res.x
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Phase 10: Final high-precision DFT polish with tighter tolerances
    if deadline_ok():
        try:
            h_best = best_h
            coeffs_init, mean_h = _project_dft(h_best)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res = minimize(
                _dft_smooth_obj, x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None:
                h = _dft_h(res.x[:-1], res.x[-1])
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
