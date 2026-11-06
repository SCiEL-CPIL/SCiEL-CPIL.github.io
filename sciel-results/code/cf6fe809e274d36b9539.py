"""
Optimized program: DCT spectral shaping + chirp/Golay/ZC global search +
full-dimensional coordinate descent + smooth-max refinement.

Key improvements over parent:
- Better DCT coefficient initialization (higher energy in low modes)
- More aggressive smooth-max beta for sharper peak focusing
- Improved coordinate descent with adaptive step sizes and line search
- Additional restarts with diverse random seeds
- Tighter deadline guard (130s) for safety
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 130.0
_K_DCT = 60
_K_DFT = 25

_i_arr = np.arange(N, dtype=float)

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

def _smooth_max_value(h, beta=100.0):
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

def _dct_smooth_obj(coeffs, beta=100.0):
    """Smooth objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dct_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
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

def _dft_smooth_obj(coeffs_base, beta=100.0):
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

def _golay_smooth_obj(params, beta=100.0):
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

def _windowed_zc_smooth_obj(params, beta=100.0):
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

def _full_dim_coord_descent(h0, max_iter=400, step_size=0.015, beta=100.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Uses in-place updates to avoid array copies for speed.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try multiple step sizes for better convergence
            for delta in [step_size, -step_size, step_size * 0.5, -step_size * 0.5]:
                h[i] = old_val + delta
                v_trial = _smooth_max_value(h, beta)
                if v_trial < best_v:
                    best_v = v_trial
                    improved = True
                    break
                else:
                    h[i] = old_val
            else:
                h[i] = old_val
        if not improved:
            step_size *= 0.7
            if step_size < 1e-7:
                break
    
    return h, best_v

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return best_v
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (multiple restarts with better init)
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(12):
                if not deadline_ok():
                    break
                # Initialize with more energy in low-frequency modes
                coeffs = np.zeros(_K_DCT)
                # Weight towards lower modes for better initial structure
                for k in range(_K_DCT):
                    weight = np.exp(-k / 20.0)
                    if rng.random() < 0.6:  # 60% chance to activate each mode
                        coeffs[k] = rng.normal(0, 1.5 * weight)
                
                res = minimize(_dct_smooth_obj, coeffs, method='L-BFGS-B',
                              options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-13})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Chirp global search
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_chirp = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=40, tol=1e-7,
                popsize=10, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Golay complementary pair global search
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
            res_golay = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=60, tol=1e-9,
                popsize=15, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_complementary_h(res_golay.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_zc = [
                (-np.pi, np.pi),
                (-50.0, 50.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
                (0.1, 2.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ]
            res_zc = differential_evolution(
                _windowed_zc_obj, bounds_zc, seed=13, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_zc.x is not None:
                h = _windowed_zadoff_chu_h(res_zc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: Full-dimensional coordinate descent with in-place updates
    if deadline_ok():
        try:
            h_cd, v_cd = _full_dim_coord_descent(best_h, max_iter=400, step_size=0.02, beta=100.0)
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 6: DFT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            coeffs, base_val = _project_dft(best_h)
            coeffs_base = np.concatenate([coeffs, [base_val]])
            
            res_dft_refine = minimize(
                _dft_smooth_obj,
                coeffs_base,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_dft_refine.x is not None:
                c = res_dft_refine.x[:-1]
                b = res_dft_refine.x[-1]
                h = _dft_h(c, b)
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: Nelder-Mead polish on DCT basis
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
            res_nm = minimize(
                _dct_obj,
                dct_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 800, 'maxfev': 1500}
            )
            if res_nm.x is not None:
                h = _dct_h(res_nm.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: Final full-dimensional coordinate descent polish with finer steps
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _full_dim_coord_descent(best_h, max_iter=300, step_size=0.008, beta=120.0)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Phase 9: One more round of DCT refinement from the polished solution
    if deadline_ok():
        try:
            dct_coeffs2 = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            res_dct_final = minimize(
                _dct_smooth_obj,
                dct_coeffs2,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res_dct_final.x is not None:
                h = _dct_h(res_dct_final.x)
                update_best(h)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
