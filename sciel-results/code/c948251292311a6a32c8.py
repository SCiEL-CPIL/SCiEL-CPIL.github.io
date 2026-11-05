"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Combined strategy:
- Primary: Two orthogonal time-shifted Zadoff-Chu sequences (from current) with global search.
- Secondary: DCT spectral shaping with L-BFGS-B (from inspiration, higher K).
- Tertiary: Windowed single Zadoff-Chu.
- Refinement: DFT basis + full-dimensional L-BFGS-B with smooth max.
- Key improvement: Decoupled DC offset optimization (from knowledge: 1.057 -> 1.16 gain).
- Deadline guard ensures return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_K_DCT = 40  # number of DCT basis vectors for spectral shaping (increased from 30)
_K_DFT = 20  # number of DFT frequency components for refinement (increased from 15)

# Precompute common arrays
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

_DCT_BASIS = _dct_basis_matrix(_K_DCT, N)  # (N, K_DCT)
_DCT_BASIS_T = _DCT_BASIS.T  # (K_DCT, N)

def _dct_h(coeffs):
    """Generate h from DCT coefficients."""
    return _DCT_BASIS @ coeffs

def _dct_smooth_obj(coeffs, beta=80.0):
    """Smooth objective for DCT coefficients with decoupled DC offset."""
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

def _dft_smooth_obj(coeffs_base, beta=80.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _two_zadoff_chu_h(params):
    """Two orthogonal time-shifted Zadoff-Chu sequences.
    
    h[i] = base + amp1 * cos(2*pi*rate1*i*(i+1)/N + phase1) + amp2 * cos(2*pi*rate2*(i+shift)*(i+shift+1)/N + phase2)
    
    params: [phase1, rate1, amp1, phase2, rate2, amp2, shift, base]
    """
    phase1, rate1, amp1, phase2, rate2, amp2, shift, base = params
    i = _i_arr
    
    # First Zadoff-Chu sequence
    ang1 = 2.0 * np.pi * rate1 * i * (i + 1) / N + phase1
    
    # Second Zadoff-Chu sequence with time shift
    i2 = i + shift
    ang2 = 2.0 * np.pi * rate2 * i2 * (i2 + 1) / N + phase2
    
    h = base + amp1 * np.cos(ang1) + amp2 * np.cos(ang2)
    return h

def _two_zadoff_chu_obj(params):
    """Objective for two Zadoff-Chu parameters."""
    h = _two_zadoff_chu_h(params)
    return _value(h)

def _two_zadoff_chu_smooth_obj(params, beta=80.0):
    """Smooth objective for two Zadoff-Chu parameters."""
    h = _two_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu: h[i] = base + amp * window[i] * cos(2*pi*rate*i*(i+1)/N + phase)."""
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

def _project_dft(h):
    """Project h onto DFT basis + base, return (coeffs, base_val)."""
    mean_h = float(np.mean(h))
    coeffs = np.zeros(2 * _K_DFT)
    target = h - mean_h
    A = _BASIS_T  # (N, 2*K_DFT)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
        coeffs = sol
    except:
        pass
    return coeffs, mean_h

def _optimize_dc_offset(h_centered):
    """Given a zero-mean signal shape, find the optimal DC offset to minimize the value.
    
    This is the key improvement from the knowledge: decoupling DC offset from shape optimization.
    For h = h_centered + c, we want to minimize 2*N*max(conv(h,h)) / (sum(h))^2.
    sum(h) = N*c (since h_centered has zero mean).
    conv(h,h) = conv(h_centered + c, h_centered + c) = conv(h_centered, h_centered) + c*(conv(h_centered, 1) + conv(1, h_centered)) + c^2*conv(1,1)
    conv(1,1) = [N, N-1, ..., 1, 1, ..., N-1, N]
    conv(h_centered, 1) = [sum(h_centered), sum(h_centered), ..., sum(h_centered)] = 0 (since zero mean)
    So conv(h,h) = conv(h_centered, h_centered) + c^2 * conv(1,1)
    max(conv(h,h)) = max(conv(h_centered, h_centered) + c^2 * conv(1,1))
    
    Since conv(1,1) is positive and symmetric, and c^2 >= 0, the max is achieved at the center.
    Actually, let's just do a 1D optimization over c.
    """
    h_centered = np.asarray(h_centered, dtype=float)
    # Ensure zero mean
    h_centered = h_centered - np.mean(h_centered)
    
    def obj(c):
        h = h_centered + c
        return _value(h)
    
    # Optimize c using scalar minimization
    res = minimize(obj, 0.0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
    if res.fun < np.inf:
        return res.x[0]
    return 0.0

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
            return
        # Apply DC offset optimization
        h_centered = h - np.mean(h)
        c_opt = _optimize_dc_offset(h_centered)
        h_opt = h_centered + c_opt
        v = _value(h_opt)
        if v < best_v:
            best_v = v
            best_h = h_opt.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (multiple restarts)
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(10):
                if not deadline_ok():
                    break
                # Initialize DCT coefficients: sparse random
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(15, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1, size=n_active)
                
                res = minimize(
                    lambda c: _dct_smooth_obj(c, beta=50.0),
                    coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                if res.fun < np.inf:
                    h_candidate = _dct_h(res.x)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Two Zadoff-Chu sequences global search (primary strategy)
    if deadline_ok():
        try:
            bounds_2zc = [
                (-np.pi, np.pi),       # phase1
                (-3.0, 3.0),           # rate1
                (-5.0, 5.0),           # amp1
                (-np.pi, np.pi),       # phase2
                (-3.0, 3.0),           # rate2
                (-5.0, 5.0),           # amp2
                (-50.0, 50.0),         # shift
                (-5.0, 5.0),           # base
            ]
            
            res_de = differential_evolution(
                _two_zadoff_chu_obj,
                bounds_2zc,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < np.inf:
                h_candidate = _two_zadoff_chu_h(res_de.x)
                update_best(h_candidate)
                
                # Refine with smooth objective using L-BFGS-B
                if deadline_ok():
                    res_refine = minimize(
                        lambda p: _two_zadoff_chu_smooth_obj(p, beta=50.0),
                        res_de.x,
                        method='L-BFGS-B',
                        bounds=bounds_2zc,
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                    )
                    if res_refine.fun < np.inf:
                        h_refined = _two_zadoff_chu_h(res_refine.x)
                        update_best(h_refined)
        except Exception:
            pass
    
    # Phase 2b: Windowed single Zadoff-Chu as fallback
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),       # phase
                (-5.0, 5.0),           # rate
                (-10.0, 10.0),         # amp
                (-10.0, 10.0),         # base
                (0.1, 1.5),            # win_width
                (0.2, 0.8),            # win_center
                (0.0, 1.0),            # win_type
            ]
            
            res_de_wzc = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=123,
                maxiter=60,
                popsize=15,
                tol=1e-6,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de_wzc.fun < np.inf:
                h_candidate = _windowed_zadoff_chu_h(res_de_wzc.x)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Local refinement on DFT basis from best_h
    if deadline_ok():
        try:
            coeffs_init, base_val = _project_dft(best_h)
            
            def dft_full_obj(params):
                coeffs = params[:-1]
                base = params[-1]
                return _dft_smooth_obj(np.concatenate([coeffs, [base]]), beta=30.0)
            
            params_init = np.concatenate([coeffs_init, [base_val]])
            bounds_dft = [(-10.0, 10.0)] * (2 * _K_DFT) + [(-10.0, 10.0)]
            
            res_dft = minimize(
                dft_full_obj,
                params_init,
                method='L-BFGS-B',
                bounds=bounds_dft,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            if res_dft.fun < np.inf:
                coeffs_opt = res_dft.x[:-1]
                base_opt = res_dft.x[-1]
                h_candidate = _dft_h(coeffs_opt, base_opt)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 4: Final high-dimensional polish on h directly with adaptive tau
    if deadline_ok():
        try:
            betas = [30.0, 15.0, 8.0]
            for beta in betas:
                if not deadline_ok():
                    break
                def h_smooth_obj(h_vec, b=beta):
                    return _smooth_max_value(h_vec, beta=b)
                
                res_final = minimize(
                    h_smooth_obj,
                    best_h,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0}
                )
                if res_final.fun < np.inf:
                    h_final = res_final.x
                    v_final = _value(h_final)
                    if v_final < best_v and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                        best_h = h_final
                        best_v = v_final
        except Exception:
            pass
    
    # Final DC offset optimization on the best candidate
    if deadline_ok():
        try:
            h_centered = best_h - np.mean(best_h)
            c_opt = _optimize_dc_offset(h_centered)
            h_opt = h_centered + c_opt
            v_opt = _value(h_opt)
            if v_opt < best_v:
                best_h = h_opt
                best_v = v_opt
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    # Final sanity check
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
