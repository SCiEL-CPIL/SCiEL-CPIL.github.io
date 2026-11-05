"""
Third autocorrelation inequality: minimize 2*N*max(conv(h,h))/sum(h)^2 over h in R^N, N=400.

Strategy:
- Primary: Square wave times polynomial window (to suppress sidelobes)
- Secondary: DCT spectral shaping with L-BFGS-B
- Tertiary: Windowed Zadoff-Chu with global search
- Final: High-dimensional L-BFGS-B polish with smooth max
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400
_t0 = time.time()
_DEADLINE = 130.0  # safety margin under 150s wall-clock
_K_DCT = 40  # number of DCT basis vectors for spectral shaping
_K_DFT = 20  # number of DFT frequency components for refinement
_POLY_DEG = 8  # polynomial window degree

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
    shifted = np.clip(shifted, -500, 500)
    smooth_m = m + np.log(np.sum(np.exp(shifted))) / beta
    return abs(2.0 * N * smooth_m / (s * s))

def _epigraph_smooth_obj(h, t, beta=50.0):
    """Epigraph formulation: minimize t subject to smooth_max(conv(h,h)) <= t.
    Returns a scalar penalty: t + penalty for violating the constraint.
    """
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = _fft_convolve(h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    smooth_m = m + np.log(np.sum(np.exp(shifted))) / beta
    # Objective: 2*N*smooth_m/s^2 + lambda * max(0, smooth_m - t)^2
    val = 2.0 * N * smooth_m / (s * s)
    violation = max(0.0, smooth_m - t)
    penalty = 100.0 * violation * violation
    return val + penalty

def _epigraph_full_obj(params, beta=50.0):
    """Full epigraph objective: params = [h..., t]"""
    h = params[:-1]
    t = params[-1]
    return _epigraph_smooth_obj(h, t, beta)

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

# DFT basis for refinement
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

def _dft_smooth_obj(coeffs_base, beta=50.0):
    """Smooth objective for DFT parameters."""
    coeffs = coeffs_base[:-1]
    base_val = coeffs_base[-1]
    h = _dft_h(coeffs, base_val)
    return _smooth_max_value(h, beta)

def _poly_window_h(params):
    """Square wave times polynomial window.
    
    params: [amp, base, poly_coeffs...]
    h[i] = base + amp * square_wave[i] * poly_window(i/N)
    where square_wave[i] = sign(sin(2*pi*i/N)) and poly_window is a polynomial in t=i/N
    """
    amp, base = params[0], params[1]
    poly_coeffs = params[2:]  # coefficients for polynomial of degree _POLY_DEG
    
    i = np.arange(N, dtype=float)
    t = i / N  # normalized position [0, 1)
    
    # Square wave: sign of sin(2*pi*t)
    sw = np.sign(np.sin(2.0 * np.pi * t))
    sw[sw == 0] = 1.0  # avoid zeros
    
    # Polynomial window: w(t) = sum_k c_k * t^k
    # Evaluate polynomial using Horner's method
    w = np.zeros(N)
    for k in range(len(poly_coeffs) - 1, -1, -1):
        w = w * t + poly_coeffs[k]
    
    h = base + amp * sw * w
    return h

def _poly_window_obj(params):
    """Objective for polynomial window parameters."""
    h = _poly_window_h(params)
    return _value(h)

def _poly_window_smooth_obj(params, beta=50.0):
    """Smooth objective for polynomial window parameters."""
    h = _poly_window_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu: h[i] = base + amp * window[i] * cos(2*pi*rate*i*(i+1)/N + phase)."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    win_type = params[6] if len(params) > 6 else 0.0
    
    i = np.arange(N, dtype=float)
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

def _windowed_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

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
    
    # Phase 1: Polynomial window on square wave
    if deadline_ok():
        try:
            # Initialize polynomial coefficients: start with a smooth taper
            # w(t) should be ~1 in the middle, taper to 0 at edges
            # Use a polynomial that approximates a smooth window
            poly_init = np.zeros(_POLY_DEG + 1)
            poly_init[0] = 1.0  # constant term
            # Add some curvature to taper edges
            poly_init[2] = -2.0
            poly_init[4] = 1.0
            
            params_init = np.concatenate([[1.0, 0.0], poly_init])
            
            # Optimize with L-BFGS-B
            res_poly = minimize(
                lambda p: _poly_window_smooth_obj(p, beta=30.0),
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
            )
            if res_poly.fun < np.inf:
                h_candidate = _poly_window_h(res_poly.x)
                update_best(h_candidate)
                
                # Try different initializations
                for restart in range(5):
                    if not deadline_ok():
                        break
                    rng = np.random.default_rng(100 + restart)
                    poly_rand = rng.normal(0, 0.5, size=_POLY_DEG + 1)
                    poly_rand[0] = 1.0 + rng.normal(0, 0.2)
                    params_rand = np.concatenate([[1.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)], poly_rand])
                    
                    res_r = minimize(
                        lambda p: _poly_window_smooth_obj(p, beta=30.0),
                        params_rand,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                    )
                    if res_r.fun < np.inf:
                        h_c = _poly_window_h(res_r.x)
                        update_best(h_c)
        except Exception:
            pass
    
    # Phase 2: DCT spectral shaping
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(10):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(3, min(15, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1, size=n_active)
                
                res = minimize(
                    lambda c: _dct_smooth_obj(c, beta=30.0),
                    coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                if res.fun < np.inf:
                    h_candidate = _dct_h(res.x)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-5.0, 5.0),
                (-10.0, 10.0),
                (-10.0, 10.0),
                (0.1, 1.5),
                (0.2, 0.8),
                (0.0, 1.0),
            ]
            
            res_de = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=42,
                maxiter=80,
                popsize=10,
                tol=1e-6,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < np.inf:
                h_candidate = _windowed_zadoff_chu_h(res_de.x)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 4: DFT refinement
    if deadline_ok():
        try:
            base_val = np.mean(best_h)
            h_centered = best_h - base_val
            coeffs_init, _, _, _ = np.linalg.lstsq(_BASIS_T, h_centered, rcond=None)
            
            params_init = np.concatenate([coeffs_init, [base_val]])
            bounds_dft = [(-10.0, 10.0)] * (2 * _K_DFT) + [(-10.0, 10.0)]
            
            res_dft = minimize(
                lambda p: _dft_smooth_obj(p, beta=20.0),
                params_init,
                method='L-BFGS-B',
                bounds=bounds_dft,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            if res_dft.fun < np.inf:
                coeffs_opt = res_dft.x[:-1]
                base_opt = res_dft.x[-1]
                h_candidate = _dft_h(coeffs_opt, base_opt)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 5: Final high-dimensional polish with two-stage smooth max
    if deadline_ok():
        try:
            # Stage 5a: Coarse polish with moderate beta
            res_coarse = minimize(
                lambda hv: _smooth_max_value(hv, beta=20.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            if res_coarse.fun < np.inf and np.all(np.isfinite(res_coarse.x)):
                h_coarse = res_coarse.x
                v_coarse = _value(h_coarse)
                if v_coarse < best_v and abs(np.sum(h_coarse)) > 1e-12:
                    best_h = h_coarse
                    best_v = v_coarse
        except Exception:
            pass
    
    if deadline_ok():
        try:
            # Stage 5b: Fine polish with high beta (closer to true max)
            res_fine = minimize(
                lambda hv: _smooth_max_value(hv, beta=100.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'iprint': 0}
            )
            if res_fine.fun < np.inf and np.all(np.isfinite(res_fine.x)):
                h_fine = res_fine.x
                v_fine = _value(h_fine)
                if v_fine < best_v and abs(np.sum(h_fine)) > 1e-12:
                    best_h = h_fine
                    best_v = v_fine
        except Exception:
            pass
    
    if deadline_ok():
        try:
            # Stage 5c: Epigraph formulation for even finer control
            # Initialize t to current max of autoconvolution
            conv_init = _fft_convolve(best_h)
            t_init = float(np.max(conv_init)) + 0.1
            params_init = np.concatenate([best_h, [t_init]])
            
            res_epi = minimize(
                lambda p: _epigraph_full_obj(p, beta=100.0),
                params_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14, 'iprint': 0}
            )
            if res_epi.fun < np.inf and np.all(np.isfinite(res_epi.x)):
                h_epi = res_epi.x[:-1]
                v_epi = _value(h_epi)
                if v_epi < best_v and abs(np.sum(h_epi)) > 1e-12:
                    best_h = h_epi
                    best_v = v_epi
        except Exception:
            pass
    
    if deadline_ok():
        try:
            # Stage 5d: Final exact-value polish with very tight tolerances
            res_final = minimize(
                lambda hv: _smooth_max_value(hv, beta=200.0),
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-17, 'gtol': 1e-15, 'iprint': 0}
            )
            if res_final.fun < np.inf and np.all(np.isfinite(res_final.x)):
                h_final = res_final.x
                v_final = _value(h_final)
                if v_final < best_v and abs(np.sum(h_final)) > 1e-12:
                    best_h = h_final
                    best_v = v_final
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)
    
    # Final sanity check: ensure the value is actually improved
    if final_val >= best_v - 1e-15:
        # Recompute to be sure
        best_v = final_val
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
