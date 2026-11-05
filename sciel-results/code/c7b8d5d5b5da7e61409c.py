import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp

N = 400
_t0 = time.time()
_DEADLINE = 138.0  # safety margin under 150s wall-clock
_K_DCT = 40  # number of DCT basis vectors for spectral shaping
_K_DFT = 20  # number of DFT frequency components for refinement

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
    # Normalize conv by s^2 to match the objective function structure
    # Objective: 2 * N * max(conv) / s^2
    # We want to smooth max(conv) / s^2.
    # Since s is a scalar, we can just smooth max(conv) and divide by s^2.
    # However, if s changes, the shape changes.
    # In our new strategy, we will fix s=1.
    m = np.max(conv)
    shifted = beta * (conv - m)
    smooth_m = m + logsumexp(shifted) / beta
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
    """Generate h from DCT coefficients, normalized to sum=1."""
    h_raw = _DCT_BASIS @ coeffs
    # The constant term (DC component) is not in DCT-II basis if we exclude k=0?
    # DCT-II usually includes k=0.
    # Let's check if _dct_basis_matrix includes k=0.
    # k = np.arange(K). If K>0, k=0 is included.
    # If k=0 is included, basis[:,0] is constant.
    # So h_raw already has a DC component.
    # We want sum(h) = 1.
    # sum(h_raw) = sum(_DCT_BASIS @ coeffs).
    # We can add a constant offset to make sum 1.
    # But that adds a parameter.
    # Alternatively, we can just normalize h_raw to have sum 1.
    # h = h_raw / sum(h_raw)
    # This is non-linear and might cause issues if sum(h_raw) is close to 0.
    # Better: h = h_raw + c, where c is chosen such that sum(h)=1.
    # c = (1 - sum(h_raw)) / N.
    s_raw = float(np.sum(h_raw))
    c = (1.0 - s_raw) / N
    return h_raw + c

def _dct_smooth_obj(params, beta=50.0):
    """Smooth objective for DCT coefficients. params: [coeffs...]"""
    coeffs = params
    h = _dct_h(coeffs)
    return _smooth_max_value(h, beta)

def _dual_zc_h(params):
    """Dual Zadoff-Chu complementary pair:
    h[i] = base + a1*cos(2*pi*r1*i*(i+1)/N + p1) + a2*cos(2*pi*r2*i*(i+1)/N + p2)
    
    params: [p1, r1, a1, p2, r2, a2, base]
    The two chirps with different rates can have complementary autocorrelation properties.
    """
    p1, r1, a1, p2, r2, a2, base = params
    i = np.arange(N, dtype=float)
    ang1 = 2.0 * np.pi * r1 * i * (i + 1) / N + p1
    ang2 = 2.0 * np.pi * r2 * i * (i + 1) / N + p2
    h = base + a1 * np.cos(ang1) + a2 * np.cos(ang2)
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
    else:
        h = np.ones(N) / N
    return h

def _dual_zc_obj(params):
    """Objective for dual Zadoff-Chu parameters."""
    h = _dual_zc_h(params)
    return _value(h)

def _dual_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for dual Zadoff-Chu parameters."""
    h = _dual_zc_h(params)
    return _smooth_max_value(h, beta)

def _windowed_zadoff_chu_h(params):
    """Windowed Zadoff-Chu: h[i] = base + amp * window[i] * cos(2*pi*rate*i*(i+1)/N + phase)."""
    phase, rate, amp, base = params[0], params[1], params[2], params[3]
    win_width = params[4] if len(params) > 4 else 1.0
    win_center = params[5] if len(params) > 5 else 0.5
    
    i = np.arange(N, dtype=float)
    ang = 2.0 * np.pi * rate * i * (i + 1) / N + phase
    rel_pos = i / N
    x = (rel_pos - win_center) / (0.5 * win_width)
    window = np.where(np.abs(x) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * x)), 0.0)
    
    h = base + amp * window * np.cos(ang)
    # Normalize to sum=1
    s = np.sum(h)
    if abs(s) > 1e-12:
        h = h / s
    else:
        h = np.ones(N) / N
    return h

def _windowed_zc_obj(params):
    """Objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _value(h)

def _windowed_zc_smooth_obj(params, beta=50.0):
    """Smooth objective for windowed Zadoff-Chu parameters."""
    h = _windowed_zadoff_chu_h(params)
    return _smooth_max_value(h, beta)

# Low-frequency DFT basis for refinement
_n_idx = np.arange(N)
_basis = np.zeros((2 * _K_DFT, N))
for k in range(1, _K_DFT + 1):
    angle = 2.0 * np.pi * k * _n_idx / N
    _basis[2 * (k - 1), :] = np.cos(angle)
    _basis[2 * (k - 1) + 1, :] = np.sin(angle)
_BASIS_T = _basis.T  # (N, 2*K_DFT)

def _dft_h(coeffs):
    """Generate h from DFT coefficients, normalized to sum=1."""
    h_raw = _BASIS_T @ coeffs
    s_raw = float(np.sum(h_raw))
    c = (1.0 - s_raw) / N
    return h_raw + c

def _dft_smooth_obj(params, beta=40.0):
    """Objective for DFT parameters: [coeffs...]."""
    coeffs = params
    h = _dft_h(coeffs)
    return _smooth_max_value(h, beta)

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N) / N  # Normalized seed
    best_v = _value(best_h)
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    def update_best(h):
        nonlocal best_h, best_v
        v = _value(h)
        if v < best_v and np.all(np.isfinite(h)) and abs(np.sum(h)) > 1e-12:
            best_v = v
            best_h = h.copy()
        return v
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (primary)
    if deadline_ok():
        try:
            rng = np.random.default_rng(123)
            for restart in range(12):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(6, min(16, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                params_init = coeffs
                
                res = minimize(
                    lambda p: _dct_smooth_obj(p, beta=50.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * _K_DCT,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
                )
                if res.fun < np.inf:
                    coeffs_opt = res.x
                    h_candidate = _dct_h(coeffs_opt)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 2: Dual Zadoff-Chu complementary pair optimization
    if deadline_ok():
        try:
            rng = np.random.default_rng(789)
            for restart in range(6):
                if not deadline_ok():
                    break
                r1_init = rng.uniform(0.3, 3.0)
                r2_init = r1_init + rng.uniform(0.5, 2.0)
                
                params_init = np.array([
                    rng.uniform(-np.pi, np.pi),
                    r1_init,
                    rng.uniform(-3.0, 3.0),
                    rng.uniform(-np.pi, np.pi),
                    r2_init,
                    rng.uniform(-3.0, 3.0),
                    1.0,
                ])
                
                res = minimize(
                    lambda p: _dual_zc_smooth_obj(p, beta=50.0),
                    params_init,
                    method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi), (0.1, 10.0), (-5.0, 5.0),
                            (-np.pi, np.pi), (0.1, 10.0), (-5.0, 5.0),
                            (0.1, 10.0)],
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
                )
                if res.fun < np.inf:
                    h_candidate = _dual_zc_h(res.x)
                    update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu with differential evolution
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),
                (-5.0, 5.0),
                (-10.0, 10.0),
                (0.1, 10.0),
                (0.3, 1.5),
                (0.2, 0.8),
            ]
            
            res_de = differential_evolution(
                _windowed_zc_obj,
                bounds_wzc,
                seed=42,
                maxiter=50,
                popsize=12,
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
    
    # Phase 4: DFT refinement from best_h
    if deadline_ok():
        try:
            h_centered = best_h - np.mean(best_h)
            coeffs_init, _, _, _ = np.linalg.lstsq(_BASIS_T, h_centered, rcond=None)
            
            params_init = coeffs_init
            bounds_dft = [(-10.0, 10.0)] * (2 * _K_DFT)
            
            res_dft = minimize(
                lambda p: _dft_smooth_obj(p, beta=40.0),
                params_init,
                method='L-BFGS-B',
                bounds=bounds_dft,
                options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-11, 'iprint': 0}
            )
            if res_dft.fun < np.inf:
                coeffs_opt = res_dft.x
                h_candidate = _dft_h(coeffs_opt)
                update_best(h_candidate)
        except Exception:
            pass
    
    # Phase 5: Final high-dimensional polish with progressive smooth max
    if deadline_ok():
        try:
            beta_schedule = [30.0, 15.0, 10.0]
            current_h = best_h.copy()
            for beta in beta_schedule:
                if not deadline_ok():
                    break
                res_final = minimize(
                    lambda hv: _smooth_max_value(hv, beta=beta),
                    current_h,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-13, 'iprint': 0}
                )
                if res_final.fun < np.inf:
                    h_final = res_final.x
                    v_final = _value(h_final)
                    if v_final < best_v and np.all(np.isfinite(h_final)) and abs(np.sum(h_final)) > 1e-12:
                        best_h = h_final
                        best_v = v_final
                        current_h = h_final.copy()
        except Exception:
            pass
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    final_val = _value(best_h)
    if not np.isfinite(final_val) or final_val > 2.0:
        return np.ones(N)
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Score: {2.0 / _value(h)}")
