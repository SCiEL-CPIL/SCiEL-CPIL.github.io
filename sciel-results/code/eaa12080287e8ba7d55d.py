"""Combined program: DCT spectral shaping + Golay complementary pair + windowed Zadoff-Chu +
full-dimensional coordinate descent + smooth-max refinement.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Golay complementary pair (two chirps with different rates) where the sum of
  individual autocorrelations cancels out off-peak values.
- Tertiary: Windowed Zadoff-Chu sequences with global optimization.
- Fourth: Full-dimensional coordinate descent with smooth logsumexp surrogate to escape
  Fourier subspace restriction, allowing high-frequency corrections.
- Refine best candidates with L-BFGS-B on DFT basis and Nelder-Mead polish.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 135.0  # safety margin under 150s wall-clock
_K_DCT = 40
_K_DFT = 20

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

def _smooth_max_value_and_grad(h, beta=80.0):
    """Smooth max value and gradient w.r.t. h using log-sum-exp weights."""
    h = np.asarray(h, dtype=float)
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf, np.zeros(N)
    conv = _fft_convolve(h)
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    weights = np.exp(scaled)
    weights /= np.sum(weights)
    log_sum_exp = m + np.log(np.sum(np.exp(scaled))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    
    # Gradient: d/dh_i of (2N * max(conv) / s^2)
    # conv[k] = sum_j h[j]*h[j+k], so d(conv[k])/d(h[i]) = h[i+k] (if i+k<N) + h[i-k] (if i>=k)
    # d(log_sum_exp)/d(conv[k]) = weights[k]
    # d(log_sum_exp)/d(h[i]) = sum_k weights[k] * d(conv[k])/d(h[i])
    # = sum_k weights[k] * (h[i+k] if i+k<N else 0 + h[i-k] if i>=k else 0)
    # This is essentially a weighted convolution of weights with h
    
    # Build gradient via FFT for efficiency
    # d(conv[k])/d(h[i]) for k=0..2N-2, i=0..N-1
    # conv[k] = sum_j h[j]*h[j+k] for k>=0 (with j+k<N)
    # For k<0: conv[N-1+k] = sum_j h[j]*h[j-1-k] ... let's use the full conv index
    # Actually conv has indices 0..2N-2 where conv[k] = sum_j h[j]*h[j+k] for valid j
    # d(conv[k])/d(h[i]) = h[i+k] if 0<=i+k<N else 0, plus h[i-k] if 0<=i-k<N else 0
    # Wait: conv[k] = sum_j h[j] * h[j+k]. d/dh[i] = h[i+k] (when j=i) + h[i-k] (when j+k=i, i.e., j=i-k)
    # So d(conv[k])/d(h[i]) = (h[i+k] if i+k<N else 0) + (h[i-k] if i-k>=0 else 0)
    
    # Gradient of log_sum_exp w.r.t. h[i] = sum_k weights[k] * d(conv[k])/d(h[i])
    # = sum_k weights[k] * (h[i+k] + h[i-k]) with bounds
    # This is: (weights * h)[circular-ish] ... let's just do it directly with FFT
    # sum_k weights[k] * h[i+k] = (weights circ-conv h)[i] but only for valid k
    # Since weights is length 2N-1 and h is length N, we need care.
    
    # Simpler: use the fact that d(conv)/d(h) is a Toeplitz-like matrix
    # grad_logsumexp[i] = sum_{k=0}^{2N-2} weights[k] * (h[i+k] if i+k<N else 0 + h[i-k] if i-k>=0 else 0)
    
    # Part 1: sum_k weights[k] * h[i+k] for i+k < N
    # Part 2: sum_k weights[k] * h[i-k] for i-k >= 0
    
    # Use FFT for efficiency
    n_conv = len(conv)  # 2N-1
    size = 2 * n_conv
    W = np.fft.rfft(weights, size)
    H = np.fft.rfft(h, size)
    # Part 1: (weights * h) linear conv, take first N elements
    part1 = np.fft.irfft(W * H, size)[:N]
    # Part 2: flip weights, convolve with h
    W_flip = np.fft.rfft(weights[::-1], size)
    part2 = np.fft.irfft(W_flip * H, size)[:N]
    
    grad_lse = part1 + part2
    
    # val = 2N * lse / s^2 (assuming lse > 0, which it is since max of conv of real h is >= 0 when sum>0... not necessarily)
    # Actually val = abs(2N * lse / s^2). Let's handle the sign.
    sign = np.sign(2.0 * N * log_sum_exp / (s * s))
    if sign == 0:
        sign = 1.0
    grad_val = sign * 2.0 * N * grad_lse / (s * s) - sign * 2.0 * N * log_sum_exp * 2.0 * s / (s * s) * np.ones(N)
    # d/dh_i of (2N*lse/s^2) = 2N*grad_lse[i]/s^2 - 2N*lse*2/(s^3)
    # = 2N*grad_lse[i]/s^2 - 4N*lse/(s^3)
    grad_val = 2.0 * N * grad_lse / (s * s) - 4.0 * N * log_sum_exp / (s ** 3)
    # Apply abs: if the inner value is negative, flip gradient
    inner_val = 2.0 * N * log_sum_exp / (s * s)
    if inner_val < 0:
        grad_val = -grad_val
    
    return val, grad_val

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

def _full_dim_coord_descent(h0, max_iter=500, step_size=0.01, beta=80.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Escapes Fourier subspace restriction, allowing high-frequency corrections.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try small positive and negative steps
            for delta in [step_size, -step_size]:
                h_trial = h.copy()
                h_trial[i] += delta
                v_trial = _smooth_max_value(h_trial, beta)
                if v_trial < best_v:
                    h = h_trial
                    best_v = v_trial
                    improved = True
                    break
        if not improved:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return h, best_v

def _full_dim_grad_refine(h0, max_iter=200, beta_start=200.0, beta_end=50.0, tol=1e-10):
    """
    Full-dimensional gradient descent with adaptive beta for the smooth surrogate.
    Uses analytic gradient of log-sum-exp smoothed max of autoconvolution.
    Much faster than coordinate descent for high-dimensional refinement.
    """
    h = h0.copy()
    best_v = _value(h)
    n_iter = max_iter
    
    for it in range(n_iter):
        # Adaptive beta: start high (smoother), decrease toward exact
        t = it / max(n_iter - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        v, grad = _smooth_max_value_and_grad(h, beta)
        if not np.isfinite(v):
            break
        
        # Gradient descent with adaptive step size
        g_norm = np.linalg.norm(grad)
        if g_norm < tol:
            break
        
        # Line search: try a few step sizes
        best_step_v = v
        best_step_h = h.copy()
        lr = 0.01
        for _ in range(10):
            h_trial = h - lr * grad
            v_trial = _smooth_max_value(h_trial, beta)
            if v_trial < best_step_v:
                best_step_v = v_trial
                best_step_h = h_trial.copy()
            lr *= 0.7
        
        h = best_step_h
        v_exact = _value(h)
        if v_exact < best_v:
            best_v = v_exact
    
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
    
    # Phase 1: DCT spectral shaping with L-BFGS-B (multiple restarts) - primary strategy
    if deadline_ok():
        try:
            rng = np.random.default_rng(42)
            for restart in range(12):
                if not deadline_ok():
                    break
                coeffs = np.zeros(_K_DCT)
                n_active = rng.integers(4, min(14, _K_DCT))
                active_idx = rng.choice(_K_DCT, size=n_active, replace=False)
                coeffs[active_idx] = rng.normal(0, 1.0, size=n_active)
                
                # Use adaptive beta: start smoother, end tighter
                beta_for_this = 100.0 - 5.0 * restart
                res = minimize(lambda c: _smooth_max_value(_dct_h(c), beta=max(beta_for_this, 30.0)),
                              coeffs, method='L-BFGS-B',
                              options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.x is not None:
                    h = _dct_h(res.x)
                    update_best(h)
        except Exception:
            pass
    
    # Phase 2: Golay complementary pair global search
    if deadline_ok():
        try:
            bounds_golay = [
                (-np.pi, np.pi),       # phase1
                (-50.0, 50.0),         # rate1
                (-3.0, 3.0),           # amp1
                (-np.pi, np.pi),       # phase2
                (-50.0, 50.0),         # rate2
                (-3.0, 3.0),           # amp2
                (-3.0, 3.0),           # base
            ]
            res_golay = differential_evolution(
                _golay_obj, bounds_golay, seed=7, maxiter=50, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_golay.x is not None:
                h = _golay_complementary_h(res_golay.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 3: Windowed Zadoff-Chu global search
    if deadline_ok():
        try:
            bounds_wzc = [
                (-np.pi, np.pi),       # phase
                (-50.0, 50.0),         # rate
                (-3.0, 3.0),           # amp
                (-3.0, 3.0),           # base
                (0.3, 1.5),            # win_width
                (0.2, 0.8),            # win_center
                (0.0, 1.5),            # win_type
            ]
            res_wzc = differential_evolution(
                _windowed_zc_obj, bounds_wzc, seed=13, maxiter=45, tol=1e-8,
                popsize=12, init='sobol', polish=False, update_interval=1
            )
            if res_wzc.x is not None:
                h = _windowed_zadoff_chu_h(res_wzc.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 4: Chirp global search for diversity
    if deadline_ok():
        try:
            bounds_chirp = [
                (-np.pi, np.pi),
                (-200.0, 200.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ]
            res_chirp = differential_evolution(
                _chirp_obj, bounds_chirp, seed=99, maxiter=30, tol=1e-6,
                popsize=8, init='sobol', polish=False, update_interval=1
            )
            if res_chirp.x is not None:
                h = _chirp_h(res_chirp.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 5: Full-dimensional gradient refinement with adaptive beta
    # Much faster than coordinate descent, allows more iterations
    if deadline_ok():
        try:
            h_gr, v_gr = _full_dim_grad_refine(best_h, max_iter=150, beta_start=300.0, beta_end=40.0)
            update_best(h_gr)
        except Exception:
            pass
    
    # Phase 5b: Coordinate descent as fallback/complement
    if deadline_ok():
        try:
            h_cd, v_cd = _full_dim_coord_descent(best_h, max_iter=100, step_size=0.01, beta=60.0)
            update_best(h_cd)
        except Exception:
            pass
    
    # Phase 6: L-BFGS-B refinement on DFT basis from best solution (two passes, adaptive beta)
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_dft = minimize(
                lambda x: _dft_smooth_obj(x, beta=100.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dft.x is not None:
                h = _dft_h(res_dft.x[:-1], res_dft.x[-1])
                update_best(h)
        except Exception:
            pass
    
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_dft2 = minimize(
                lambda x: _dft_smooth_obj(x, beta=50.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_dft2.x is not None:
                h = _dft_h(res_dft2.x[:-1], res_dft2.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 7: L-BFGS-B refinement on Golay complementary parameters
    if deadline_ok():
        try:
            x0_golay = np.array([0.0, 5.0, 1.0, np.pi/2, -7.0, 0.8, 0.5])
            
            res_golay_refine = minimize(
                lambda x: _golay_smooth_obj(x, beta=100.0),
                x0_golay,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_golay_refine.x is not None:
                h = _golay_complementary_h(res_golay_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 8: DCT-based L-BFGS-B refinement from best solution
    if deadline_ok():
        try:
            dct_coeffs = scipy_dct(best_h, type=2, n=_K_DCT, norm='ortho')
            
            res_dct_refine = minimize(
                lambda c: _smooth_max_value(_dct_h(c), beta=80.0),
                dct_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_dct_refine.x is not None:
                h = _dct_h(res_dct_refine.x)
                update_best(h)
        except Exception:
            pass
    
    # Phase 9: Nelder-Mead polish on DFT basis
    if deadline_ok():
        try:
            coeffs_init, mean_h = _project_dft(best_h)
            x0 = np.concatenate([coeffs_init, [mean_h]])
            
            res_nm = minimize(
                _dft_obj,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_nm.x is not None:
                h = _dft_h(res_nm.x[:-1], res_nm.x[-1])
                update_best(h)
        except Exception:
            pass
    
    # Phase 10: Final gradient refinement with very tight beta (near-exact)
    if deadline_ok():
        try:
            h_gr2, v_gr2 = _full_dim_grad_refine(best_h, max_iter=100, beta_start=80.0, beta_end=20.0, tol=1e-12)
            update_best(h_gr2)
        except Exception:
            pass
    
    # Phase 11: Final coordinate descent polish with very small steps
    if deadline_ok():
        try:
            h_cd2, v_cd2 = _full_dim_coord_descent(best_h, max_iter=80, step_size=0.003, beta=120.0)
            update_best(h_cd2)
        except Exception:
            pass
    
    # Phase 12: One more gradient pass at even tighter settings
    if deadline_ok():
        try:
            h_gr3, v_gr3 = _full_dim_grad_refine(best_h, max_iter=50, beta_start=40.0, beta_end=15.0, tol=1e-13)
            update_best(h_gr3)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    return best_h
