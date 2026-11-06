"""Combined program: DCT spectral shaping + coordinate descent + smooth-max refinement.

Strategy:
- Primary: DCT spectral shaping with L-BFGS-B on smooth max surrogate (multiple restarts).
- Secondary: Full-dimensional coordinate descent with smooth logsumexp surrogate to escape
  Fourier subspace restriction, allowing high-frequency corrections.
- Tertiary: Final high-precision polish with L-BFGS-B on raw h vector.
- Deadline guard ensures we return before 150s wall-clock.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # safety margin under 150s wall-clock
_K_DCT = 40

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

def _project_to_dct(h):
    """Project h onto DCT basis, return coefficients."""
    return _DCT_BASIS_T @ h

def _full_dim_smooth_obj(h, beta=80.0):
    """Smooth objective for full-dimensional h."""
    return _smooth_max_value(h, beta)

def _full_dim_coord_descent(h0, max_iter=300, step_size=0.01, beta=80.0):
    """
    Full-dimensional coordinate descent with smooth logsumexp surrogate.
    Escapes Fourier subspace restriction, allowing high-frequency corrections.
    """
    h = h0.copy()
    best_v = _smooth_max_value(h, beta)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 20:
            break
        improved = False
        for i in range(N):
            if time.time() > _DEADLINE - 20:
                break
            old_val = h[i]
            # Try a few step sizes
            for ds in [step_size, -step_size, step_size * 0.5, -step_size * 0.5]:
                h[i] = old_val + ds
                v = _smooth_max_value(h, beta)
                if v < best_v:
                    best_v = v
                    improved = True
                    break
            if not improved:
                h[i] = old_val
        if not improved:
            break
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin
    
    # Strategy: Use a pattern with strong negative auto-correlation
    # Start with alternating pattern and optimize
    
    best_h = np.ones(N)
    best_val = _value(best_h)  # = 2.0
    
    # Try several initial patterns
    patterns = []
    
    # Pattern 1: Alternating with optimized ratio
    for c in [0.5, 0.8, 0.9, 0.95, 0.99]:
        h = np.ones(N)
        h[1::2] = -c
        patterns.append(h.copy())
    
    # Pattern 2: Block alternating
    for block in [2, 4, 8, 16]:
        h = np.ones(N)
        for i in range(N):
            if (i // block) % 2 == 1:
                h[i] = -0.5
        patterns.append(h.copy())
    
    # Pattern 3: Smooth oscillation
    for freq in [0.5, 1.0, 2.0, 5.0, 10.0]:
        x = np.linspace(0, freq * 2 * np.pi, N)
        patterns.append(1.0 + 0.5 * np.sin(x))
    
    # Pattern 4: Random with bias
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = rng.randn(N)
        h = h - np.mean(h)  # zero mean component
        h = 1.0 + 0.3 * h / (np.std(h) + 1e-10)
        patterns.append(h)
    
    # Evaluate all patterns
    for h in patterns:
        val = _value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now optimize from the best starting point using DCT basis
    if time.time() < deadline - 30:
        # Project best_h to DCT space
        coeffs0 = _project_to_dct(best_h)
        
        # L-BFGS-B on DCT coefficients with smooth surrogate
        result = minimize(_dct_smooth_obj, coeffs0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success or result.fun < best_val:
            h_candidate = _dct_h(result.x)
            if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                val_candidate = _value(h_candidate)
                if val_candidate < best_val:
                    best_h = h_candidate
                    best_val = val_candidate
        
        # Multiple restarts in DCT space
        n_restarts = 0
        while time.time() < deadline - 20 and n_restarts < 15:
            n_restarts += 1
            rng = np.random.RandomState(n_restarts * 12345 + 67890)
            # Perturb the best solution in DCT space
            coeffs0 = _project_to_dct(best_h)
            coeffs_init = coeffs0 + 0.1 * rng.randn(_K_DCT)
            
            res = minimize(_dct_smooth_obj, coeffs_init, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                h_candidate = _dct_h(res.x)
                if np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                    val_candidate = _value(h_candidate)
                    if val_candidate < best_val:
                        best_h = h_candidate
                        best_val = val_candidate
    
    # Full-dimensional coordinate descent to escape local minima
    if time.time() < deadline - 15:
        h_cd = _full_dim_coord_descent(best_h, max_iter=200, step_size=0.01, beta=80.0)
        val_cd = _value(h_cd)
        if val_cd < best_val:
            best_h = h_cd
            best_val = val_cd
    
    # Final high-precision polish with L-BFGS-B on raw h vector
    if time.time() < deadline - 5:
        result = minimize(_full_dim_smooth_obj, best_h, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success or result.fun < best_val:
            if np.all(np.isfinite(result.x)) and abs(np.sum(result.x)) > 1e-10:
                val_final = _value(result.x)
                if val_final < best_val:
                    best_h = result.x
                    best_val = val_final
    
    # Final check
    final_val = _value(best_h)
    if final_val < 2.0:  # Better than seed
        return best_h
    else:
        return np.ones(N)
