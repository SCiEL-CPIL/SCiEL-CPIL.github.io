"""
Structural change: eigenvalue-based optimization.
The max of autoconvolution is bounded by the largest eigenvalue of the convolution operator.
We parameterize h in a low-dimensional basis (Fourier + constant) and use a smooth surrogate
with log-sum-exp for the max, optimized via L-BFGS-B with analytic gradient.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 147.0  # Safety margin before 150s kill

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _fft_convolve(h):
    """Compute full autoconvolution via FFT for speed."""
    n = 2 * N - 1
    f = np.fft.rfft(h, n=n)
    conv = np.fft.irfft(f * f, n=n)
    return conv

def _smooth_objective(coeffs, beta=100.0):
    """
    Smooth surrogate objective.
    coeffs: [c0, c1, c2, ..., c_{2K}] where c0 is the DC (constant) component,
    and c_{2k-1}, c_{2k} are sine/cosine coefficients for frequency k.
    h = c0 + sum_k (c_{2k-1} * sin(2*pi*k*t) + c_{2k} * cos(2*pi*k*t))
    """
    h = _coeffs_to_h(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_convolve(h)
    max_c = np.max(conv)
    shifted = conv - max_c
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(coeffs, beta=100.0):
    """Gradient of the smooth surrogate with respect to coefficients."""
    h = _coeffs_to_h(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(coeffs)
    
    conv = _fft_convolve(h)
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum < 1e-300:
        return np.zeros_like(coeffs)
    weights /= wsum
    
    # grad_lse[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * h[j]
    n = 2 * N - 1
    idx_i = np.arange(N)[:, None]
    idx_j = np.arange(N)[None, :]
    W = weights[idx_i + idx_j]
    grad_lse_h = 2.0 * (W @ h)
    
    lse = max_c + np.log(wsum) / beta
    f = 2.0 * N * lse / (s * s)
    
    df_dh = 2.0 * N * grad_lse_h / (s * s) - 4.0 * N * lse / (s * s * s)
    
    if f >= 0:
        df_dh = df_dh
    else:
        df_dh = -df_dh
    
    # Chain rule: df/dcoeffs = df/dh @ dh/dcoeffs
    # dh/dcoeffs is the basis matrix
    B = _get_basis_matrix(len(coeffs) // 2)
    grad_coeffs = B.T @ df_dh
    return grad_coeffs

def _coeffs_to_h(coeffs):
    """Convert coefficients to step heights."""
    c0 = coeffs[0]
    h = np.full(N, c0)
    t = np.linspace(0, 1, N, endpoint=False)
    for k in range(1, len(coeffs) // 2 + 1):
        if 2*k - 1 < len(coeffs):
            h += coeffs[2*k-1] * np.sin(2 * np.pi * k * t)
        if 2*k < len(coeffs):
            h += coeffs[2*k] * np.cos(2 * np.pi * k * t)
    return h

_basis_cache = {}
def _get_basis_matrix(K):
    """Get basis matrix for K frequency components."""
    if K in _basis_cache:
        return _basis_cache[K]
    n_coeffs = 2 * K + 1
    B = np.zeros((N, n_coeffs))
    B[:, 0] = 1.0
    t = np.linspace(0, 1, N, endpoint=False)
    for k in range(1, K + 1):
        B[:, 2*k-1] = np.sin(2 * np.pi * k * t)
        B[:, 2*k] = np.cos(2 * np.pi * k * t)
    _basis_cache[K] = B
    return B

def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    rng = np.random.RandomState(123)
    
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    # Try different numbers of frequency components
    for K in [5, 10, 15, 20, 30, 40, 50]:
        if elapsed() > DEADLINE - 20:
            break
        
        n_coeffs = 2 * K + 1
        B = _get_basis_matrix(K)
        
        # Multiple random initializations
        for init_seed in range(8):
            if elapsed() > DEADLINE - 10:
                break
            
            # Initialize coefficients
            if init_seed == 0:
                # Start from all ones: c0=1, rest 0
                coeffs0 = np.zeros(n_coeffs)
                coeffs0[0] = 1.0
            elif init_seed == 1:
                # Small random perturbation from ones
                coeffs0 = np.zeros(n_coeffs)
                coeffs0[0] = 1.0
                coeffs0[1:] = rng.uniform(-0.1, 0.1, n_coeffs - 1)
            else:
                # Random coefficients
                coeffs0 = rng.uniform(-1, 1, n_coeffs)
                coeffs0[0] += 1.0  # Ensure non-zero sum likely
            
            # Try different beta values
            for beta in [50.0, 100.0, 200.0]:
                if elapsed() > DEADLINE - 5:
                    break
                
                try:
                    result = minimize(
                        lambda x, b=beta: _smooth_objective(x, b),
                        coeffs0,
                        jac=lambda x, b=beta: _smooth_grad(x, b),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-10}
                    )
                    
                    if elapsed() > DEADLINE - 3:
                        break
                    
                    h_candidate = _coeffs_to_h(result.x)
                    v = _eval_value(h_candidate)
                    if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                        best_val = v
                        best_h = h_candidate.copy()
                except Exception:
                    pass
        
        # Also try a polish with higher beta from the best solution found so far
        if elapsed() < DEADLINE - 10:
            try:
                # Project best_h onto the basis
                h_proj = B @ np.linalg.lstsq(B, best_h, rcond=None)[0]
                coeffs_best = np.linalg.lstsq(B, best_h, rcond=None)[0]
                
                for beta_polish in [300.0, 500.0, 1000.0]:
                    if elapsed() > DEADLINE - 5:
                        break
                    res = minimize(
                        lambda x, b=beta_polish: _smooth_objective(x, b),
                        coeffs_best,
                        jac=lambda x, b=beta_polish: _smooth_grad(x, b),
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-12}
                    )
                    h_candidate = _coeffs_to_h(res.x)
                    v = _eval_value(h_candidate)
                    if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                        best_val = v
                        best_h = h_candidate.copy()
                        coeffs_best = res.x
            except Exception:
                pass
    
    # Final local refinement in full space
    if elapsed() < DEADLINE - 5:
        try:
            h_best = best_h.copy()
            for _ in range(500):
                if elapsed() > DEADLINE - 1:
                    break
                # Random coordinate perturbation
                idx = rng.randint(0, N)
                delta = rng.uniform(-0.3, 0.3)
                h_try = h_best.copy()
                h_try[idx] += delta
                s_try = np.sum(h_try)
                if abs(s_try) > 1e-10:
                    v_try = _eval_value(h_try)
                    if v_try < best_val:
                        best_val = v_try
                        h_best = h_try.copy()
        except Exception:
            pass
    
    # Safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
