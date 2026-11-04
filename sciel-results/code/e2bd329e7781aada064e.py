"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 400

# EVOLVE-BLOCK-START
def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    if n == 0:
        return np.array([])
    # Use FFT for speed
    size = 2 * n - 1
    # Pad to next power of 2 for efficiency
    fft_size = 1
    while fft_size < size:
        fft_size <<= 1
    F = np.fft.rfft(x, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]


def _score_from_signal(x):
    """
    Compute the metric: max(|autoconv|) / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12  # Avoid division by zero
    conv = _autoconv_full(x)
    max_abs = np.max(np.abs(conv))
    return max_abs / (integral * integral)


def _param_to_signal(params):
    """
    Convert spectral coefficients to a time-domain signal.
    params: array of length M, representing coefficients of cos/sin terms.
    We use a real-valued signal constructed from DFT coefficients.
    
    To ensure the signal is real, we construct it from a set of frequency bins.
    Let's use M frequency bins (positive frequencies), and construct:
    x[n] = sum_{k=1}^{M} a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N)
    
    But to keep it simpler and ensure we can represent any real signal with
    N-1 degrees of freedom (since the DC term affects the integral),
    let's parameterize as:
    x[n] = c0 + sum_{k=1}^{M} c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    
    However, the problem is that the metric is scale-invariant in a specific way:
    max(|conv|) / (sum(x))^2
    If we scale x by alpha, conv scales by alpha^2, and (sum(x))^2 scales by alpha^2,
    so the metric is invariant to scaling. Thus, we can fix one parameter, e.g., c0 = 1,
    or fix the norm.
    
    Let's fix the integral to be 1 by construction? No, that's hard.
    Instead, let's just let the optimizer handle the scale.
    
    To reduce dimensionality, we use M = 20 frequency bins (40 parameters: 20 cos + 20 sin).
    Plus a DC offset. Total 41 parameters.
    """
    M = len(params) // 2  # Number of frequency bins
    # params = [c0, c1, d1, c2, d2, ..., cM, dM]
    c0 = params[0]
    x = np.zeros(N)
    x[:] = c0
    for k in range(1, M + 1):
        c_k = params[2 * k - 1]
        d_k = params[2 * k]
        n = np.arange(N)
        x += c_k * np.cos(2 * np.pi * k * n / N) + d_k * np.sin(2 * np.pi * k * n / N)
    return x


def _objective(params):
    """Objective function for differential evolution."""
    x = _param_to_signal(params)
    return _score_from_signal(x)


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Initial best: all ones
    best_x = np.ones(N)
    best_score = _score_from_signal(best_x)
    
    # Differential evolution with spectral parameterization
    # Use M = 15 frequency bins -> 31 parameters
    M = 15
    n_params = 2 * M + 1
    
    # Bounds: allow coefficients in [-10, 10]
    bounds = [(-10, 10)] * n_params
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            updating='immediate'
        )
        
        if result.success or result.fun < best_score:
            x_opt = _param_to_signal(result.x)
            score_opt = _score_from_signal(x_opt)
            if score_opt < best_score:
                best_x = x_opt
                best_score = score_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        return best_x
    
    # Try a few more random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(3):
        if time.time() > deadline - 10:
            break
        
        # Random initial population
        init = rng.uniform(-5, 5, size=n_params)
        
        try:
            result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                seed=restart + 100,
                x0=init,
                polish=True,
                updating='immediate'
            )
            
            if result.success or result.fun < best_score:
                x_opt = _param_to_signal(result.x)
                score_opt = _score_from_signal(x_opt)
                if score_opt < best_score:
                    best_x = x_opt
                    best_score = score_opt
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
