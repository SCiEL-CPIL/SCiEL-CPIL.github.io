"""
Strategy: Low-Frequency Fourier Parameterization with Differential Evolution

Instead of optimizing the 400 time-domain samples directly (which is a high-dimensional, 
non-smooth problem due to the max operator on the convolution), we parameterize the 
signal h as the inverse Discrete Fourier Transform (IDFT) of a sparse set of low-frequency 
coefficients. 

By restricting the signal to the first K frequency bins (plus DC), we enforce a smooth, 
low-bandwidth structure on h. This drastically reduces the effective dimensionality of the 
search space. We then use Differential Evolution (a global optimization algorithm) to 
optimize these low-dimensional parameters. This approach is structurally different from 
local methods like Powell or coordinate descent, as it explores the global landscape of 
the low-frequency subspace.

Key differences from previous strategies:
1. Uses Fourier-domain parameterization (low-frequency IDFT) instead of time-domain 
   Zadoff-Chu or direct sample optimization.
2. Uses Differential Evolution (global stochastic search) instead of local deterministic 
   optimization (Powell, coordinate descent).
3. The signal structure is constrained to be smooth/low-bandwidth by construction, 
   which may help avoid high-frequency noise that increases the peak of the 
   autoconvolution.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _h_from_coeffs(coeffs, K):
    """
    Reconstruct h from low-frequency Fourier coefficients.
    coeffs: array of length 2*K + 1, representing DC, K positive, K negative frequencies.
    We use a real-valued signal, so we ensure conjugate symmetry.
    """
    h = np.zeros(N, dtype=np.float64)
    # DC component
    h += coeffs[0]
    
    # Positive and negative frequency components
    for k in range(1, K + 1):
        # coeffs[k] is the amplitude for positive frequency k
        # coeffs[K + k] is the amplitude for negative frequency k
        # For real signal: h[n] += 2 * Re( A_k * exp(2j*pi*k*n/N) )
        # But since we want to optimize real amplitudes directly, let's use:
        # h[n] += A_k * cos(2*pi*k*n/N) + B_k * sin(2*pi*k*n/N)
        # We'll store A_k and B_k in coeffs.
        # Let's redefine coeffs layout:
        # coeffs[0] = DC
        # coeffs[1] = A_1 (cos coeff for freq 1)
        # coeffs[2] = B_1 (sin coeff for freq 1)
        # coeffs[3] = A_2
        # coeffs[4] = B_2
        # ...
        # coeffs[2*k-1] = A_k
        # coeffs[2*k] = B_k
        
        # This layout is cleaner. Let's adjust the function to use this layout.
        pass
    
    # Let's use the cleaner layout: coeffs[0] = DC, coeffs[2k-1] = A_k, coeffs[2k] = B_k
    # Total length = 1 + 2*K
    n = np.arange(N)
    h = np.full(N, coeffs[0])
    for k in range(1, K + 1):
        A_k = coeffs[2*k - 1]
        B_k = coeffs[2*k]
        h += A_k * np.cos(2 * np.pi * k * n / N) + B_k * np.sin(2 * np.pi * k * n / N)
    return h

def _objective(coeffs, K):
    """Objective function for optimization."""
    h = _h_from_coeffs(coeffs, K)
    return _evaluate(h)

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Choose K, the number of frequency bins to include
    # K=10 gives 21 parameters, K=20 gives 41 parameters
    K = 15  # 31 parameters total
    
    # Bounds for each coefficient
    # DC component can be large
    # Other components can be moderate
    bounds = [(0.0, 10.0)]  # DC
    for _ in range(K):
        bounds.append((-5.0, 5.0))  # A_k
        bounds.append((-5.0, 5.0))  # B_k
    
    # Initial population size
    popsize = 10
    
    try:
        res = differential_evolution(
            _objective,
            bounds,
            args=(K,),
            maxiter=50,
            popsize=popsize,
            tol=1e-4,
            seed=42,
            polish=True,  # Use local optimization at the end
            update_interval=10,
            callback=None,
            init='latinhypercube'
        )
        
        if res.success or res.fun < np.inf:
            best_h = _h_from_coeffs(res.x, K)
            best_val = res.fun
        else:
            best_h = np.ones(N)
            best_val = 2.0
            
    except Exception as e:
        best_h = np.ones(N)
        best_val = 2.0
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
        
    return best_h
