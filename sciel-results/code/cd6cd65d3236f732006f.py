"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a global Chebyshev polynomial basis with local Gaussian RBFs.
    
    The model approximates y(x) as:
    y(x) = P(t) + sum_{i=0}^{4} A_i * G(x; c_i, sigma)
    
    Where:
    - P(t) is a cubic Chebyshev polynomial in t = x/2.
      P(t) = p0*T0(t) + p1*T1(t) + p2*T2(t) + p3*T3(t)
    - G(x; c, sigma) is a Gaussian basis function exp(-0.5 * ((x-c)/sigma)^2)
    - There are 5 Gaussian centers c_i evenly spaced in [-2, 2].
    
    Parameters (10 total):
    0-3: Chebyshev coefficients p0, p1, p2, p3
    4-8: Gaussian amplitudes A_0, ..., A_4
    9:   Gaussian width sigma (log-scale for stability: sigma = exp(p9))
         We constrain sigma to be positive and reasonable.
         Actually, let's use p9 directly as sigma, but ensure it's not too small.
         Or better, use a fixed sigma for stability with BFGS, as variable width can be tricky.
         Let's use a fixed sigma = 0.8 for the Gaussians to complement the polynomial.
         Wait, if sigma is fixed, we only have 9 params?
         Let's re-allocate:
         0-3: Chebyshev coefficients (4 params)
         4-8: Gaussian amplitudes (5 params)
         9:   Offset / Bias term? Or a 5th Chebyshev coeff?
         Let's do:
         0-4: Chebyshev coefficients p0..p4 (5 params)
         5-9: Gaussian amplitudes A_0..A_4 (5 params)
         Fixed sigma = 0.8.
         This gives a global smooth trend (degree 4 poly) plus 5 local bumps.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > 145.0:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    # Unpack parameters
    p0, p1, p2, p3, p4 = params[0], params[1], params[2], params[3], params[4]
    A0, A1, A2, A3, A4 = params[5], params[6], params[7], params[8], params[9]
    
    # 1. Chebyshev Polynomial Part
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T0..T4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    poly_part = p0 * T0 + p1 * T1 + p2 * T2 + p3 * T3 + p4 * T4
    
    # 2. Gaussian RBF Part
    # Fixed centers for 5 Gaussians
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Fixed width sigma
    # A sigma of 0.8 provides good coverage and overlap for local features
    sigma = 0.8
    inv_2_sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    # Compute Gaussians
    # x shape (N,), centers shape (5,)
    # diff shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    exponents = -inv_2_sigma2 * diff * diff
    gaussians = np.exp(exponents)
    
    # Linear combination of Gaussians
    # A is shape (5,)
    A = params[5:10]
    rbf_part = gaussians @ A
    
    # Combine
    result = poly_part + rbf_part
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
