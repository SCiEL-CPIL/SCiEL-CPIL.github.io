# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import math


def _build_quadrature(k=10):
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x, w = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    return x, w, phi, dphi


def _compute_harmonics(x, phi, k=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -k..k at points (x=cos theta, phi).
    Returns array of shape (n_x * n_phi, 2*k+1)
    """
    n_x = len(x)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_x, n_phi, n_m), dtype=np.float64)
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        log_num = math.lgamma(k + 1 - abs_m)
        log_den = math.lgamma(k + 1 + abs_m)
        N_m = math.sqrt((2 * k + 1) / (4 * np.pi) * math.exp(log_num - log_den))
        
        P = lpmv(abs_m, k, x)
        
        if m == 0:
            Y[:, :, 10] = N_m * P[:, np.newaxis]
        elif m > 0:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y[:, :, 10 + m] = math.sqrt(2) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return Y.reshape(n_x * n_phi, n_m)


def _compute_quartic_ratio(coeffs, Y, w, dphi):
    """
    Compute (||f||_4^4) / (||f||_2^4) = (Integral f^4) / (Integral f^2)^2.
    This is equivalent to (||f||_4 / ||f||_2)^4.
    Minimizing this is equivalent to minimizing ||f||_4 / ||f||_2.
    """
    f = Y @ coeffs
    n_x = len(w)
    n_phi = Y.shape[0] // n_x
    weights_full = np.tile(w * dphi, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights_full * f2)
    norm4 = np.sum(weights_full * f4)
    
    if norm2_sq <= 1e-15:
        return 1e10 # Large penalty if zero
    
    # Ratio^4
    return norm4 / (norm2_sq * norm2_sq)


def _objective(coeffs, Y, w, dphi):
    """Objective to minimize."""
    return _compute_quartic_ratio(coeffs, Y, w, dphi)


def rudin_problem10() -> np.ndarray:
    k = 10
    n_m = 2 * k + 1
    
    # Build quadrature
    x, w, phi, dphi = _build_quadrature(k)
    
    # Compute harmonics at all quadrature points
    Y = _compute_harmonics(x, phi, k)
    
    # Use Nelder-Mead with multiple random restarts
    # This is a derivative-free method, robust for non-convex quartic forms
    best_coeffs = None
    best_ratio = np.inf
    
    rng = np.random.default_rng(123)
    
    # Generate diverse initializations
    # 1. Zonal (m=0)
    # 2. Random
    # 3. Sparse
    
    n_restarts = 50
    
    for i in range(n_restarts):
        if i == 0:
            coeffs0 = np.zeros(n_m)
            coeffs0[10] = 1.0 # m=0
        elif i == 1:
            coeffs0 = np.zeros(n_m)
            coeffs0[15] = 1.0 # m=5
        else:
            coeffs0 = rng.standard_normal(n_m)
        
        # Normalize initial guess to avoid scaling issues, though objective is scale invariant
        norm = np.linalg.norm(coeffs0)
        if norm > 1e-10:
            coeffs0 = coeffs0 / norm
            
        result = minimize(
            _objective,
            coeffs0,
            args=(Y, w, dphi),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        
        c = result.x
        if np.any(np.isnan(c)) or np.any(np.isinf(c)):
            continue
        
        # Normalize result to unit L2 norm for consistency, though not strictly required for ratio
        norm_c = np.linalg.norm(c)
        if norm_c > 1e-10:
            c = c / norm_c
            
        r4 = _compute_quartic_ratio(c, Y, w, dphi)
        # We want to minimize r4. 
        # Note: The task asks for R = ||f||_4 / ||f||_2. 
        # R^4 = r4. Minimizing r4 minimizes R.
        
        if r4 < best_ratio:
            best_ratio = r4
            best_coeffs = c.copy()
    
    if best_coeffs is None:
        return np.ones(n_m, dtype=np.float64)
    
    coeffs = best_coeffs
    
    # Validate
    max_abs = np.max(np.abs(coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        if max_abs > 0:
            coeffs = coeffs / max_abs
        else:
            return np.ones(n_m, dtype=np.float64)
    
    return coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
