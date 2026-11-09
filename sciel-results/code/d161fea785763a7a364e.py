# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time
import math


def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) and 130-point uniform phi grid."""
    # 70-point Gauss-Legendre on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(70)
    # x = cos(theta), w are weights for integral over cos(theta) in [-1,1]
    # Surface measure: dOmega = dphi d(cos theta), so weight for each (x_i, phi_j) is w_i * dphi
    nphi = 130
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    dphi = 2 * np.pi / nphi
    return x, w, phi, dphi


def _spherical_harmonics_matrix(x, phi, l=10):
    """
    Compute real orthonormal spherical harmonics Y_m for m = -l..l.
    Y_0 = N_0 P_l^0(cos theta)
    Y_m = sqrt(2) N_m P_l^m(cos theta) cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P_l^{|m|}(cos theta) sin(|m| phi) for m < 0
    where N_m = sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!)
    P_l^m includes Condon-Shortley phase.
    
    Returns matrix of shape (len(x)*len(phi), 2*l+1)
    """
    ntheta = len(x)
    nphi = len(phi)
    npts = ntheta * nphi
    Y = np.zeros((npts, 2 * l + 1))
    
    # Precompute cos(m*phi) and sin(|m|*phi) for each m
    cos_m = np.zeros((l + 1, nphi))
    sin_m = np.zeros((l + 1, nphi))
    for m in range(1, l + 1):
        cos_m[m] = np.cos(m * phi)
        sin_m[m] = np.sin(m * phi)
    
    for m in range(0, l + 1):
        # Normalization
        N_m = np.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m))
        
        # Associated Legendre with Condon-Shortley phase
        # scipy.special.lpmv(m, l, x) gives P_l^m(x) with Condon-Shortley phase
        P_lm = lpmv(m, l, x)  # shape (ntheta,)
        
        idx = m + l  # index into Y columns (m ranges -l..l, so m=0 -> index l)
        
        if m == 0:
            Y[:, idx] = np.tile(P_lm * N_m, nphi)
        else:
            sqrt2 = np.sqrt(2.0)
            # For m > 0: Y_m = sqrt(2) N_m P_l^m cos(m phi)
            # For m < 0: Y_{-m} = sqrt(2) N_m P_l^m sin(m phi)
            # We need to fill both positive and negative m columns
            
            # Positive m column
            pos_idx = m + l
            # Y[:, pos_idx] = sqrt2 * N_m * P_lm * cos_m[m]
            # Reshape: for each theta point, multiply P_lm[i] * cos_m[m] for all phi
            Y[:, pos_idx] = np.repeat(P_lm * sqrt2 * N_m, nphi) * np.tile(cos_m[m], ntheta)
            
            # Negative m column
            neg_idx = -m + l
            Y[:, neg_idx] = np.repeat(P_lm * sqrt2 * N_m, nphi) * np.tile(sin_m[m], ntheta)
    
    return Y


def _compute_ratio(coeffs, Y, w, dphi, nphi):
    """
    Compute R = ||f||_4 / ||f||_2 using quadrature.
    coeffs: shape (21,)
    Y: shape (npts, 21)
    w: shape (70,) - Gauss-Legendre weights
    dphi: scalar
    nphi: int
    """
    f = Y @ coeffs  # shape (npts,)
    ntheta = len(w)
    # Reshape to (ntheta, nphi) for proper weighting
    f_2d = f.reshape(ntheta, nphi)
    
    # Weights: w_i * dphi for each point
    # Integral over sphere: sum_i sum_j w_i * dphi * f(x_i, phi_j)^p
    # ||f||_2^2 = sum w_i dphi f^2
    # ||f||_4^4 = sum w_i dphi f^4
    
    weights_2d = np.tile(w * dphi, (1, nphi))  # shape (ntheta, nphi)
    
    norm2_sq = np.sum(weights_2d * f_2d**2)
    norm4_4 = np.sum(weights_2d * f_2d**4)
    
    if norm2_sq <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # (f^4)^(1/4) = |f|
    # Actually norm4 = (sum w f^4)^(1/4)
    norm4 = norm4_4 ** 0.25
    
    return norm4 / norm2


def _objective(coeffs, Y, w, dphi, nphi):
    """Objective to minimize: R = ||f||_4 / ||f||_2"""
    return _compute_ratio(coeffs, Y, w, dphi, nphi)


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 minimizing ||f||_4/||f||_2
    for f = sum c_m Y_m where Y_m are real orthonormal degree-10 spherical harmonics.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    k = 10
    x, w, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x, phi, l=k)
    nphi = len(phi)
    
    # Initial guess: all ones
    best_coeffs = np.ones(2 * k + 1, dtype=np.float64)
    best_ratio = _compute_ratio(best_coeffs, Y, w, dphi, nphi)
    
    # Use scipy optimization with multiple restarts
    from scipy.optimize import minimize
    
    # Try different initial conditions and optimize
    np.random.seed(42)
    
    n_restarts = 50
    for restart in range(n_restarts):
        if time.time() > deadline - 60.0:
            break
        
        if restart == 0:
            c0 = best_coeffs.copy()
        else:
            # Random initial condition with some structure
            # Mix of random signs and magnitudes
            c0 = np.random.randn(2 * k + 1)
            # Normalize to reasonable magnitude
            c0 = c0 / np.max(np.abs(c0)) * 0.5
        
        try:
            result = minimize(
                lambda c: _compute_ratio(c, Y, w, dphi, nphi),
                c0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_ratio:
                if np.all(np.isfinite(result.x)) and result.fun < best_ratio:
                    best_ratio = result.fun
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                lambda c: _compute_ratio(c, Y, w, dphi, nphi),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(result.x)) and result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21:
        best_coeffs = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1e-2 / max_abs)
    if max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
    
    return best_coeffs.astype(np.float64)


# EVOLVE-BLOCK-END
