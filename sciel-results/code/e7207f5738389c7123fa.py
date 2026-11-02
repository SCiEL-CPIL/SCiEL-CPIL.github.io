"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Pre-compute factorial and power of 2 for Hermite norms
# Norm of H_n is sqrt(2^n * n! * sqrt(pi))
# We work with the squared norm: 2^n * n! * sqrt(pi)
# For the ratio, sqrt(pi) cancels out.

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi) which cancels in the ratio)."""
    return (2.0 ** n) * np.math.factorial(n)

def _compute_bound(a0, a4, a8):
    """
    Compute the Heisenberg Uncertainty Product (variance) for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Physicists' Hermite).
    """
    # Determine a12 from P(0) = 0
    # a0*1 + a4*12 + a8*1680 + a12*2032128 = 0
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # 1. Calculate ||f||_2^2 (numerator denominator base)
    # ||f||_2^2 = sum c_k^2 * 2^k * k! * sqrt(pi)
    # We ignore sqrt(pi) for the ratio.
    norms_sq = np.array([_get_hermite_norm_sq(d) for d in degrees])
    norm_total = np.sum(coeffs**2 * norms_sq)
    
    if norm_total < 1e-30:
        return 1e10
        
    # 2. Calculate <x^2> = integral x^2 f^2 dx / ||f||_2^2
    # Using the recurrence x^2 H_n = 1/4 H_{n+2} + n H_n + n(n-1) H_{n-2}
    # We only need the components of x^2 f that overlap with f (degrees 0, 4, 8, 12).
    # 
    # Overlaps:
    # H_0 in x^2 f: None (x^2 H_0 -> H_2).
    # H_4 in x^2 f: From x^2 H_4 term: 4 * c_4 * H_4. (x^2 H_0, H_8, H_12 don't produce H_4).
    # H_8 in x^2 f: From x^2 H_8 term: 8 * c_8 * H_8. (x^2 H_4 -> H_2, H_4, H_6. x^2 H_12 -> H_10, H_12, H_14).
    # H_12 in x^2 f: From x^2 H_12 term: 12 * c_12 * H_12.
    
    # Inner product <x^2 f, f> = sum (coeff of H_k in x^2 f) * c_k * Norm(H_k)^2
    # Note: The "coeff of H_k in x^2 f" is the scalar multiplying H_k in the expansion of x^2 f.
    
    coeff_x2f_H4 = 4.0 * a4
    coeff_x2f_H8 = 8.0 * a8
    coeff_x2f_H12 = 12.0 * a12
    
    # Norms squared for H4, H8, H12 (excluding sqrt(pi))
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    integral_x2_f2 = (coeff_x2f_H4 * a4 * norm_H4_sq + 
                      coeff_x2f_H8 * a8 * norm_H8_sq + 
                      coeff_x2f_H12 * a12 * norm_H12_sq)
    
    # Variance <x^2>
    variance = integral_x2_f2 / norm_total
    
    # The uncertainty product for self-Fourier functions is Delta_x * Delta_xi = <x^2> (since mean is 0)
    # Actually, the Heisenberg product is Delta_x * Delta_p. For f ~ e^-x^2/2, p is conjugate.
    # If f is self-Fourier, Delta_x = Delta_p. Product is <x^2>.
    
    return variance

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Seed values from "Prior state of the art"
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_bound(*best_coeffs)
    
    # Local refinement using L-BFGS-B
    # The objective is smooth and differentiable.
    try:
        res = minimize(
            _compute_bound,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return np.array(best_coeffs)
    
    # Second pass with Nelder-Mead for high precision polishing if time permits
    try:
        res_nm = minimize(
            _compute_bound,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
        )
        
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_coeffs = res_nm.x
    except Exception:
        pass
        
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
