"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Pre-compute factorial and power of 2 for Hermite norms
# Norm of H_n is sqrt(2^n * n! * sqrt(pi))
# We work with the squared norm: 2^n * n! * sqrt(pi)
# For the ratio, sqrt(pi) cancels out.

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi) which cancels in the ratio)."""
    return (2.0 ** n) * math.factorial(n)

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product Delta_x * Delta_xi for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Physicists' Hermite).
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (i)^n * H_n(xi)*exp(-xi^2/2)
    (up to a constant factor that cancels in the ratio).
    For even n, (i)^n is real: (i)^0=1, (i)^4=1, (i)^8=1, (i)^12=1.
    So the Fourier-space coefficients are the same as position-space.
    Thus Delta_x^2 = Delta_xi^2, and the product is <x^2>.
    """
    # Determine a12 from P(0) = 0
    # a0*1 + a4*12 + a8*1680 + a12*2032128 = 0
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # 1. Calculate ||f||_2^2
    norms_sq = np.array([_get_hermite_norm_sq(d) for d in degrees])
    norm_total = np.sum(coeffs**2 * norms_sq)
    
    if norm_total < 1e-30:
        return 1e10
        
    # 2. Calculate <x^2> = integral x^2 f^2 dx / ||f||_2^2
    # x^2 H_n = 1/4 H_{n+2} + n H_n + n(n-1) H_{n-2}
    # Overlaps with f (degrees 0, 4, 8, 12):
    # H_4: from x^2 H_4 -> 4*H_4
    # H_8: from x^2 H_8 -> 8*H_8
    # H_12: from x^2 H_12 -> 12*H_12
    
    coeff_x2f_H4 = 4.0 * a4
    coeff_x2f_H8 = 8.0 * a8
    coeff_x2f_H12 = 12.0 * a12
    
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    integral_x2_f2 = (coeff_x2f_H4 * a4 * norm_H4_sq + 
                      coeff_x2f_H8 * a8 * norm_H8_sq + 
                      coeff_x2f_H12 * a12 * norm_H12_sq)
    
    variance = integral_x2_f2 / norm_total
    
    return variance

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    # Seed values from "Prior state of the art"
    best_coeffs = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_val = _compute_bound(*best_coeffs)
    
    # Local refinement using L-BFGS-B with tighter tolerances
    try:
        res = minimize(
            _compute_bound,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 100}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return np.array(best_coeffs)
    
    # Global search using differential evolution
    try:
        from scipy.optimize import differential_evolution
        # Define bounds for the coefficients
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
        
        de_result = differential_evolution(
            _compute_bound,
            bounds,
            seed=42,
            maxiter=500,
            tol=1e-15,
            atol=1e-15,
            popsize=20,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube',
            updating='deferred'
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_coeffs = de_result.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return np.array(best_coeffs)
    
    # Multi-start local refinement around best solution
    rng = np.random.default_rng(42)
    for i in range(30):
        if time.time() > deadline:
            break
        # Perturb around best solution with decreasing scale
        scale = 1.0 / (1.0 + i * 0.3)
        perturbation = rng.normal(0, scale, size=3)
        trial = best_coeffs + perturbation
        trial_val = _compute_bound(*trial)
        if trial_val < best_val:
            # Refine this promising point
            try:
                res2 = minimize(
                    _compute_bound,
                    trial,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-18}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_coeffs = res2.x
            except Exception:
                pass
    
    # Check time
    if time.time() > deadline:
        return np.array(best_coeffs)
    
    # Final high-precision polish with Nelder-Mead
    try:
        res_nm = minimize(
            _compute_bound,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-20, 'fatol': 1e-20, 'adaptive': True}
        )
        
        if res_nm.fun < best_val:
            best_val = res_nm.fun
            best_coeffs = res_nm.x
    except Exception:
        pass
    
    # Final ultra-precise polish with Powell's method
    try:
        res_pw = minimize(
            _compute_bound,
            best_coeffs,
            method='Powell',
            options={'maxiter': 20000, 'xtol': 1e-20, 'ftol': 1e-20}
        )
        
        if res_pw.fun < best_val:
            best_val = res_pw.fun
            best_coeffs = res_pw.x
    except Exception:
        pass
        
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
