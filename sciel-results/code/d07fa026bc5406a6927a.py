"""Rewritten program: refined optimization for Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution, basinhopping

# Pre-compute norms for Hermite polynomials
# ||H_n||^2 = 2^n * n! * sqrt(pi)
# For ratios, sqrt(pi) cancels, so we use 2^n * n!

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi))."""
    return (2.0 ** n) * math.factorial(n)

# Pre-compute constants
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 2032128.0

_norm_sq = {
    0: _get_hermite_norm_sq(0),
    4: _get_hermite_norm_sq(4),
    8: _get_hermite_norm_sq(8),
    12: _get_hermite_norm_sq(12)
}

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product Delta_x * Delta_xi for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    
    For even n, the Fourier transform of H_n(x)*exp(-x^2/2) is (i)^n * H_n(xi)*exp(-xi^2/2).
    For n in {0, 4, 8, 12}, (i)^n = 1, so the Fourier-space coefficients are the same.
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
    # Only the n H_n term contributes to the overlap with f, because:
    # - 1/4 H_{n+2}: for n=0 -> H_2 (not in f), n=4 -> H_6 (not in f), n=8 -> H_10 (not in f), n=12 -> H_14 (not in f)
    # - n(n-1) H_{n-2}: for n=0 -> 0, n=4 -> 12*H_2 (not in f), n=8 -> 56*H_6 (not in f), n=12 -> 132*H_10 (not in f)
    # So only the n H_n term contributes.
    
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
    deadline = start_time + 200.0  # Safety margin before 210s limit
    
    # Seed values from "Prior state of the art"
    seed_coeffs = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _compute_bound(*seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    # 1. Initial Local Refinement from Seed
    try:
        res = minimize(
            _compute_bound,
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass

    if time_exceeded():
        return np.array(best_coeffs)

    # 2. Multi-Start Differential Evolution Global Search
    rng = np.random.default_rng(42)
    
    try:
        de_result = differential_evolution(
            _compute_bound,
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            seed=42,
            maxiter=300,
            tol=1e-14,
            atol=1e-14,
            popsize=40,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            init='latinhypercube',
            updating='deferred'
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_coeffs = de_result.x
    except Exception:
        pass

    if time_exceeded():
        return np.array(best_coeffs)

    # 3. Basin Hopping for further refinement
    try:
        res_bh = basinhopping(
            _compute_bound,
            x0=best_coeffs,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'ftol': 1e-15, 'gtol': 1e-15}},
            seed=42
        )
        if res_bh.fun < best_val:
            best_val = res_bh.fun
            best_coeffs = res_bh.x
    except Exception:
        pass

    if time_exceeded():
        return np.array(best_coeffs)

    # 4. Final High-Precision Polish
    try:
        res_pw = minimize(
            _compute_bound,
            best_coeffs,
            method='Powell',
            options={'maxiter': 10000, 'xtol': 1e-16, 'ftol': 1e-16}
        )
        if res_pw.fun < best_val:
            best_val = res_pw.fun
            best_coeffs = res_pw.x
    except Exception:
        pass

    # 5. One last check with L-BFGS-B for very tight convergence
    try:
        res_final = minimize(
            _compute_bound,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-18}
        )
        if res_final.fun < best_val:
            best_val = res_final.fun
            best_coeffs = res_final.x
    except Exception:
        pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
