"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import factorial, sqrt as msqrt

# Precompute constants for analytic moment computation
# Degrees present: 0, 4, 8, 12
_DEGS = [0, 4, 8, 12]
_N_DEGS = len(_DEGS)

# H_n(0) for even n: H_{2k}(0) = (-1)^k * (2k)! / k!
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

# Orthogonality: ∫ H_n H_m e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_nm
# So the "norm squared" of H_n under e^{-x^2} is: sqrt(pi) * 2^n * n!
def _norm_sq(n):
    """Compute sqrt(pi) * 2^n * n! for the Hermite polynomial H_n."""
    return msqrt(np.pi) * (2.0 ** n) * factorial(n)

# For x² H_n(x): use recurrence x² H_n = (1/2) H_{n+2} + n H_n + (n/2) H_{n-2}
# So ∫ x² H_n H_m e^{-x^2} dx = (1/2) δ_{n,m+2} * sqrt(pi)*2^{n+2}*(n+2)! + n * sqrt(pi)*2^n*n!*δ_{nm} + (n/2) δ_{n,m-2} * sqrt(pi)*2^{n-2}*(n-2)!
# Simplify: (1/2) * sqrt(pi) * 2^{n+2} * (n+2)! * δ_{n,m+2} = sqrt(pi) * 2^{n+1} * (n+2)! * δ_{n,m+2}
# And (n/2) * sqrt(pi) * 2^{n-2} * (n-2)! * δ_{n,m-2} = n * sqrt(pi) * 2^{n-3} * (n-2)! * δ_{n,m-2}

def _x2_integral(n, m):
    """Compute ∫ x² H_n(x) H_m(x) e^{-x^2} dx analytically."""
    result = 0.0
    # Term from (1/2) H_{n+2}: contributes when m = n+2
    if m == n + 2:
        result += msqrt(np.pi) * (2.0 ** (n + 1)) * factorial(n + 2)
    # Term from n H_n: contributes when m = n
    if m == n:
        result += n * msqrt(np.pi) * (2.0 ** n) * factorial(n)
    # Term from (n/2) H_{n-2}: contributes when m = n-2
    if n >= 2 and m == n - 2:
        result += n * msqrt(np.pi) * (2.0 ** (n - 3)) * factorial(n - 2)
    return result


def _compute_bound(a0, a4, a8):
    """Compute the upper bound C4 on A(f)*A(Fourier(f)) analytically.
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 chosen so g(0) = 0.
    """
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 665280.0
    
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute norm: ∫ P(x)² e^{-x^2} dx = sum_{i,j} c_i c_j ∫ H_i H_j e^{-x^2} dx
    # = sum_i c_i² * sqrt(pi) * 2^i * i!  (by orthogonality)
    norm = 0.0
    for i, deg in enumerate(_DEGS):
        norm += coeffs[i] ** 2 * _norm_sq(deg)
    
    if norm < 1e-30:
        return 1e10
    
    # Compute ∫ x² P(x)² e^{-x^2} dx = sum_{i,j} c_i c_j ∫ x² H_i H_j e^{-x^2} dx
    x2_integral = 0.0
    for i, deg_i in enumerate(_DEGS):
        for j, deg_j in enumerate(_DEGS):
            x2_integral += coeffs[i] * coeffs[j] * _x2_integral(deg_i, deg_j)
    
    # Since P is even (all Hermite degrees are even), mean_x = 0
    # var_x = <x²> = x2_integral / norm
    var_x = x2_integral / norm
    
    # Fourier side: F[H_n(x) e^{-x²/2}] = i^n H_n(ξ) e^{-ξ²/2}
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = -1
    # So P̂(ξ) = a0*H0 + a4*H4 + a8*H8 - a12*H12
    coeffs_hat = np.array([a0, a4, a8, -a12])
    
    # Same computation for the Fourier side
    norm_hat = 0.0
    for i, deg in enumerate(_DEGS):
        norm_hat += coeffs_hat[i] ** 2 * _norm_sq(deg)
    
    if norm_hat < 1e-30:
        return 1e10
    
    x2_integral_hat = 0.0
    for i, deg_i in enumerate(_DEGS):
        for j, deg_j in enumerate(_DEGS):
            x2_integral_hat += coeffs_hat[i] * coeffs_hat[j] * _x2_integral(deg_i, deg_j)
    
    var_xi = x2_integral_hat / norm_hat
    
    if var_x < 0 or var_xi < 0:
        return 1e10
    
    bound = 2.0 * np.sqrt(var_x * var_xi)
    
    return bound


def construct():
    """Optimize the coefficients to minimize the upper bound C4."""
    start_time = time.time()
    deadline = start_time + 170  # Safety margin (evaluator kills at 210s)
    
    # Initial guess
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = x0.copy()
    best_bound = _compute_bound(*x0)
    
    # Primary optimization with Nelder-Mead
    try:
        result = minimize(
            lambda x: _compute_bound(*x),
            x0,
            method='Nelder-Mead',
            options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        if result.fun < best_bound:
            best_bound = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Multi-start optimization from diverse initial points
    if time.time() < deadline - 10:
        rng = np.random.RandomState(42)
        n_starts = 50
        for trial in range(n_starts):
            if time.time() >= deadline - 5:
                break
            
            # Varying scales to explore different regions
            scale = 0.1 * (1.0 + trial * 0.3)
            x_trial = x0 + rng.normal(0, scale, size=3)
            
            try:
                b = _compute_bound(*x_trial)
                if b < best_bound:
                    best_bound = b
                    best_x = x_trial.copy()
                    
                    # Refine with Nelder-Mead
                    if time.time() < deadline - 5:
                        result = minimize(
                            lambda x: _compute_bound(*x),
                            x_trial,
                            method='Nelder-Mead',
                            options={'maxiter': 30000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                        )
                        if result.fun < best_bound:
                            best_bound = result.fun
                            best_x = result.x.copy()
            except Exception:
                pass
        
        # Also try L-BFGS-B with bounds for better convergence
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    lambda x: _compute_bound(*x),
                    best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'gtol': 1e-12}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
        
        # Final polish with very tight tolerances
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    lambda x: _compute_bound(*x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
