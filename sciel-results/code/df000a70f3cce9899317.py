import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Precompute Hermite polynomials as coefficient vectors for fast evaluation
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 26880x^4 - 6720x^2 + 1680
# H_12(x) = 4096x^12 - 122880x^10 + 1548288x^8 - 9276480x^6 + 25165824x^4 - 32125440x^2 + 4623360

H0_coeffs = np.array([1.0])  # constant
H4_coeffs = np.array([16.0, 0.0, -48.0, 0.0, 12.0])  # x^4 to x^0
H8_coeffs = np.array([256.0, 0.0, -4480.0, 0.0, 26880.0, 0.0, -6720.0, 0.0, 1680.0])
H12_coeffs = np.array([4096.0, 0.0, -122880.0, 0.0, 1548288.0, 0.0, -9276480.0, 0.0, 25165824.0, 0.0, -32125440.0, 0.0, 4623360.0])

# H_n(0) values
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4623360.0

# Precompute grid for fast L1 norm evaluation
_GRID = np.linspace(-15, 15, 30001)
_GAUSS = np.exp(-_GRID**2 / 2.0)
_DX = _GRID[1] - _GRID[0]

def _eval_poly(coeffs, x):
    """Evaluate polynomial with given coefficients (highest degree first) at x."""
    result = np.zeros_like(x)
    for c in coeffs:
        result = result * x + c
    return result

def _get_a12(a0, a4, a8):
    """Derive a12 from g(0)=0 constraint."""
    return -(a0 + a4 * H4_0 + a8 * H8_0) / H12_0

def _compute_bound(a0, a4, a8):
    """Compute A(f) * A(f_hat) using fast grid-based L1 norms."""
    a12 = _get_a12(a0, a4, a8)
    
    # Evaluate P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 on the grid
    P = (a0 * _eval_poly(H0_coeffs, _GRID) +
         a4 * _eval_poly(H4_coeffs, _GRID) +
         a8 * _eval_poly(H8_coeffs, _GRID) +
         a12 * _eval_poly(H12_coeffs, _GRID))
    
    # f(x) = P(x) * exp(-x^2/2), so |f(x)| = |P(x)| * exp(-x^2/2)
    A_f = np.sum(np.abs(P) * _GAUSS) * _DX
    
    # Since all degrees are multiples of 4, the Fourier transform has the same form
    # (i^n = 1 for n = 0, 4, 8, 12)
    A_fhat = A_f  # Same function due to symmetry
    
    return A_f * A_fhat

def _objective(x):
    """Objective function for optimization."""
    try:
        return _compute_bound(x[0], x[1], x[2])
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 190  # Safety margin before 210s kill
    
    # Seed value
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = _compute_bound(*seed_coeffs)
    
    best_value = seed_value
    best_coeffs = seed_coeffs.copy()
    
    # Global optimization using differential_evolution
    if time.time() < deadline - 30:
        try:
            bounds = [
                (-5.0, 5.0),   # a0
                (-2.0, 2.0),   # a4
                (-0.1, 0.1)    # a8
            ]
            res = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=False
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception as e:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception as e:
            pass
    
    # Additional local refinement from seed if time permits
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective,
                seed_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception as e:
            pass
    
    return best_coeffs

# EVOLVE-BLOCK-END
