"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 175.0  # seconds, safety margin before 210s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    h0_0 = 1.0
    h4_0 = -12.0
    h8_0 = 1680.0
    h12_0 = -103950.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12

def _compute_bound(c0, c4, c8):
    """Compute the upper bound on A(f)*A(F(f)) for the given coefficients."""
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Use a fine grid to compute L1 norm numerically
    x = np.linspace(-20, 20, 200001)
    dx = x[1] - x[0]
    
    # g(x) = sum c_k H_k(x) exp(-x^2/2)
    # H_k(x) for k = 0, 4, 8, 12
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 76800*x**10 + 518400*x**8 - 1548480*x**6 + 2342400*x**4 - 1548480*x**2 + 103950
    
    g = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    g = g * np.exp(-x**2 / 2.0)
    
    # L1 norm of g
    A_f = np.sum(np.abs(g)) * dx
    
    # Compute the FT numerically using FFT
    N = len(x)
    xi = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    F_g = np.fft.fft(g) * dx
    F_g = np.fft.fftshift(F_g)
    xi = np.fft.fftshift(xi)
    
    # L1 norm of FT
    dxi = xi[1] - xi[0] if len(xi) > 1 else 1.0
    A_F = np.sum(np.abs(F_g)) * dxi
    
    return A_f * A_F

def construct():
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(*seed)
    best_params = seed.copy()
    
    # Try local optimization from seed
    if _time_left() > 40:
        try:
            res = minimize(
                _compute_bound,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Try differential evolution for global search if time permits
    if _time_left() > 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-0.5, 0.5)]
            res_de = differential_evolution(
                _compute_bound,
                bounds,
                maxiter=150,
                popsize=12,
                seed=42,
                tol=1e-9,
                mutation=(0.3, 1.0)
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_params = res_de.x.copy()
        except Exception:
            pass
    
    # Refine with Nelder-Mead from best found
    if _time_left() > 25 and not np.array_equal(best_params, seed):
        try:
            res2 = minimize(
                _compute_bound,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if _time_left() > 15:
        try:
            res3 = minimize(
                _compute_bound,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-15, 'fatol': 1e-17}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
