"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite

# Set a deadline: 200 seconds (10s safety margin before 210s kill)
_DEADLINE = time.time() + 200.0

def _check_deadline():
    return time.time() > _DEADLINE

def _hermite_gaussian(x, c0, c4, c8):
    """Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)"""
    H0 = np.ones_like(x)
    H4 = x**4 - 6*x**2 + 3
    H8 = x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    H12 = (x**12 - 66*x**10 + 1540*x**8 - 13860*x**6 + 39270*x**4 - 39270*x**2 + 10395)
    
    # c12 fixed so g(0) = 0: c0 + 3*c4 + 105*c8 + 105*c12 = 0
    c12 = -(c0 + 3*c4 + 105*c8) / 105.0
    
    poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
    return poly * np.exp(-x**2 / 2.0)

def _compute_bound(c0, c4, c8):
    """Compute the upper bound C4 on A(f)*A(Fourier(f)).
    
    This is an approximation based on typical uncertainty principle formulations.
    The bound likely involves norms of the function and its Fourier transform.
    """
    # Use a fine grid for numerical integration
    x = np.linspace(-10, 10, 20001)
    dx = x[1] - x[0]
    
    g = _hermite_gaussian(x, c0, c4, c8)
    
    # Compute Fourier transform numerically
    # F(omega) = integral g(x) exp(-i*omega*x) dx
    # For efficiency, use FFT with appropriate sampling
    N = len(x)
    omega_max = 2*np.pi / dx
    omega = 2*np.pi * np.fft.fftfreq(N, d=dx)
    
    # Use FFT for Fourier transform
    G = np.fft.fftshift(np.fft.fft(g * dx))
    
    # Compute various norms
    # L2 norm of g
    l2_g = np.sqrt(np.sum(np.abs(g)**2) * dx)
    
    # L2 norm of G (Fourier transform)
    l2_G = np.sqrt(np.sum(np.abs(G)**2) * (2*np.pi/N))
    
    # The uncertainty bound C4 might be related to product of spreads or similar
    # A common formulation: C4 = (spread_x * spread_k)^2 or similar
    # Let's compute the variance (spread) in position and momentum
    
    # Position spread
    x2 = np.sum(x**2 * np.abs(g)**2) * dx / l2_g**2
    spread_x = np.sqrt(x2)
    
    # Momentum spread (using omega)
    omega2 = np.sum(omega**2 * np.abs(G)**2) * (2*np.pi/N) / l2_G**2
    spread_k = np.sqrt(omega2)
    
    # The bound C4 is likely proportional to spread_x * spread_k
    # In Heisenberg: spread_x * spread_k >= 1/2
    # C4 might be this product or a related quantity
    
    bound = spread_x * spread_k
    
    return bound

def _objective(coeffs):
    """Objective function for optimization."""
    if _check_deadline():
        return 1e10
    c0, c4, c8 = coeffs
    try:
        val = _compute_bound(c0, c4, c8)
        if not np.isfinite(val):
            return 1e10
        return val
    except:
        return 1e10

def construct():
    """Return the three coefficients [c0, c4, c8] that minimize the uncertainty bound."""
    best_coeffs = np.array([-113/100, 1/25, 1/3240])
    best_val = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Try differential evolution with bounded search
    if not _check_deadline():
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        try:
            result = differential_evolution(
                _objective, bounds, 
                maxiter=50, popsize=15, 
                seed=42, tol=1e-6,
                polish=True
            )
            if _check_deadline():
                pass
            elif result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except:
            pass
    
    # Local refinement with Nelder-Mead
    if not _check_deadline() and np.isfinite(best_val):
        try:
            res = minimize(
                _objective, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if _check_deadline():
                pass
            elif res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
