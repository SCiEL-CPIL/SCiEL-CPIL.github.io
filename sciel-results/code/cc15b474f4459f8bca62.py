"""Rewritten: spherical-coordinate DE search for the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.special import factorial2

# ---------- Hermite polynomials (physicists' convention) ----------
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 45360x^4 - 45360x^2 + 10395

def _H0(x):
    return np.ones_like(x)

def _H4(x):
    return x**4 - 6.0*x**2 + 3.0

def _H8(x):
    return x**8 - 28.0*x**6 + 210.0*x**4 - 420.0*x**2 + 105.0

def _H12(x):
    return x**12 - 60.0*x**10 + 1260.0*x**8 - 12600.0*x**6 + 45360.0*x**4 - 45360.0*x**2 + 10395.0

# g(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
# g(0) = 0 constraint:
# H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
# a0 + 3*a4 + 105*a8 + 10395*a12 = 0
# => a12 = -(a0 + 3*a4 + 105*a8) / 10395

def _a12_from(a0, a4, a8):
    return -(a0 + 3.0*a4 + 105.0*a8) / 10395.0

# ---------- Build g(x) and its Fourier transform ----------
# g(x) = sum a_k * H_k(x) * exp(-x^2/2)  (Gaussian times Hermite)
# The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(k) * exp(-k^2/2)
# (up to normalization; for the uncertainty product we need consistent normalization)

# We'll use the convention where:
# f(x) = g(x) = sum a_k H_k(x) exp(-x^2/2)
# F(k) = (1/sqrt(2*pi)) * integral f(x) exp(-i k x) dx
# For the Hermite-Gaussian: FT{H_n(x) exp(-x^2/2)} = (-i)^n H_n(k) exp(-k^2/2)
# (with the unitary convention, this holds exactly)

# A(f) = sqrt( integral |f(x)|^2 x^2 dx / integral |f(x)|^2 dx )  (standard deviation of |f|^2)
# Actually, the uncertainty bound involves:
# A(f) = sqrt( <x^2> ) where <x^2> = integral x^2 |f(x)|^2 dx / integral |f(x)|^2 dx
# A(F) = sqrt( <k^2> ) where <k^2> = integral k^2 |F(k)|^2 dk / integral |F(k)|^2 dk
# The product A(f)*A(F) >= 1/2 (Heisenberg), but we're bounding C4 which is a 4th-moment quantity.

# For the "C4" bound, it's likely:
# C4 = ( <x^2> * <k^2> - <x k>^2 ) or similar 4th-order moment product
# Actually, the task says "upper bound on A(f)*A(Fourier(f))" and "score is the upper bound on C4"

# Let me think more carefully. The standard uncertainty principle bound:
# sigma_x * sigma_k >= 1/2
# where sigma_x = sqrt(<x^2> - <x>^2), sigma_k = sqrt(<k^2> - <k>^2)

# For an even function (which our Hermite combination is, since H0, H4, H8, H12 are all even),
# <x> = 0 and <k> = 0, so sigma_x = sqrt(<x^2>), sigma_k = sqrt(<k^2>)

# The product sigma_x * sigma_k is what we minimize.

# Let me compute this numerically on a grid.

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product A(f)*A(F) for the given coefficients."""
    a12 = _a12_from(a0, a4, a8)
    
    # Grid
    x = np.linspace(-15, 15, 4001)
    dx = x[1] - x[0]
    k = x  # by symmetry, same grid
    
    # Build g(x)
    g = a0*_H0(x) + a4*_H4(x) + a8*_H8(x) + a12*_H12(x)
    g = g * np.exp(-0.5*x**2)
    
    # Build F(k) = FT{g}(k)
    # FT{H_n(x) exp(-x^2/2)} = (-i)^n H_n(k) exp(-k^2/2)
    # For even n, (-i)^n = (-1)^(n/2)
    # n=0: (-i)^0 = 1
    # n=4: (-i)^4 = 1
    # n=8: (-i)^8 = 1
    # n=12: (-i)^12 = 1
    # All even, so all factors are +1!
    F = a0*_H0(k) + a4*_H4(k) + a8*_H8(k) + a12*_H12(k)
    F = F * np.exp(-0.5*k**2)
    
    # Compute norms
    norm_f2 = np.sum(np.abs(g)**2) * dx
    norm_F2 = np.sum(np.abs(F)**2) * dx
    
    if norm_f2 < 1e-15 or norm_F2 < 1e-15:
        return np.inf
    
    # <x^2> for f
    x2_f = np.sum(x**2 * np.abs(g)**2) * dx / norm_f2
    # <k^2> for F
    k2_F = np.sum(k**2 * np.abs(F)**2) * dx / norm_F2
    
    if x2_f < 0 or k2_F < 0:
        return np.inf
    
    return np.sqrt(x2_f * k2_F)

def _objective_spherical(theta, phi):
    """
    Spherical coordinates:
    a0 = r * sin(phi) * cos(theta)
    a4 = r * sin(phi) * sin(theta)
    a8 = r * cos(phi)
    Since the objective is homogeneous (scales as r^2), we can set r=1.
    """
    a0 = np.sin(phi) * np.cos(theta)
    a4 = np.sin(phi) * np.sin(theta)
    a8 = np.cos(phi)
    return _compute_bound(a0, a4, a8)

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # safety margin before 210s
    
    # Initial guess: the prior best
    a0_init = -113.0 / 100.0
    a4_init = 1.0 / 25.0
    a8_init = 1.0 / 3240.0
    norm_init = np.sqrt(a0_init**2 + a4_init**2 + a8_init**2)
    a0_init /= norm_init
    a4_init /= norm_init
    a8_init /= norm_init
    
    # Convert to spherical
    r = 1.0
    phi_init = np.arccos(np.clip(a8_init, -1, 1))
    if np.sin(phi_init) > 1e-10:
        theta_init = np.arctan2(a4_init, a0_init)
    else:
        theta_init = 0.0
    
    # DE bounds
    bounds = [(-np.pi, np.pi), (0, np.pi)]
    
    best_val = _compute_bound(a0_init, a4_init, a8_init)
    
    # Run differential evolution
    try:
        result = differential_evolution(
            _objective_spherical,
            bounds=bounds,
            seed=42,
            maxiter=300,
            popsize=15,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol',
            callback=lambda x, convergence: (
                None  # just for time checking
            ),
        )
        if result.success or result.fun < best_val:
            if np.isfinite(result.fun):
                best_val = result.fun
                theta_best, phi_best = result.x
                a0_best = np.sin(phi_best) * np.cos(theta_best)
                a4_best = np.sin(phi_best) * np.sin(theta_best)
                a8_best = np.cos(phi_best)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the initial guess if we're over time
        return np.array([a0_init, a4_init, a8_init])
    
    # Use the best found
    if 'theta_best' in dir() or 'phi_best' in dir():
        return np.array([a0_best, a4_best, a8_best])
    else:
        return np.array([a0_init, a4_init, a8_init])

# EVOLVE-BLOCK-END
