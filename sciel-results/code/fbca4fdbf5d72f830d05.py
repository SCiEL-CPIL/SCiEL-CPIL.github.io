"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
import time

def _hermite_coeffs(a0, a4, a8):
    """Return coefficient array for P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12 with P(0)=0."""
    # H0(x) = 1, H4(x) = x^4 - 6x^2 + 3, H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 84x^10 + 2520x^8 - 25200x^6 + 90720x^4 - 120960x^2 + 10395
    # At x=0: H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    # P(0) = a0*1 + a4*3 + a8*105 + a12*10395 = 0
    # => a12 = -(a0 + 3*a4 + 105*a8) / 10395
    a12 = -(a0 + 3*a4 + 105*a8) / 10395.0
    return np.array([a0, a4, a8, a12])

def _build_g(x, coeffs):
    """Construct g(x) = P(x) * exp(-x^2/2) where P is the Hermite polynomial combination."""
    a0, a4, a8, a12 = coeffs
    H0 = np.ones_like(x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    return P * np.exp(-x**2 / 2.0)

def _compute_bound(coeffs):
    """Compute the upper bound C4 for the uncertainty product A(f)*A(Fourier(f)).
    
    Based on the uncertainty inequality context, this likely involves:
    - L1 norm of g and its Fourier transform, or
    - Related integral quantities
    
    We'll compute multiple candidate bounds and use the most relevant one.
    """
    # Use a fine grid for numerical integration
    x = np.linspace(-15, 15, 20001)
    dx = x[1] - x[0]
    
    g = _build_g(x, coeffs)
    
    # Compute Fourier transform using FFT with appropriate scaling
    # The Fourier transform convention: F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx
    # For our Gaussian-weighted functions, we can compute numerically
    
    N = len(x)
    G = np.fft.fft(g) * dx
    # Shift to center frequency
    G = np.fft.fftshift(G)
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # Compute L1 norms
    l1_g = np.sum(np.abs(g)) * dx
    l1_G = np.sum(np.abs(G)) * dx
    
    # The uncertainty product bound typically involves these quantities
    # A common form: A(f) * A(F(f)) >= C, where A involves L1 or related norms
    
    # Try different candidate bounds
    # Bound 1: Product of L1 norms
    bound1 = l1_g * l1_G
    
    # Bound 2: Based on second moments (variance-based uncertainty)
    x2 = x**2
    moment2_g = np.sum(x2 * g**2) * dx / (np.sum(g**2) * dx + 1e-300)
    freq2 = freq**2
    moment2_G = np.sum(freq2 * np.abs(G)**2) * dx / (np.sum(np.abs(G)**2) * dx + 1e-300)
    bound2 = np.sqrt(moment2_g) * np.sqrt(moment2_G)
    
    # Return the product that represents the upper bound we want to minimize
    # Based on the task description, we want to minimize the upper bound
    return bound1

def _objective(params):
    """Objective function: compute the bound for given (a0, a4, a8)."""
    coeffs = _hermite_coeffs(params[0], params[1], params[2])
    return _compute_bound(coeffs)

def construct():
    """Find optimal coefficients that minimize the uncertainty bound."""
    t_start = time.time()
    deadline = t_start + 180.0  # Safety margin before 210s kill
    
    # Initial guess: prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_params = x0.copy()
    best_val = _objective(best_params)
    
    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 60.0:
        try:
            result = differential_evolution(
                _objective,
                bounds=[(-5, 5), (-5, 5), (-5, 5)],
                seed=42,
                maxiter=50,
                tol=1e-6,
                polish=False,
                updating='deferred'
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
    
    # Phase 3: Final polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_params = result.x
                best_val = result.fun
        except Exception:
            pass
    
    return np.array([best_params[0], best_params[1], best_params[2]])
# EVOLVE-BLOCK-END
