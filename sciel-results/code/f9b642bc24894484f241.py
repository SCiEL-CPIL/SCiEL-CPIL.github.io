"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2*x
    elif n == 2:
        return 4*x**2 - 2
    elif n == 3:
        return 8*x**3 - 12*x
    elif n == 4:
        return 16*x**4 - 48*x**2 + 12
    elif n == 8:
        return 256*x**8 - 1344*x**6 + 2688*x**4 - 1344*x**2 + 168
    elif n == 12:
        return 4096*x**12 - 48384*x**10 + 241920*x**8 - 503808*x**6 + 453504*x**4 - 159456*x**2 + 9450
    else:
        raise ValueError(f"Unsupported Hermite order {n}")


def _compute_uncertainty_product(c0, c4, c8):
    """
    Compute A(f) * A(F̂f) for g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x²/2)
    where c12 is chosen so g(0) = 0.
    """
    # Compute c12 from g(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=945
    c12 = -(c0 * 1 + c4 * 3 + c8 * 105) / 945.0
    
    # Use numerical quadrature
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    
    # Compute Hermite polynomials
    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g = P * np.exp(-x**2/2)
    
    # L2 norm squared
    norm_sq = np.sum(g**2) * dx
    if norm_sq < 1e-15:
        return 1e10
    
    # Second moment in x-space: <x²>
    x2_moment = np.sum(x**2 * g**2) * dx / norm_sq
    
    # For the Fourier transform, use the fact that for g(x) = P(x)exp(-x²/2),
    # F̂g(k) = (1/sqrt(2π)) ∫ P(x) exp(-x²/2) exp(-ikx) dx
    # = (1/sqrt(2π)) * sum of coefficients * Fourier transform of H_n(x)exp(-x²/2)
    
    # The Fourier transform of H_n(x)exp(-x²/2) is (-i)^n sqrt(2π) H_n(k) exp(-k²/2)
    # (with physicist's convention)
    
    # So F̂g(k) = (1/sqrt(2π)) * [c0*H0(k) + c4*(-i)^4*H4(k) + c8*(-i)^8*H8(k) + c12*(-i)^12*H12(k)] * exp(-k²/2) * sqrt(2π)
    # = [c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)] * exp(-k²/2)
    # since (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    
    # Wait, that's not right. Let me be more careful.
    # The Fourier transform of H_n(x) exp(-x²/2) is (-i)^n H_n(k) exp(-k²/2)
    # with the convention F̂f(k) = (1/sqrt(2π)) ∫ f(x) exp(-ikx) dx
    
    # So F̂g(k) = (1/sqrt(2π)) * [c0*H0(k) + c4*(-i)^4*H4(k) + c8*(-i)^8*H8(k) + c12*(-i)^12*H12(k)] * exp(-k²/2) * sqrt(2π)
    # = [c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)] * exp(-k²/2)
    
    # So the Fourier transform has the same form! This is a key property.
    
    # Therefore <k²> = ∫k²|F̂g(k)|²dk / ∫|F̂g(k)|²dk
    # = ∫k²|P(k)exp(-k²/2)|²dk / ∫|P(k)exp(-k²/2)|²dk
    # = same as <x²> since the form is identical!
    
    # So A(f)*A(F̂f) = <x²> (since both are equal)
    
    return x2_moment


def construct():
    """
    Return three coefficients (c0, c4, c8) that minimize the uncertainty product.
    """
    start_time = time.time()
    deadline = 175.0  # Safety margin
    
    # Initial guess: known good values
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Objective function
    def objective(c):
        return _compute_uncertainty_product(c[0], c[1], c[2])
    
    # Initial evaluation
    best_c = x0.copy()
    best_val = objective(best_c)
    
    # Try Nelder-Mead optimization
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < best_val and time.time() - start_time < deadline:
            best_c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    np.random.seed(123)
    for _ in range(10):
        if time.time() - start_time > deadline - 30:
            break
        
        x_try = x0 + np.random.randn(3) * 0.05
        try:
            result = minimize(objective, x_try, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_val:
                best_c = result.x
                best_val = result.fun
        except Exception:
            continue
    
    return best_c
# EVOLVE-BLOCK-END
