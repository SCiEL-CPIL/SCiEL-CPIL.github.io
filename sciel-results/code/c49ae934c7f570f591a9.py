"""Optimized Fourier-invariant Hermite-Gaussian test function for uncertainty inequality."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import time

# Deadline guard
_DEADLINE = 210.0
_START_TIME = time.time()

def _remaining_time():
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)  # 10s safety margin

def _hermite(n, x):
    """Compute physicist's Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the Hermite combination.
    
    The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    A(f) is the position spread and A(Fourier(f)) is the momentum spread.
    For a Fourier-invariant problem, we need to compute:
    - ||g||_2^2 = integral of |g(x)|^2 dx
    - ||x*g||_2^2 = integral of x^2|g(x)|^2 dx
    - ||g'||_2^2 = integral of |g'(x)|^2 dx (for momentum)
    
    The uncertainty product bound involves these quantities.
    """
    # Coefficients for H0, H4, H8
    # H12 coefficient is derived to impose g(0) = 0
    # H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1680, H_12(0)=-133056
    # g(0) = (a0*1 + a4*(-12) + a8*1680 + a12*(-133056)) * exp(0) = 0
    # So a12 = (a0 - 12*a4 + 1680*a8) / 133056
    
    a12 = (a0 - 12 * a4 + 1680 * a8) / 133056.0
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    def g(x):
        """Test function g(x) = sum(a_n * H_n(x)) * exp(-x^2/2)"""
        result = np.zeros_like(x)
        for c, n in zip(coeffs, degrees):
            result += c * _hermite(n, x)
        return result * np.exp(-x**2 / 2.0)
    
    def g_sq(x):
        """|g(x)|^2"""
        return g(x)**2
    
    def x2_g_sq(x):
        """x^2 * |g(x)|^2"""
        return x**2 * g(x)**2
    
    # Compute g'(x) analytically or numerically
    # g(x) = P(x) * exp(-x^2/2) where P(x) = sum a_n H_n(x)
    # g'(x) = P'(x) * exp(-x^2/2) + P(x) * (-x) * exp(-x^2/2)
    #       = [P'(x) - x*P(x)] * exp(-x^2/2)
    
    def P(x):
        """Polynomial part P(x) = sum a_n H_n(x)"""
        result = np.zeros_like(x)
        for c, n in zip(coeffs, degrees):
            result += c * _hermite(n, x)
        return result
    
    def P_prime(x):
        """Derivative of P(x). H_n'(x) = 2n H_{n-1}(x)"""
        result = np.zeros_like(x)
        for c, n in zip(coeffs, degrees):
            if n > 0:
                result += c * 2 * n * _hermite(n - 1, x)
        return result
    
    def g_prime(x):
        """Derivative of g(x)"""
        return (P_prime(x) - x * P(x)) * np.exp(-x**2 / 2.0)
    
    def g_prime_sq(x):
        """|g'(x)|^2"""
        return g_prime(x)**2
    
    # Use adaptive quadrature over the entire real line
    # For Gaussian-decaying functions, we can integrate over a large finite interval
    # or use quad with infinite limits
    
    # ||g||_2^2
    norm2, _ = quad(g_sq, -np.inf, np.inf, limit=200, epsabs=1e-12, epsrel=1e-12)
    
    # ||x*g||_2^2
    x2_norm2, _ = quad(x2_g_sq, -np.inf, np.inf, limit=200, epsabs=1e-12, epsrel=1e-12)
    
    # ||g'||_2^2
    deriv_norm2, _ = quad(g_prime_sq, -np.inf, np.inf, limit=200, epsabs=1e-12, epsrel=1e-12)
    
    if norm2 <= 0:
        return np.inf
    
    # Position variance: Var_x = <x^2> - <x>^2
    # By symmetry, <x> = 0, so Var_x = <x^2> = x2_norm2 / norm2
    var_x = x2_norm2 / norm2
    
    # Momentum variance: For g(x) = P(x)*exp(-x^2/2), the momentum operator is -i d/dx
    # ||g'||_2^2 / ||g||_2^2 gives <p^2> since <p> = 0 by symmetry
    var_p = deriv_norm2 / norm2
    
    # The uncertainty product is sqrt(Var_x * Var_p)
    # But the task asks for the upper bound on A(f)*A(Fourier(f))
    # A(f) is typically the spread, so A(f)*A(Fourier(f)) = sqrt(Var_x * Var_p)
    
    uncertainty_product = np.sqrt(var_x * var_p)
    
    return uncertainty_product

def construct():
    """
    Construct the three coefficients of the Hermite combination (degrees 0, 4, 8).
    The degree-12 weight is fixed so g(0) = 0.
    Returns np.array([a0, a4, a8])
    """
    # Start with the prior state of the art
    best_params = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_score = _compute_bound(*best_params)
    
    # Try local optimization from multiple starting points
    start_points = [
        best_params.copy(),
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.0001]),
        np.array([-2.0, 0.08, 0.0005]),
        np.array([-1.13, 0.04, 0.00031]),
        np.array([-0.8, 0.03, 0.0002]),
    ]
    
    for x0 in start_points:
        if _remaining_time() < 15:
            break
        
        try:
            result = minimize(
                lambda x: _compute_bound(*x),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Try a second round with tighter tolerance if time permits
    if _remaining_time() > 30:
        try:
            result = minimize(
                lambda x: _compute_bound(*x),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params
