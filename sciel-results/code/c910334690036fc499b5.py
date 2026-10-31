"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import hermite
import time


def _hermite(n, x):
    """Physicists' Hermite polynomial H_n(x)."""
    return hermite(n)(x)


def _compute_bound(c0, c4, c8):
    """
    Compute the upper bound C4 for the given coefficients.
    The degree-12 coefficient is fixed so that g(0) = 0.
    
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    where c12 is chosen so g(0) = 0.
    
    The bound involves A(f) * A(Fourier(f)) where f is related to g.
    For the uncertainty product bound, we need to compute:
    - A(f) = integral of |f(x)|^2 dx (or related norm)
    - A(F) = integral of |F(k)|^2 dk
    
    Actually, looking at the structure more carefully:
    The test function is g(x) * exp(-x^2/2) or similar Gaussian envelope.
    The Hermite polynomials times Gaussian form the basis.
    
    Let me reconsider. The function is:
    g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    
    With c12 chosen so g(0) = 0.
    
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 84x^10 + 1260x^8 - 6048x^6 + 12600x^4 - 10080x^2 + 945
    
    g(0) = (c0*1 + c4*3 + c8*105 + c12*945) * exp(0) = c0 + 3*c4 + 105*c8 + 945*c12 = 0
    So c12 = -(c0 + 3*c4 + 105*c8) / 945
    
    The uncertainty product bound for a function f:
    C4 = (integral x^2 |f(x)|^2 dx) * (integral k^2 |F(k)|^2 dk) / (integral |f(x)|^2 dx)^2
    
    Actually, the standard uncertainty product is:
    (Δx)^2 * (Δk)^2 where Δx^2 = <x^2> - <x>^2 and Δk^2 = <k^2> - <k>^2
    
    For the bound, we want to minimize this product.
    
    Let me define:
    f(x) = g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    
    Then:
    <x^0> = integral x^0 * |f(x)|^2 dx / integral |f(x)|^2 dx
    <x^2> = integral x^2 * |f(x)|^2 dx / integral |f(x)|^2 dx
    <x> = integral x * |f(x)|^2 dx / integral |f(x)|^2 dx
    
    Similarly for Fourier transform.
    
    The Fourier transform of H_n(x) * exp(-x^2/2) is (i)^n * H_n(k) * exp(-k^2/2) (up to normalization).
    
    So F(k) = (c0*(i)^0*H0(k) + c4*(i)^4*H4(k) + c8*(i)^8*H8(k) + c12*(i)^12*H12(k)) * exp(-k^2/2)
           = (c0*H0(k) + c4*H4(k) + c8*H8(k) + c12*H12(k)) * exp(-k^2/2)
           = g(k)  (since (i)^4 = (i)^8 = (i)^12 = 1)
    
    So F(k) = g(k), meaning the function is self-Fourier (up to sign/normalization).
    
    Therefore <k^2> = <x^2> and <k> = <x>.
    
    So the uncertainty product is:
    (Δx)^2 * (Δk)^2 = (Δx)^4 = (<x^2> - <x>^2)^2
    
    And we want to minimize (<x^2> - <x>^2)^2, or equivalently minimize |<x^2> - <x>^2|.
    
    Actually, the bound C4 is likely:
    C4 = (integral x^2 |f|^2 dx) * (integral k^2 |F|^2 dk) / (integral |f|^2 dx)^2
    
    Since F = g (self-Fourier), this becomes:
    C4 = (integral x^2 |g|^2 dx)^2 / (integral |g|^2 dx)^2
    
    Wait, that's not right either. Let me think about what A(f) means.
    
    A(f) is likely the "area" or norm: A(f) = (integral |f(x)|^2 dx)^(1/2) or similar.
    
    The product A(f) * A(F) would be:
    (integral |f|^2 dx)^(1/2) * (integral |F|^2 dk)^(1/2)
    
    By Parseval, integral |F|^2 dk = 2*pi * integral |f|^2 dx (depending on convention).
    So this product is just a constant times integral |f|^2 dx, which doesn't depend on the shape.
    
    That can't be right for an optimization problem. Let me reconsider.
    
    Perhaps A(f) involves moments:
    A(f) = (integral x^2 |f(x)|^2 dx)^(1/2) / (integral |f(x)|^2 dx)^(1/2) = RMS(x)
    
    Then A(f) * A(F) = RMS(x) * RMS(k)
    
    And we want to minimize this.
    
    Given the self-Fourier property, RMS(k) = RMS(x), so we minimize RMS(x)^2 = <x^2>.
    
    But we also have the constraint g(0) = 0, which fixes c12.
    
    Let me just compute <x^2> = integral x^2 |g(x)|^2 dx / integral |g(x)|^2 dx
    
    and minimize this.
    """
    # Compute c12 from constraint g(0) = 0
    c12 = -(c0 + 3.0 * c4 + 105.0 * c8) / 945.0
    
    # Define g(x)
    def g(x):
        return (c0 * _hermite(0, x) + 
                c4 * _hermite(4, x) + 
                c8 * _hermite(8, x) + 
                c12 * _hermite(12, x)) * np.exp(-x**2 / 2.0)
    
    # Numerical integration
    x = np.linspace(-20, 20, 20000)
    dx = x[1] - x[0]
    
    g_vals = g(x)
    norm2 = np.sum(g_vals**2) * dx
    
    if norm2 < 1e-30:
        return 1e30
    
    x2_norm2 = np.sum(x**2 * g_vals**2) * dx
    
    # The bound is likely <x^2> = x2_norm2 / norm2
    # Or it could be (x2_norm2 / norm2) * (x2_norm2 / norm2) = (x2_norm2/norm2)^2
    # Let's return <x^2> as the quantity to minimize
    return x2_norm2 / norm2


def construct():
    """
    Find the three coefficients (c0, c4, c8) that minimize the upper bound C4.
    Uses differential evolution for global search followed by local refinement.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial best from seed
    best_c = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_bound = _compute_bound(*best_c)
    
    # Define bounds for the coefficients
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    # Differential evolution for global search
    if time.time() < deadline - 30.0:
        try:
            result_de = differential_evolution(
                _compute_bound,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            if result_de.fun < best_bound:
                best_bound = result_de.fun
                best_c = result_de.x
        except Exception:
            pass
    
    # Local refinement using L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            result_local = minimize(
                _compute_bound,
                best_c,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result_local.fun < best_bound:
                best_bound = result_local.fun
                best_c = result_local.x
        except Exception:
            pass
    
    # Additional local refinement from multiple starting points
    if time.time() < deadline - 5.0:
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            start = np.random.uniform([-10, -10, -10], [10, 10, 10])
            try:
                result = minimize(
                    _compute_bound,
                    start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                if result.fun < best_bound:
                    best_bound = result.fun
                    best_c = result.x
            except Exception:
                pass
    
    return np.array(best_c)
# EVOLVE-BLOCK-END
