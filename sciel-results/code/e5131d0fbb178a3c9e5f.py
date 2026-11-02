"""Optimized program for uncertainty inequality C4 bound minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds

def _compute_C4(a0, a4, a8):
    """
    Compute the C4 upper bound for the Hermite combination with coefficients a0, a4, a8.
    
    The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    H_n are physicists' Hermite polynomials.
    H0 = 1
    H4 = 16x^4 - 48x^2 + 12
    H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    H12 = 4096x^12 - 46080x^10 + 276480x^8 - 806400x^6 + 1209600x^4 - 1036800x^2 + 40320
    
    The constraint g(0) = 0 gives:
    a0 + 12*a4 + 168*a8 + 40320*a12 = 0
    => a12 = -(a0 + 12*a4 + 168*a8) / 40320
    
    For a Fourier-invariant function (up to scaling), A(f)*A(F(f)) relates to
    the L2 norm of |f| and |F(f)|.
    
    The C4 bound involves integrals of |g(x)|^2 and |G(k)|^2 where G is the Fourier transform.
    
    For Hermite functions, the Fourier transform maps H_n(x)*exp(-x^2/2) to (-i)^n * H_n(k)*exp(-k^2/2)
    (up to normalization).
    
    Let's use the standard normalization where the Hermite functions are orthonormal:
    psi_n(x) = (1/sqrt(2^n * n! * sqrt(pi))) * H_n(x) * exp(-x^2/2)
    
    Then the Fourier transform of psi_n is (-i)^n psi_n (up to a phase convention).
    
    Actually, let me work with the specific formulation. The C4 constant relates to:
    C4 = sup |f|^4 / (|f|^2 * |F(f)|^2) type expression, or more precisely
    it's an upper bound on A(f)*A(F(f)) where A(f) = (∫|f|^2)^(1/2) / (∫|f|^4)^(1/2) or similar.
    
    Let me think about this more carefully. The uncertainty product for a function g is:
    Δx * Δξ where Δx^2 = ∫x^2|g|^2/∫|g|^2 and Δξ^2 = ∫ξ^2|G|^2/∫|G|^2
    
    But the task mentions C4 which is typically the constant in:
    (∫|f|^2)^2 ≤ C4 * ∫|f|^4 * ∫|F(f)|^4 / (∫|F(f)|^2)^2 ... 
    
    Actually, for the uncertainty inequality in the form involving A(f) = (∫|f|^2)^(1/2)/(∫|f|^4)^(1/2),
    the product A(f)*A(F(f)) has an upper bound.
    
    Let me just compute the relevant integrals numerically for a given set of coefficients.
    
    The function is g(x) = P(x) * exp(-x^2/2) where P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    
    With a12 = -(a0 + 12*a4 + 168*a8)/40320 to ensure g(0) = 0.
    
    A(g) = (∫|g(x)|^2 dx)^(1/2) / (∫|g(x)|^4 dx)^(1/2)
    A(G) = (∫|G(k)|^2 dk)^(1/2) / (∫|G(k)|^4 dk)^(1/2)
    
    The upper bound on A(g)*A(G) is what we want to minimize.
    """
    # Compute a12 from the constraint g(0) = 0
    # H0(0) = 1, H4(0) = 12, H8(0) = 168, H12(0) = 40320
    a12 = -(a0 + 12*a4 + 168*a8) / 40320.0
    
    # Define the polynomial P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    def H0(x): return np.ones_like(x, dtype=float)
    def H4(x): return 16*x**4 - 48*x**2 + 12
    def H8(x): return 256*x**8 - 1344*x**6 + 2688*x**4 - 1344*x**2 + 168
    def H12(x): return 4096*x**12 - 46080*x**10 + 276480*x**8 - 806400*x**6 + 1209600*x**4 - 1036800*x**2 + 40320
    
    def P(x):
        return a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    
    def g(x):
        return P(x) * np.exp(-x**2/2.0)
    
    # The Fourier transform of g(x) = P(x)*exp(-x^2/2)
    # Using the fact that F[H_n(x)*exp(-x^2/2)] = (-i)^n H_n(k)*exp(-k^2/2)
    # (with appropriate normalization for the Fourier transform)
    # 
    # With the convention F[f](k) = ∫ f(x) e^{-2πikx} dx, the relation is different.
    # Let's use the convention where F[f](k) = (1/sqrt(2π)) ∫ f(x) e^{-ikx} dx
    # Then F[H_n(x) e^{-x^2/2}] = (-i)^n H_n(k) e^{-k^2/2}
    #
    # So G(k) = [a0*H0(k) + a4*(-i)^4*H4(k) + a8*(-i)^8*H8(k) + a12*(-i)^12*H12(k)] * exp(-k^2/2)
    # Since (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1, we get:
    # G(k) = P(k) * exp(-k^2/2) (same form!)
    #
    # Wait, that means g and G have the same functional form (up to a constant factor).
    # This is because all the Hermite degrees are even, so the phases are all +1.
    
    # So both g(x) and G(k) have the form P(x)*exp(-x^2/2)
    # Therefore A(g) = A(G) and the product is A(g)^2
    
    # Let's compute A(g) = (∫|g|^2)^(1/2) / (∫|g|^4)^(1/2)
    
    # Use numerical integration
    x = np.linspace(-10, 10, 10001)
    dx = x[1] - x[0]
    
    g_vals = g(x)
    g2 = g_vals**2
    g4 = g_vals**4
    
    int_g2 = np.trapz(g2, x)
    int_g4 = np.trapz(g4, x)
    
    if int_g2 <= 0 or int_g4 <= 0:
        return np.inf
    
    A_g = np.sqrt(int_g2) / np.sqrt(int_g4)
    A_G = A_g  # By symmetry
    
    return A_g * A_G

def construct():
    """
    Find optimal coefficients (a0, a4, a8) to minimize the C4 bound.
    Returns an array of three coefficients.
    """
    start_time = time.time()
    deadline = 180.0  # Must finish within 180s per the task
    
    # Starting point from the seed program
    best_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _compute_C4(*best_coeffs)
    
    # Do a local optimization starting from the seed
    def objective(x):
        return _compute_C4(x[0], x[1], x[2])
    
    # Try local optimization
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_score:
            best_score = result.fun
            best_coeffs = result.x
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > deadline - 30:
        return best_coeffs
    
    # Try a few random restarts
    np.random.seed(42)
    for i in range(20):
        if time.time() - start_time > deadline - 30:
            break
        
        # Random perturbation around the current best
        scale = 0.5 / (1 + i*0.1)
        trial = best_coeffs + np.random.randn(3) * scale
        
        try:
            result = minimize(objective, trial, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() - start_time < deadline - 15:
        try:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
