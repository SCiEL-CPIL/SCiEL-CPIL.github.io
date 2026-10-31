"""
Optimized coefficients for the uncertainty inequality test function.
The task asks for three coefficients (a0, a4, a8) of a Hermite-Gaussian
combination where the a12 coefficient is fixed to enforce g(0) = 0.
The goal is to minimize the upper bound on A(f)*A(Fourier(f)).
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial

def _hermite(n, x):
    """Compute the n-th Hermite polynomial H_n(x) using the physicists' convention."""
    return eval_hermite(n, x)

def _compute_bound(coeffs):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for a given set of coefficients.
    
    The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is determined by g(0) = 0.
    
    A(f) = max |f(x)| / (max |f(x)| - min |f(x)|)  -- Actually, let's use the standard definition
    A(f) = sup_x |f(x)| / inf_x |f(x)| if inf > 0, but typically for these problems:
    A(f) = (max |f| - min |f|) / (2 * min |f|) or similar.
    
    Actually, for the uncertainty inequality in this context (Goncalves et al.),
    A(f) is typically defined as the ratio of the L-infinity norm to the L-1 norm,
    or more commonly for Hermite functions, it relates to the concentration.
    
    Let me reconsider. The standard "uncertainty inequality" bound for Hermite functions
    involves the product of the "spreads" or "widths". For a function f,
    A(f) is often defined as ||f||_inf / ||f||_1 or similar measures of concentration.
    
    However, looking at the seed program and the context of "upper bound on A(f)*A(Fourier(f))",
    this likely refers to the product of the "A" quantities where A(f) = sup|f| / (sup|f| - inf|f|)
    or a similar ratio that measures how "peaked" the function is.
    
    For Hermite-Gaussian functions, a common definition is:
    A(f) = max_x |f(x)| / min_x |f(x)|  (if f doesn't change sign and min > 0)
    
    But g(0) = 0 is imposed, so the function changes sign or has a zero at the origin.
    
    Let me use a more robust definition. In the context of these optimization problems
    for uncertainty inequalities, A(f) is typically:
    A(f) = ||f||_L∞ / ||f||_L1
    
    And the Fourier transform of a Hermite-Gaussian is another Hermite-Gaussian,
    so we can compute both analytically.
    
    For g(x) = sum_{k=0,4,8,12} a_k H_k(x) e^{-x^2/2},
    the Fourier transform (with appropriate normalization) is:
    F(g)(ξ) = sum_{k=0,4,8,12} a_k (-i)^k H_k(ξ) e^{-ξ^2/2}
    
    (up to normalization constants depending on the Fourier transform convention)
    
    Let's use the convention where the Fourier transform of H_n(x) e^{-x^2/2} is
    (-i)^n H_n(ξ) e^{-ξ^2/2} (with the standard physics convention).
    
    Then:
    ||g||_∞ = max_x |g(x)|
    ||g||_1 = ∫ |g(x)| dx
    
    A(g) = ||g||_∞ / ||g||_1
    A(F(g)) = ||F(g)||_∞ / ||F(g)||_1
    
    The bound is A(g) * A(F(g)).
    """
    a0, a4, a8 = coeffs
    
    # Determine a12 such that g(0) = 0
    # g(0) = (a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0)) * exp(0)
    # H0(0) = 1
    # H4(0) = -12
    # H8(0) = 13440
    # H12(0) = -10321920
    H0_0 = _hermite(0, 0)
    H4_0 = _hermite(4, 0)
    H8_0 = _hermite(8, 0)
    H12_0 = _hermite(12, 0)
    
    if abs(H12_0) < 1e-15:
        return 1e10  # Degenerate case
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Define g(x) and its Fourier transform F(g)(x)
    def g(x):
        return (a0 * _hermite(0, x) + a4 * _hermite(4, x) + 
                a8 * _hermite(8, x) + a12 * _hermite(12, x)) * np.exp(-x**2 / 2)
    
    def F_g(x):
        # Fourier transform: F(H_n(x) e^{-x^2/2}) = (-i)^n H_n(x) e^{-x^2/2}
        # For real-valued bound, we take absolute values, so (-i)^n becomes a phase
        # The magnitude is the same as g(x) but with signs flipped for odd n.
        # Since all n are even (0,4,8,12), (-i)^n is real: 1, -1, 1, -1
        # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
        # Actually: (-i)^0 = 1, (-i)^2 = -1, (-i)^4 = 1, (-i)^6 = -1, etc.
        # So (-i)^n for even n: if n ≡ 0 mod 4, (-i)^n = 1; if n ≡ 2 mod 4, (-i)^n = -1
        # 0 mod 4: 0 -> 1
        # 4 mod 4: 0 -> 1
        # 8 mod 4: 0 -> 1
        # 12 mod 4: 0 -> 1
        # So all coefficients have the same sign in the Fourier transform!
        return (a0 * _hermite(0, x) + a4 * _hermite(4, x) + 
                a8 * _hermite(8, x) + a12 * _hermite(12, x)) * np.exp(-x**2 / 2)
    
    # Compute ||g||_∞ and ||g||_1
    # Use numerical integration over a sufficiently large domain
    x = np.linspace(-20, 20, 10000)
    dx = x[1] - x[0]
    
    g_vals = g(x)
    F_g_vals = F_g(x)
    
    # L-infinity norm
    inf_g = np.max(np.abs(g_vals))
    inf_F_g = np.max(np.abs(F_g_vals))
    
    # L-1 norm (numerical integration)
    l1_g = np.sum(np.abs(g_vals)) * dx
    l1_F_g = np.sum(np.abs(F_g_vals)) * dx
    
    # Avoid division by zero
    if l1_g < 1e-15 or l1_F_g < 1e-15:
        return 1e10
    
    A_g = inf_g / l1_g
    A_F_g = inf_F_g / l1_F_g
    
    return A_g * A_F_g

def construct():
    """
    Return the three coefficients (a0, a4, a8) that minimize the upper bound.
    """
    start_time = time.time()
    deadline = 180.0  # Must finish within 180 seconds
    
    # Initial guess from the seed program
    x0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple starting points to avoid local minima
    best_x = x0.copy()
    best_val = _compute_bound(x0)
    
    # Generate several starting points
    np.random.seed(42)
    for i in range(50):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random perturbations around the initial guess
        scale = 1.0 / (1 + i)
        x_try = x0 + np.random.randn(3) * scale
        
        try:
            val = _compute_bound(x_try)
            if val < best_val:
                best_val = val
                best_x = x_try.copy()
        except Exception:
            continue
    
    # Local optimization from the best starting point
    if time.time() - start_time < deadline - 30:
        try:
            result = minimize(_compute_bound, best_x, method='Nelder-Mead', 
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Additional refinement
    if time.time() - start_time < deadline - 15:
        try:
            result = minimize(_compute_bound, best_x, method='Powell', 
                            options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass
    
    # Final check
    if time.time() - start_time < deadline - 5:
        try:
            final_val = _compute_bound(best_x)
            if final_val < best_val:
                best_val = final_val
        except Exception:
            pass
    
    return np.array(best_x)
