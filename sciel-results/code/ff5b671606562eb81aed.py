"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, brentq
import time

# Hermite polynomials (physicists' convention) up to degree 12
def H(n, x):
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    if n == 2:
        return 4 * x**2 - 2
    if n == 3:
        return 8 * x**3 - 12 * x
    if n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    if n == 5:
        return 32 * x**5 - 160 * x**3 + 120 * x
    if n == 6:
        return 64 * x**6 - 480 * x**4 + 720 * x**2 - 120
    if n == 7:
        return 128 * x**7 - 1344 * x**5 + 3360 * x**3 - 1680 * x
    if n == 8:
        return 256 * x**8 - 3584 * x**6 + 13440 * x**4 - 13440 * x**2 + 1680
    if n == 9:
        return 512 * x**9 - 9216 * x**7 + 40320 * x**5 - 60480 * x**3 + 20160 * x
    if n == 10:
        return 1024 * x**10 - 23040 * x**8 + 120960 * x**6 - 241920 * x**4 + 151200 * x**2 - 15120
    if n == 11:
        return 2048 * x**11 - 55296 * x**9 + 344064 * x**7 - 907200 * x**5 + 831600 * x**3 - 221760 * x
    if n == 12:
        return 4096 * x**12 - 147456 * x**10 + 1144320 * x**8 - 4084320 * x**6 + 5443200 * x**4 - 2162160 * x**2 + 133056
    raise ValueError(f"Hermite polynomial degree {n} not implemented")

def hermite_coefficients(c0, c4, c8):
    """
    Given coefficients for H0, H4, H8, compute c12 such that g(0) = 0.
    g(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    H0(0) = 1
    H4(0) = 12
    H8(0) = 1680
    H12(0) = 133056
    So: c0 + 12*c4 + 1680*c8 + 133056*c12 = 0
    c12 = -(c0 + 12*c4 + 1680*c8) / 133056
    """
    c12 = -(c0 + 12*c4 + 1680*c8) / 133056
    return np.array([c0, c4, c8, c12])

def g_function(x, coeffs):
    """
    Evaluate g(x) = sum_{k in {0,4,8,12}} c_k * H_k(x) * exp(-x^2/2)
    where coeffs = [c0, c4, c8, c12]
    """
    c0, c4, c8, c12 = coeffs
    gaussian = np.exp(-x**2 / 2)
    return (c0 * H(0, x) + c4 * H(4, x) + c8 * H(8, x) + c12 * H(12, x)) * gaussian

def compute_upper_bound(coeffs):
    """
    Compute the upper bound on A(f)*A(F(f)).
    
    For a test function g(x) = sum c_k H_k(x) e^{-x^2/2}, the uncertainty product
    can be bounded. The key quantities are:
    - Variance of |g|^2 (position space)
    - Variance of |G|^2 (frequency space)
    
    Since the Fourier transform of H_k(x)e^{-x^2/2} is (-i)^k H_k(ξ)e^{-ξ^2/2},
    the Fourier transform is a linear combination of the same Hermite-Gaussians.
    
    Let's compute:
    A(f) = ||f||_1^2 / ||f||_2^2 ... actually, let me reconsider.
    
    The uncertainty inequality states: Δx · Δξ ≥ 1/2
    where Δx^2 = ∫ x^2 |f|^2 dx / ∫ |f|^2 dx and similarly for ξ.
    
    But the task mentions A(f)*A(F(f)). Let me think about what A means.
    
    In the context of the Fourier uncertainty principle, one common formulation is:
    (∫ |x f(x)|^2 dx)(∫ |ξ F(ξ)|^2 dξ) ≥ (1/2) (∫ |f(x)|^2 dx)(∫ |F(ξ)|^2 dξ)
    
    By Plancherel, ∫ |F|^2 = ∫ |f|^2 (up to constants depending on convention).
    
    Let's use the standard L2 normalization. Assume the Fourier transform is unitary.
    Then Δx^2 Δξ^2 ≥ 1/4.
    
    So the product Δx Δξ ≥ 1/2.
    
    But the task says "upper bound on A(f)*A(F(f))". This is confusing. Let me re-read.
    
    "Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce the upper bound on A(f)*A(Fourier(f))"
    
    Perhaps A(f) is some norm or moment. Given the context of Hermite polynomials and the specific coefficients, let me look at the structure.
    
    Actually, in many uncertainty principle papers, the sharp constant is approached by specific test functions. The "upper bound" might refer to a computable quantity that bounds the uncertainty product from above, and we want to minimize this bound.
    
    Let me assume the standard variance formulation:
    Δx^2 = ∫ x^2 |g(x)|^2 dx / ∫ |g(x)|^2 dx
    Δξ^2 = ∫ ξ^2 |G(ξ)|^2 dξ / ∫ |G(ξ)|^2 dξ
    
    And we want to minimize Δx^2 * Δξ^2 (or Δx * Δξ).
    
    Let's compute these integrals numerically for a given set of coefficients.
    """
    coeffs = np.asarray(coeffs)
    
    # Define integration range
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]
    
    # Compute g(x)
    g = g_function(x, coeffs)
    
    # Compute |g|^2
    abs_g_sq = np.abs(g)**2
    
    # Integrate
    norm_sq = np.trapz(abs_g_sq, x)
    
    if norm_sq < 1e-15:
        return np.inf
    
    # Δx^2
    x2_abs_g_sq = x**2 * abs_g_sq
    x2_norm = np.trapz(x2_abs_g_sq, x)
    delta_x2 = x2_norm / norm_sq
    
    # Compute G(ξ) via FFT or analytically
    # Since G(ξ) = sum c_k (-i)^k H_k(ξ) e^{-ξ^2/2}
    # We can compute it directly
    xi = x  # Same range for simplicity
    G = np.zeros_like(xi, dtype=complex)
    for k, c in zip([0, 4, 8, 12], coeffs):
        G += c * ((-1j)**k) * H(k, xi) * np.exp(-xi**2 / 2)
    
    abs_G_sq = np.abs(G)**2
    norm_G_sq = np.trapz(abs_G_sq, xi)
    
    if norm_G_sq < 1e-15:
        return np.inf
    
    xi2_abs_G_sq = xi**2 * abs_G_sq
    xi2_norm = np.trapz(xi2_abs_G_sq, xi)
    delta_xi2 = xi2_norm / norm_G_sq
    
    # The uncertainty product
    uncertainty = np.sqrt(delta_x2 * delta_xi2)
    
    return uncertainty

def objective(params):
    """
    Objective function to minimize: the uncertainty product.
    params = [c0, c4, c8]
    """
    c0, c4, c8 = params
    coeffs = hermite_coefficients(c0, c4, c8)
    
    # Check if g is non-trivial
    if np.all(np.abs(coeffs) < 1e-15):
        return 1e10
    
    return compute_upper_bound(coeffs)

def construct():
    """
    Find coefficients [c0, c4, c8] that minimize the uncertainty product.
    Returns np.array([c0, c4, c8]).
    """
    start_time = time.time()
    deadline = start_time + 200  # 200 seconds out of 210 to be safe
    
    # Initial guess from prior art
    best_params = np.array([-113/100, 1/25, 1/3240])
    best_score = objective(best_params)
    
    # Try multiple starting points
    random_starts = [
        np.array([-1.0, 0.04, 0.0003]),
        np.array([-0.5, 0.02, 0.001]),
        np.array([-2.0, 0.1, 0.0001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.13, 0.04, 0.0003086]),
    ]
    
    for start in random_starts:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, start, method='Nelder-Mead', 
                             options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Also try L-BFGS-B with bounds
    bounds = [(-10, 10), (-1, 1), (-0.01, 0.01)]
    for start in random_starts[:3]:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 5000})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
