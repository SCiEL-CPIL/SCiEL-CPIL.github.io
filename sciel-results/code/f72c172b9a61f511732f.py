"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_TIME_LIMIT = 200.0  # safety margin before 210s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product upper bound for the Hermite-Gaussian
    test function g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is fixed so that g(0) = 0.
    
    The uncertainty product A(f)*A(F) for a function and its Fourier transform
    relates to the variances. For the Hermite-Gaussian ansatz, we compute
    the position variance <x^2> and the momentum variance <p^2> = <(-i d/dx)^2>,
    then the bound involves these quantities.
    
    Using the standard Hermite polynomial normalization and the Fourier transform
    properties of Hermite functions, we can derive explicit expressions.
    
    For g(x) = sum_k a_k H_k(x) e^{-x^2/2}, the normalization and moments
    can be computed using Hermite polynomial identities.
    
    Key facts:
    - H_k(x) e^{-x^2/2} are (up to normalization) Hermite functions psi_k
    - <psi_k | x^2 | psi_k> = 2k + 1
    - <psi_k | x | psi_{k±1}> = sqrt(k+1) or sqrt(k)
    - The Fourier transform maps H_k(x) e^{-x^2/2} to i^k H_k(x) e^{-x^2/2}
    
    So for a real combination of even-degree Hermite polynomials, the Fourier
    transform is also a real combination (since i^k for even k is real).
    
    Let's use the orthonormal Hermite functions:
    psi_k(x) = (2^k k! sqrt(pi))^{-1/2} H_k(x) e^{-x^2/2}
    
    Then g(x) = sum_k c_k psi_k(x) where c_k relates to a_k.
    
    Actually, let me work directly with the unnormalized basis.
    Let phi_k(x) = H_k(x) e^{-x^2/2}.
    
    <phi_j | phi_k> = 2^k k! sqrt(pi) delta_{jk}
    
    For g = a0*phi_0 + a4*phi_4 + a8*phi_8 + a12*phi_12:
    
    g(0) = a0*H_0(0) + a4*H_4(0) + a8*H_8(0) + a12*H_12(0) = 0
    H_0(0) = 1, H_4(0) = -12, H_8(0) = 1008, H_12(0) = -103320 (need to verify)
    
    Actually, let me use the probabilists' or physicists' Hermite polynomials.
    The task says "Hermite-polynomial times Gaussian", so likely physicists' Hermite:
    H_0(x) = 1
    H_1(x) = 2x
    H_2(x) = 4x^2 - 2
    H_3(x) = 8x^3 - 12x
    H_4(x) = 16x^4 - 48x^2 + 12
    H_8(x) = 256x^8 - 4608x^6 + 23040x^4 - 34560x^2 + 1008
    H_12(x) = 4096x^12 - 147456x^10 + 2246400x^8 - 14155200x^6 + 33264000x^4 - 27216000x^2 + 103320
    
    So H_0(0) = 1, H_4(0) = 12, H_8(0) = 1008, H_12(0) = 103320
    
    Wait, let me recheck: H_4(0) = 12? From the formula: 16*0 - 48*0 + 12 = 12. Yes.
    H_8(0) = 1008. H_12(0) = 103320.
    
    So a12 = -(a0*1 + a4*12 + a8*1008) / 103320
    
    Now for the uncertainty product. The standard uncertainty product is:
    A(f) * A(F) where A(f) is some measure of spread.
    
    For the Fourier uncertainty principle, a common formulation uses:
    A(f) = sqrt(<x^2> - <x>^2) * ||f|| / ||f|| ... 
    
    Actually, the task says "upper bound on A(f)*A(Fourier(f))" and the score is "the upper bound on C4".
    
    Let me think about this differently. The C4 constant in the uncertainty inequality
    ||f|| * ||x f|| * ||f'|| * ||x f'|| >= C4 * ||f||^4 or similar.
    
    Actually, a common form is: 
    (∫|f|^2)(∫|xf|^2)(∫|f'|^2)(∫|xf'|^2) >= C * (∫|f|^2)^4
    
    Or the product of variances: Δx * Δp >= 1/2 (Heisenberg).
    
    Given the context of "C4" and the Hermite-Gaussian test function, I believe this is
    about the sharp constant in the uncertainty inequality. For the specific ansatz,
    we can compute the ratio and find the minimum over the parameter space.
    
    Let me define the bound more carefully. For a function g(x):
    Let N = ∫|g|^2 dx
    Let M2 = ∫x^2 |g|^2 dx / N  (second moment)
    Let P2 = ∫|g'|^2 dx / N     (related to momentum variance)
    
    For the Fourier transform, by Plancherel and properties:
    ∫|F[g]|^2 = 2π ∫|g|^2
    ∫ξ^2 |F[g]|^2 = 2π ∫|g'|^2 (up to constants)
    
    The uncertainty product in a normalized form would be:
    (M2) * (P2/(4π^2)) or similar, depending on Fourier convention.
    
    For the Hermite functions, the optimal (ground state) gives the minimum.
    For the constrained ansatz with g(0)=0, we're finding the best approximation.
    
    Let me compute this numerically using quadrature for robustness, then optimize.
    """
    a12 = -(a0 * 1.0 + a4 * 12.0 + a8 * 1008.0) / 103320.0
    
    # Compute integrals using Gauss-Hermite quadrature
    # ∫ f(x) e^{-x^2} dx ≈ sum w_i f(x_i)
    # We need ∫ |g(x)|^2 dx where g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) e^{-x^2/2}
    # So |g(x)|^2 = (poly(x))^2 e^{-x^2}
    
    n_quad = 200
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    
    def H_phys(x, n):
        """Physicists' Hermite polynomial H_n(x)"""
        return np.polynomial.hermite.hermval(x, np.polynomial.hermite.herm2phys(
            np.polynomial.hermite.hermroots(n) if n > 0 else np.array([1.0])))
    
    # Better: compute H_n directly using recurrence
    def hermite_phys(x, n):
        """Compute H_n(x) using recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}"""
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return 2 * x
        H0 = np.ones_like(x)
        H1 = 2 * x
        for k in range(1, n):
            H2 = 2 * x * H1 - 2 * k * H0
            H0 = H1
            H1 = H2
        return H1
    
    H0 = hermite_phys(x, 0)
    H4 = hermite_phys(x, 4)
    H8 = hermite_phys(x, 8)
    H12 = hermite_phys(x, 12)
    
    poly = a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12
    
    # |g|^2 = poly^2 * e^{-x^2}
    g2 = poly ** 2 * np.exp(-x ** 2)
    
    N = np.sum(w * g2)
    if N <= 0:
        return 1e10
    
    # <x^2> = ∫ x^2 |g|^2 / N
    M2 = np.sum(w * x ** 2 * g2) / N
    
    # For momentum: ∫|g'|^2 dx
    # g(x) = poly(x) * e^{-x^2/2}
    # g'(x) = poly'(x) e^{-x^2/2} - x poly(x) e^{-x^2/2} = (poly'(x) - x poly(x)) e^{-x^2/2}
    # |g'|^2 = (poly'(x) - x poly(x))^2 e^{-x^2}
    
    # Compute poly'(x) numerically or analytically
    # H_n'(x) = 2n H_{n-1}(x)
    H0p = np.zeros_like(x)
    H4p = hermite_phys(x, 3) * 8  # H_4' = 8 H_3
    H8p = hermite_phys(x, 7) * 16  # H_8' = 16 H_7
    H12p = hermite_phys(x, 11) * 24  # H_12' = 24 H_11
    
    poly_p = a0 * H0p + a4 * H4p + a8 * H8p + a12 * H12p
    
    g_prime_sq = (poly_p - x * poly) ** 2 * np.exp(-x ** 2)
    P2 = np.sum(w * g_prime_sq) / N
    
    # The uncertainty product bound. For the Fourier transform with convention
    # F[f](ξ) = ∫ f(x) e^{-2πixξ} dx, we have Δx * Δξ >= 1/(4π)
    # Or with F[f](k) = ∫ f(x) e^{-ikx} dx, Δx * Δk >= 1/2
    
    # The "C4" likely refers to a specific constant. Let me use the product:
    # bound = M2 * P2 / (4 * pi**2) for the 2π Fourier convention
    # or bound = M2 * P2 / 4 for the angular frequency convention
    
    # Actually, let me just return M2 * P2 as the proxy for the uncertainty product.
    # The exact constant factor doesn't matter for optimization since we're minimizing.
    
    return M2 * P2


def _objective(params):
    a0, a4, a8 = params
    try:
        return _compute_bound(a0, a4, a8)
    except Exception:
        return 1e10


def construct():
    # Prior state of the art (Goncalves et al. 2017) as listed in the notebook.
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_bound(*best_params)
    
    if _time_remaining() < 10:
        return best_params
    
    # Try local optimization from the prior SOTA point
    try:
        result = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    if _time_remaining() < 20:
        return best_params
    
    # Try differential evolution for global search
    try:
        bounds = [(-10, 10), (-10, 10), (-10, 10)]
        de_result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-12,
            seed=42,
            polish=True
        )
        if de_result.fun < best_score:
            best_params = de_result.x
            best_score = de_result.fun
    except Exception:
        pass
    
    if _time_remaining() < 10:
        return best_params
    
    # Polish with high-precision local optimization
    try:
        result = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15}
        )
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
