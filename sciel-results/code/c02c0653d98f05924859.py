"""Tune a Fourier-invariant Hermite-polynomial times Gaussian test function
to reduce the upper bound on A(f)*A(Fourier(f)).

The notebook derives the H12 coefficient to impose P(0)=0 (i.e., g(0)=0).
We return three coefficients (degrees 0, 4, 8); the degree-12 weight is
fixed so that g(0) = 0.

The score is the upper bound on C4 (lower is better). The search score is
seed/value, so 1.0 equals the seed program and above 1.0 is an improvement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

# Physical Hermite polynomials: H_n(x) = (-1)^n e^{x^2} d^n/dx^n e^{-x^2}
# These are the ones used in the Fourier-invariant construction.
# eval_hermite from scipy gives probabilists' Hermite polynomials.
# Physical Hermite: He_n(x) = 2^{-n/2} H_n(x/sqrt(2)) where H_n are physicists'.
# Actually, scipy.special.eval_hermite(n, x) gives physicists' Hermite polynomials.
# Let me verify: eval_hermite(0,x)=1, eval_hermite(1,x)=2x, eval_hermite(2,x)=4x^2-2.
# Yes, these are physicists' Hermite polynomials.

def _hermite_phys(n, x):
    """Physicists' Hermite polynomials H_n(x)."""
    return eval_hermite(n, x)

def _g_and_fourier(a0, a4, a8, x):
    """
    Construct g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    H0(0) = 1, H4(0) = -3, H8(0) = 105, H12(0) = -10395
    g(0) = (a0*1 + a4*(-3) + a8*105 + a12*(-10395)) * 1 = 0
    => a12 = (a0 - 3*a4 + 105*a8) / 10395
    """
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    H0 = _hermite_phys(0, x)
    H4 = _hermite_phys(4, x)
    H8 = _hermite_phys(8, x)
    H12 = _hermite_phys(12, x)
    poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = poly * np.exp(-x**2 / 2.0)
    
    # Fourier transform of g. For a Fourier-invariant construction,
    # the Fourier transform of H_n(x)*exp(-x^2/2) is related to H_n.
    # Specifically, if g(x) = P(x)*exp(-x^2/2), then 
    # F(g)(xi) = (1/sqrt(2*pi)) * integral g(x) exp(-i*xi*x) dx
    # For physicists' Hermite: F[H_n(x)*exp(-x^2/2)](xi) = (-i)^n * H_n(xi) * exp(-xi^2/2) * (1/sqrt(2*pi)) * ... 
    # Actually the exact relation depends on normalization.
    # 
    # The key property: for g(x) = P(x)*exp(-x^2/2) where P is a polynomial,
    # the Fourier transform F(g)(xi) = Q(xi)*exp(-xi^2/2) for some polynomial Q.
    # For the Fourier-invariant case, we need to compute this properly.
    #
    # Let me use the known identity: 
    # If f(x) = exp(-x^2/2), then F(f)(xi) = exp(-xi^2/2) (up to normalization).
    # For f(x) = H_n(x)*exp(-x^2/2), the Fourier transform is (-i)^n * H_n(xi)*exp(-xi^2/2)
    # with appropriate normalization.
    #
    # Using the convention: F(f)(xi) = (1/sqrt(2*pi)) * integral f(x) exp(-i*xi*x) dx
    # Then F[H_n(x)*exp(-x^2/2)](xi) = (-i)^n * H_n(xi) * exp(-xi^2/2) / sqrt(2*pi) * ... 
    #
    # Actually, let me just compute it numerically for robustness, but that's slow.
    # Better: use the analytical relation.
    #
    # The standard result: with F(f)(xi) = (1/sqrt(2*pi)) integral f(x) e^{-ixi x} dx,
    # F[H_n(x) e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2}
    #
    # Wait, let me check. The Fourier transform of e^{-x^2/2} is e^{-xi^2/2}.
    # The Fourier transform of x^n e^{-x^2/2} can be computed.
    # For H_n(x) e^{-x^2/2}, the FT is (-i)^n H_n(xi) e^{-xi^2/2}.
    #
    # So if g(x) = sum_k c_k H_k(x) e^{-x^2/2}, then
    # F(g)(xi) = sum_k c_k (-i)^k H_k(xi) e^{-xi^2/2}
    
    # Fourier transform of g
    F_g = (a0*(-1j)**0*H0 + a4*(-1j)**4*H4 + a8*(-1j)**8*H8 + a12*(-1j)**12*H12) * np.exp(-x**2 / 2.0)
    # Note: (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
    # So for even n divisible by 4, the coefficient is 1.
    # (-i)^4 = ((-i)^2)^2 = (-1)^2 = 1. Yes.
    # So F_g = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2) = g(x)
    # This means g is a Fourier eigenfunction with eigenvalue 1!
    # That's the Fourier-invariant property.
    
    return g, F_g

def _compute_objective(coeffs):
    """
    Compute the upper bound on C4 given coefficients [a0, a4, a8].
    
    The bound involves A(f) * A(F(f)) where A is some norm/functional.
    Based on the task, we need to minimize the upper bound on the product.
    
    For a test function g, the relevant quantities are typically:
    - ||g||_2 or similar norms
    - The bound on the uncertainty product
    
    The specific formula from the literature (Goncalves et al. 2017) for 
    the uncertainty inequality with Hermite-Gaussian test functions:
    
    The upper bound on A(f)*A(F(f)) is related to the ratio of certain norms.
    For a normalized test function, the bound involves:
    C = ||g||_1 * ||F(g)||_1 / (||g||_2 * ||F(g)||_2) or similar.
    
    Actually, the standard approach: for the uncertainty principle,
    A(f) is often the L1 norm or a moment-based quantity.
    
    Let me think about this differently. The "upper bound on C4" likely refers
    to a specific constant in an inequality. The test function g is used to
    derive the bound, and we want to minimize that bound.
    
    A common formulation: if g is a test function with g(0)=0, then
    the uncertainty bound involves integrals of |g| and |F(g)|.
    
    For the specific problem, the objective is likely:
    objective = (integral |g(x)| dx) * (integral |F(g)(xi)| dxi)
    or some normalized version.
    
    Since g is real and the Fourier transform preserves L2 norm (Parseval),
    ||F(g)||_2 = ||g||_2.
    
    The L1 norm is not preserved by Fourier transform in general.
    
    Let me compute:
    - L1 norm of g: integral |g(x)| dx
    - L1 norm of F(g): integral |F(g)(xi)| dxi
    
    And the bound is proportional to the product of these L1 norms,
    possibly normalized by L2 norms.
    
    Actually, re-reading: "upper bound on A(f)*A(Fourier(f))" where A is
    likely the L1 norm (area under the absolute value curve).
    
    So the objective to minimize is:
    A(g) * A(F(g)) = ||g||_1 * ||F(g)||_1
    
    But we should normalize to make it scale-invariant, or the problem
    has a fixed normalization.
    
    Given the seed values [-1.13, 0.04, 1/3240], let me compute numerically.
    """
    a0, a4, a8 = coeffs
    x = np.linspace(-15, 15, 20000)
    g, F_g = _g_and_fourier(a0, a4, a8, x)
    
    # Compute L1 norms numerically
    l1_g = np.trapz(np.abs(g), x)
    l1_Fg = np.trapz(np.abs(F_g), x)
    
    # Also compute L2 norms for normalization
    l2_g = np.sqrt(np.trapz(g**2, x))
    l2_Fg = np.sqrt(np.trapz(F_g**2, x))
    
    # The bound is typically the product of L1 norms
    # Normalized: (||g||_1 / ||g||_2) * (||F(g)||_1 / ||F(g)||_2)
    # This is scale-invariant
    if l2_g < 1e-15 or l2_Fg < 1e-15:
        return 1e10
    
    norm_l1_g = l1_g / l2_g
    norm_l1_Fg = l1_Fg / l2_Fg
    
    # The upper bound on the uncertainty product
    return norm_l1_g * norm_l1_Fg

def _compute_objective_fast(coeffs):
    """Faster version using fewer points for optimization."""
    a0, a4, a8 = coeffs
    x = np.linspace(-12, 12, 10000)
    g, F_g = _g_and_fourier(a0, a4, a8, x)
    
    l1_g = np.trapz(np.abs(g), x)
    l1_Fg = np.trapz(np.abs(F_g), x)
    l2_g = np.sqrt(np.trapz(g**2, x))
    l2_Fg = np.sqrt(np.trapz(F_g**2, x))
    
    if l2_g < 1e-15 or l2_Fg < 1e-15:
        return 1e10
    
    return (l1_g / l2_g) * (l1_Fg / l2_Fg)

def construct():
    """
    Return three coefficients [a0, a4, a8] that minimize the upper bound
    on A(f)*A(Fourier(f)).
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Seed values from Goncalves et al. 2017
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_coeffs = seed.copy()
    best_score = _compute_objective(seed)
    
    def time_check():
        return time.time() < deadline - 10.0  # 10s safety margin
    
    # Local optimization from seed
    try:
        result = minimize(
            _compute_objective_fast,
            seed,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if time_check():
            score = _compute_objective_fast(result.x)
            if score < best_score:
                best_score = score
                best_coeffs = result.x.copy()
    except Exception:
        pass
    
    # Try different starting points
    if time_check():
        rng = np.random.RandomState(42)
        for trial in range(20):
            if not time_check():
                break
            # Perturb the seed
            perturbation = rng.randn(3) * np.array([0.5, 0.1, 0.01])
            x0 = seed + perturbation
            try:
                result = minimize(
                    _compute_objective_fast,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                score = _compute_objective_fast(result.x)
                if score < best_score:
                    best_score = score
                    best_coeffs = result.x.copy()
            except Exception:
                pass
    
    # Differential evolution for global search
    if time_check():
        bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.05, 0.05)]
        try:
            de_result = differential_evolution(
                _compute_objective_fast,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                polish=True
            )
            if time_check():
                score = _compute_objective_fast(de_result.x)
                if score < best_score:
                    best_score = score
                    best_coeffs = de_result.x.copy()
        except Exception:
            pass
    
    # Final polish with high precision
    if time_check():
        try:
            result = minimize(
                _compute_objective,  # High precision version
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            score = _compute_objective(result.x)
            if score < best_score:
                best_score = score
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # Final verification with high precision
    final_score = _compute_objective(best_coeffs)
    if final_score < best_score:
        best_score = final_score
    
    return best_coeffs
# EVOLVE-BLOCK-END
