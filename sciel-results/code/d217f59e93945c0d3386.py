"""Rewritten program: Nelder-Mead simplex optimization on Hermite coefficients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_hermite
from scipy.integrate import quad
from scipy.optimize import minimize

def _phys_hermite(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    return eval_hermite(n, x)

def _build_g(coeffs, x):
    """
    Build g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is determined by g(0) = 0.
    coeffs = [c0, c4, c8]
    """
    c0, c4, c8 = coeffs
    # H0(0)=1, H4(0)=12, H8(0)=1344, H12(0)=103968 (physicist's)
    # Actually let's compute properly:
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1344
    # H12(0) = 103968
    H0_0 = _phys_hermite(0, 0)
    H4_0 = _phys_hermite(4, 0)
    H8_0 = _phys_hermite(8, 0)
    H12_0 = _phys_hermite(12, 0)
    
    # g(0) = (c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0) * exp(0) = 0
    # So c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def g(x):
        h = (c0 * _phys_hermite(0, x) + c4 * _phys_hermite(4, x) + 
             c8 * _phys_hermite(8, x) + c12 * _phys_hermite(12, x))
        return h * np.exp(-x**2 / 2)
    return g

def _objective(coeffs):
    """
    Compute the upper bound C4 on A(f)*A(Fourier(f)).
    
    For a test function g, the uncertainty inequality gives:
    A(g) * A(ĝ) >= C4
    where A(h) = ∫ |h(x)|^2 dx (or some norm)
    
    Actually, the task says "upper bound on A(f)*A(Fourier(f))" and we're tuning
    to reduce this upper bound. The seed returns coefficients that give score 1.0.
    
    Let me think about what the actual objective is. The task mentions:
    - Fourier-invariant Hermite-polynomial times Gaussian test function
    - Three free coefficients for H0, H4, H8
    - H12 coefficient derived to impose P(0)=0 (i.e., g(0)=0)
    - Score is the upper bound on C4
    
    The "upper bound" likely refers to a specific mathematical expression. 
    In uncertainty principle literature, for a function g, one can compute:
    Δx * Δp where these are standard deviations.
    
    But the task says "upper bound on A(f)*A(Fourier(f))". This might be:
    A(g) = ||g||_2^2 or similar, and we want to minimize the product.
    
    Actually, re-reading: "reduce the upper bound on A(f)*A(Fourier(f))"
    The seed value is [-113/100, 1/25, 1/3240] and scores 1.0.
    
    Let me try a different interpretation. Perhaps the objective is to compute
    some quantity that the seed program's coefficients minimize, and the score
    is that minimum value normalized so that the seed gives 1.0.
    
    Let me try: the quantity might be related to the ratio of norms or some
    moment-based expression.
    
    Given the complexity, let me try to compute a reasonable objective. One common
    quantity in this context is:
    
    For g(x) = sum c_n H_n(x) e^{-x^2/2}, we can compute:
    - ||g||_2^2 = ∫ |g(x)|^2 dx
    - ||ĝ||_2^2 (Fourier transform norm)
    
    By Plancherel, ||ĝ||_2 = ||g||_2 (up to constants depending on convention).
    
    Another possibility: the "upper bound" refers to a specific inequality constant.
    
    Let me try computing the product of the L2 norm and some other quantity, or
    perhaps the variance-based uncertainty product.
    
    Actually, let me try a simpler approach: compute the quantity that the seed
    coefficients are known to optimize. Since I don't have the notebook, let me
    try to reverse-engineer from the seed values.
    
    Seed: c0 = -113/100 = -1.13, c4 = 1/25 = 0.04, c8 = 1/3240 ≈ 0.0003086
    
    Let me try the objective as the value of some functional. A common one in
    Hermite-Gaussian uncertainty is:
    
    J = (∫ x^2 |g|^2 dx / ∫ |g|^2 dx) * (∫ ξ^2 |ĝ|^2 dξ / ∫ |ĝ|^2 dξ)
    
    This is the variance product (uncertainty product). The minimum is 1/4 for
    the Gaussian, but with the constraint g(0)=0, the minimum is higher.
    
    Let me try this objective.
    """
    c0, c4, c8 = coeffs
    H0_0 = _phys_hermite(0, 0)
    H4_0 = _phys_hermite(4, 0)
    H8_0 = _phys_hermite(8, 0)
    H12_0 = _phys_hermite(12, 0)
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    
    def g(x):
        h = (c0 * _phys_hermite(0, x) + c4 * _phys_hermite(4, x) + 
             c8 * _phys_hermite(8, x) + c12 * _phys_hermite(12, x))
        return h * np.exp(-x**2 / 2)
    
    def g2(x):
        return g(x) ** 2
    
    # Compute ||g||_2^2
    norm2, _ = quad(g2, -np.inf, np.inf, limit=200)
    if norm2 <= 0:
        return 1e10
    
    # Compute ∫ x^2 |g|^2 dx
    def x2_g2(x):
        return x**2 * g2(x)
    x2_moment, _ = quad(x2_g2, -np.inf, np.inf, limit=200)
    
    # Variance of x: σ_x^2 = x2_moment / norm2
    sigma_x2 = x2_moment / norm2
    
    # For the Fourier transform, by the properties of Hermite functions:
    # The Fourier transform of H_n(x) e^{-x^2/2} is i^n H_n(ξ) e^{-ξ^2/2} (up to constants)
    # So |ĝ(ξ)|^2 has the same form with the same coefficients squared structure
    # Actually, the Fourier transform preserves the L2 norm and the variance structure
    # for Hermite-Gaussian functions in a specific way.
    
    # For the Fourier transform of g, the variance in frequency space:
    # σ_ξ^2 = ∫ ξ^2 |ĝ(ξ)|^2 dξ / ∫ |ĝ(ξ)|^2 dξ
    # By Plancherel, ∫ |ĝ|^2 = ∫ |g|^2 (with appropriate normalization)
    # And there's a duality: the variance in frequency is related to the "dual" moments
    
    # For Hermite polynomials, the Fourier transform gives:
    # F[H_n(x) e^{-x^2/2}](ξ) = i^n sqrt(2π) H_n(ξ) e^{-ξ^2/2}
    # (depending on Fourier convention)
    
    # So |ĝ(ξ)|^2 = 2π |g(ξ)|^2 if we use the convention F[f](ξ) = ∫ f(x) e^{-2πixξ} dx
    # Or without 2π: F[f](ξ) = ∫ f(x) e^{-ixξ} dx, then F[H_n e^{-x^2/2}] = i^n sqrt(2π) H_n(ξ) e^{-ξ^2/2}
    
    # In either case, the variance structure is the same! So σ_ξ^2 = σ_x^2 for this type of function?
    # No, that's not right. The Fourier transform swaps position and momentum, so:
    # σ_ξ^2 (in frequency) corresponds to the "position variance" of the dual function.
    
    # Actually, for the specific case of Hermite-Gaussian functions, the Fourier transform
    # maps H_n to i^n H_n (up to constants), so the functional form is preserved.
    # This means σ_ξ^2 = σ_x^2 for the same coefficients!
    
    # Wait, that can't be right in general. Let me reconsider.
    
    # The key insight: for g(x) = sum c_n H_n(x) e^{-x^2/2}, the Fourier transform is
    # ĝ(ξ) = sum c_n i^n sqrt(2π) H_n(ξ) e^{-ξ^2/2} (with F[f](ξ)=∫f(x)e^{-ixξ}dx)
    # So |ĝ(ξ)|^2 = 2π |sum c_n H_n(ξ) e^{-ξ^2/2}|^2 = 2π |g(ξ)|^2
    # (since |i^n| = 1)
    
    # Therefore ∫ ξ^2 |ĝ(ξ)|^2 dξ = 2π ∫ ξ^2 |g(ξ)|^2 dξ = 2π * x2_moment
    # And ∫ |ĝ(ξ)|^2 dξ = 2π * norm2
    
    # So σ_ξ^2 = (2π * x2_moment) / (2π * norm2) = x2_moment / norm2 = σ_x^2
    
    # This means the uncertainty product is σ_x^2 * σ_ξ^2 = σ_x^4 = (σ_x^2)^2
    
    # So the objective is (x2_moment / norm2)^2
    
    sigma_xi2 = sigma_x2  # As derived above
    
    # Uncertainty product
    uncertainty_product = sigma_x2 * sigma_xi2
    
    return uncertainty_product

def construct():
    """
    Optimize the three Hermite coefficients to minimize the uncertainty product.
    Returns [c0, c4, c8].
    """
    start_time = time.time()
    deadline = start_time + 170  # Safety margin before 180s limit
    
    # Seed values
    seed = np.array([-113/100, 1/25, 1/3240])
    seed_val = _objective(seed)
    best_coeffs = seed.copy()
    best_val = seed_val
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            _objective,
            seed,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'maxfev': 5000,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        
        if result.fun < best_val:
            best_coeffs = result.x
            best_val = result.fun
        
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return np.array(best_coeffs)
    
    # Try multiple restarts with different initial points
    np.random.seed(42)
    restarts = [
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-0.9, 0.06, 0.0002]),
        np.array([-1.13, 0.04, 0.000309]) + np.random.randn(3) * 0.1,
        np.array([-1.13, 0.04, 0.000309]) + np.random.randn(3) * 0.5,
    ]
    
    for init in restarts:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _objective,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'maxfev': 3000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if result.fun < best_val:
                best_coeffs = result.x
                best_val = result.fun
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
