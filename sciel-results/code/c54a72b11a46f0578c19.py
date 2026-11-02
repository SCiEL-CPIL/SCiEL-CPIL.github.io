"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
import time
import math


def _hermite_combo(x, c0, c4, c8, c12):
    """
    Compute g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    using physicist's Hermite polynomials.
    """
    h0 = eval_hermite(0, x)
    h4 = eval_hermite(4, x)
    h8 = eval_hermite(8, x)
    h12 = eval_hermite(12, x)
    poly = c0 * h0 + c4 * h4 + c8 * h8 + c12 * h12
    return poly * np.exp(-x * x / 2.0)


def _build_g(c0, c4, c8):
    """
    Construct g(x) with the constraint g(0) = 0.
    H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-4790016 (physicist's convention).
    So c0*1 + c4*(-12) + c8*1680 + c12*(-4790016) = 0
    => c12 = (c0 - 12*c4 + 1680*c8) / 4790016
    """
    c12 = (c0 - 12.0 * c4 + 1680.0 * c8) / 4790016.0
    return c0, c4, c8, c12


def _upper_bound(c0, c4, c8, c12):
    """
    Compute an upper bound on A(f)*A(F(f)) for the Fourier-invariant Hermite-Gaussian test function.
    
    The key quantity is C4 = sup_x |g(x)| * sup_x |G(x)| where G is the Fourier transform.
    For Fourier-invariant functions (in the sense of the uncertainty principle context),
    we use the L-infinity norms of g and its Fourier transform.
    
    Actually, for the uncertainty inequality, the relevant quantity is:
    A(f) = sup_x |f(x)|, A(F(f)) = sup_x |F(f)(x)|
    and we want to minimize A(f)*A(F(f)).
    
    For g(x) = P(x)*exp(-x^2/2), the Fourier transform is also a Hermite-Gaussian combination.
    We compute the L-infinity norms numerically.
    """
    # Define a grid for numerical evaluation
    # The function decays as exp(-x^2/2), so we need a range that captures the tails
    # but the polynomial can make it grow initially. Let's use a reasonably wide grid.
    x = np.linspace(-12, 12, 20001)
    g = _hermite_combo(x, c0, c4, c8, c12)
    
    # Fourier transform of g(x) = P(x)*exp(-x^2/2)
    # The FT of exp(-x^2/2) is sqrt(2*pi)*exp(-k^2/2)
    # The FT of x^n * exp(-x^2/2) involves Hermite polynomials in k
    # Specifically, FT[x^n exp(-x^2/2)] = (-i)^n * H_n(k) * sqrt(2*pi) * exp(-k^2/2)
    # Wait, let me be more careful. The standard result is:
    # FT[x^n exp(-x^2/2)] = (-i)^n * H_n(k) * sqrt(2*pi) * exp(-k^2/2)
    # where H_n are physicist's Hermite polynomials.
    
    # Actually, let's use the property that the Fourier transform of a Hermite function
    # psi_n(x) = (1/sqrt(2^n n! sqrt(pi))) * H_n(x) * exp(-x^2/2)
    # is (-i)^n * psi_n(x).
    
    # For our case, g(x) = P(x) * exp(-x^2/2) where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # The FT of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(k) * sqrt(2*pi) * exp(-k^2/2)
    # Wait, I need to check the normalization.
    
    # Let's use: if f(x) = H_n(x) * exp(-x^2/2), then F(f)(k) = (-i)^n * sqrt(2*pi) * H_n(k) * exp(-k^2/2)
    # This is for the convention F(f)(k) = integral f(x) exp(-ikx) dx
    
    # So G(k) = sqrt(2*pi) * exp(-k^2/2) * [c0*H0(k) + (-i)^4*c4*H4(k) + (-i)^8*c8*H8(k) + (-i)^12*c12*H12(k)]
    # (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = -1
    # So G(k) = sqrt(2*pi) * exp(-k^2/2) * [c0*H0(k) + c4*H4(k) + c8*H8(k) - c12*H12(k)]
    
    k = x  # Use same grid for symmetry
    G = math.sqrt(2.0 * math.pi) * _hermite_combo(k, c0, c4, c8, -c12)
    
    # Compute L-infinity norms
    A_g = np.max(np.abs(g))
    A_G = np.max(np.abs(G))
    
    return A_g * A_G


def construct():
    """
    Return three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient is derived to impose g(0) = 0.
    Score is the upper bound on C4 = A(f)*A(F(f)), lower is better.
    """
    t_start = time.time()
    deadline = t_start + 200.0  # Safety margin before 210s kill
    
    # Start with the known best
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _upper_bound(*_build_g(*best_coeffs))
    
    # Local optimization using Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c0, c4, c8, c12 = _build_g(c0, c4, c8)
        return _upper_bound(c0, c4, c8, c12)
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    # First, try optimizing from the known best
    try:
        res = minimize(objective, best_coeffs, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_coeffs
    
    # Try random restarts in a wider range
    # The coefficients are expected to be of order 1 or smaller
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random starting point
        # Scale based on the known solution
        scale = 1.0
        x0 = rng.uniform(-2 * scale, 2 * scale, size=3)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x
        except Exception:
            pass
    
    return best_coeffs


# EVOLVE-BLOCK-END
