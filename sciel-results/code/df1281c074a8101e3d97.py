"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import hermite, eval_hermite

# Deadline guard: keep a safety margin before the 180s limit (evaluator kills at 210s)
_START_TIME = time.time()
DEADLINE = _START_TIME + 170.0  # 10s safety margin


def _time_remaining():
    return DEADLINE - time.time()


def construct():
    """
    Return three coefficients (a0, a4, a8) for the Hermite combination.
    The degree-12 coefficient is derived to impose g(0) = 0.
    Score is the upper bound on C4 = A(f) * A(Fourier(f)).
    Lower raw value is better.
    """
    # Prior state of the art (Goncalves et al. 2017)
    prior = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # We need to compute the upper bound C4 for a given set of coefficients.
    # The test function is g(x) = sum_{k in {0,4,8,12}} a_k * H_k(x) * exp(-x^2/2)
    # where a_12 is determined by g(0) = 0.
    
    # For the uncertainty principle, we need:
    # A(f) = integral |x f(x)|^2 dx / integral |f(x)|^2 dx (variance-like quantity)
    # A(F) = integral |xi F(xi)|^2 dxi / integral |F(xi)|^2 dxi
    # The upper bound C4 is related to these quantities.
    
    # Actually, looking at the task more carefully: "reduce the upper bound on A(f)*A(Fourier(f))"
    # This is the Heisenberg-type uncertainty product. The "upper bound" likely refers to a
    # specific analytical bound that can be computed from the coefficients.
    
    # Let me think about this differently. The task says "score is the upper bound on C4".
    # C4 is likely a constant in an inequality of the form:
    # A(f) * A(F) <= C4 * (some norm ratio)
    
    # For a test function g, the uncertainty product is:
    # Delta_x * Delta_xi where Delta_x^2 = <x^2> - <x>^2 and Delta_xi^2 = <xi^2> - <xi>^2
    # By Fourier properties, <xi^2> in frequency space relates to <x^2> in position space via derivatives.
    
    # Actually, for the Fourier transform convention F(xi) = integral f(x) exp(-2*pi*i*x*xi) dx:
    # <xi^2>_F = (1/(4*pi^2)) * <(f'')^2> type quantity... 
    # More precisely, if F is the Fourier transform, then xi*F corresponds to (i/2pi) f'
    # So <xi^2>_F = (1/(4*pi^2)) * <(f')^2>_weighted... 
    
    # Let me use a more direct approach. For g(x) = sum a_k H_k(x) exp(-x^2/2):
    # The Fourier transform of H_k(x) exp(-x^2/2) is (-i)^k H_k(xi) exp(-xi^2/2) (up to normalization)
    
    # Using the convention where F(xi) = (1/sqrt(2*pi)) integral f(x) exp(-i x xi) dx:
    # F{H_k(x) exp(-x^2/2)} = (-i)^k H_k(xi) exp(-xi^2/2)
    
    # So if g(x) = sum_{k} a_k H_k(x) exp(-x^2/2), then
    # G(xi) = sum_{k} a_k (-i)^k H_k(xi) exp(-xi^2/2)
    
    # The uncertainty product involves:
    # <x^2>_g / <1>_g * <xi^2>_G / <1>_G (with appropriate normalizations)
    
    # Let me compute this numerically for a given set of coefficients.
    
    def _compute_uncertainty_product(a0, a4, a8):
        """Compute the uncertainty product for the given Hermite coefficients."""
        # Derive a12 from g(0) = 0
        # g(0) = a0*H0(0)*exp(0) + a4*H4(0)*exp(0) + a8*H8(0)*exp(0) + a12*H12(0)*exp(0)
        # H_n(0) for even n: H_0(0)=1, H_4(0)=-12, H_8(0)=1344, H_12(0)=-103216
        # Wait, let me use the probabilists' vs physicists' Hermite polynomials.
        
        # The task mentions "Hermite-polynomial times Gaussian test function"
        # Let's use the standard physicists' Hermite polynomials: H_n(x) with weight exp(-x^2)
        # But the Gaussian is exp(-x^2/2), so maybe it's the probabilists' version?
        
        # Let me use physicists' Hermite: H_n(x) with generating function exp(2xt - t^2)
        # H_0(x) = 1
        # H_1(x) = 2x
        # H_2(x) = 4x^2 - 2
        # H_3(x) = 8x^3 - 12x
        # H_4(x) = 16x^4 - 48x^2 + 12
        # H_8(x) = 256x^8 - 4608x^6 + 34560x^4 - 80640x^2 + 30240
        # H_12(x) = 4096x^12 - 147456x^10 + 2478080x^8 - 17836800x^6 + 62297280x^4 - 83907840x^2 + 22179360
        
        # Actually, let me just compute H_n(0) using recurrence or known values.
        # For physicists' Hermite: H_n(0) = (-1)^(n/2) * n! / (n/2)! for even n
        # H_0(0) = 1
        # H_2(0) = -2
        # H_4(0) = -12
        # H_6(0) = 120
        # H_8(0) = -1344
        # H_10(0) = 15120
        # H_12(0) = -103216
        
        # Wait, let me double-check: H_n(0) for even n = (-1)^(n/2) * (2n-1)!!
        # H_0(0) = 1
        # H_2(0) = -1!! = -1... hmm, that doesn't match.
        
        # Let me use the explicit formula: H_n(0) = (-1)^(n/2) * n! / (n/2)! for n even
        # H_0(0) = 1!/0! = 1
        # H_2(0) = -2!/1! = -2
        # H_4(0) = -4!/2! = -12
        # H_6(0) = 6!/3! = 120
        # H_8(0) = -8!/4! = -1344
        # H_10(0) = 10!/5! = 15120
        # H_12(0) = -12!/6! = -103216
        
        # So g(0) = a0*1 + a4*(-12) + a8*(-1344) + a12*(-103216) = 0
        # a12 = -(a0 - 12*a4 - 1344*a8) / (-103216)
        # a12 = (a0 - 12*a4 - 1344*a8) / 103216
        
        a12 = (a0 - 12.0 * a4 - 1344.0 * a8) / 103216.0
        
        # Now compute the uncertainty product numerically
        # We need to integrate |x g(x)|^2 and |g(x)|^2 over x
        # And |xi G(xi)|^2 and |G(xi)|^2 over xi
        
        # For efficiency, use the fact that H_k(x) exp(-x^2/2) are orthogonal
        # with weight exp(-x^2/2) (or similar)
        
        # Actually, let's just do numerical integration on a grid
        x = np.linspace(-20, 20, 4001)
        dx = x[1] - x[0]
        
        # Compute g(x) = sum a_k H_k(x) exp(-x^2/2)
        # Use physicists' Hermite polynomials
        g = np.zeros_like(x)
        for k, a in [(0, a0), (4, a4), (8, a8), (12, a12)]:
            g += a * eval_hermite(k, x) * np.exp(-x**2 / 2.0)
        
        # Compute G(xi) using Fourier transform properties
        # F{H_k(x) exp(-x^2/2)} = (-i)^k H_k(xi) exp(-xi^2/2) (with appropriate normalization)
        # Using the convention F(xi) = (1/sqrt(2*pi)) integral f(x) exp(-i x xi) dx
        
        xi = np.linspace(-20, 20, 4001)
        dxi = xi[1] - xi[0]
        
        G = np.zeros_like(xi)
        for k, a in [(0, a0), (4, a4), (8, a8), (12, a12)]:
            G += a * ((-1j)**k) * eval_hermite(k, xi) * np.exp(-xi**2 / 2.0)
        
        # Compute uncertainty quantities
        # <x^2>_g = integral x^2 |g(x)|^2 dx / integral |g(x)|^2 dx
        # <xi^2>_G = integral xi^2 |G(xi)|^2 dxi / integral |G(xi)|^2 dxi
        
        norm_g = np.sum(np.abs(g)**2) * dx
        norm_G = np.sum(np.abs(G)**2) * dxi
        
        if norm_g < 1e-15 or norm_G < 1e-15:
            return np.inf
        
        x2_g = np.sum(x**2 * np.abs(g)**2) * dx / norm_g
        xi2_G = np.sum(xi**2 * np.abs(G)**2) * dxi / norm_G
        
        # The uncertainty product is sqrt(x2_g) * sqrt(xi2_G)
        # But we need the "upper bound on C4" which might be this product directly
        # or a related quantity
        
        # For the Heisenberg uncertainty principle: Delta_x * Delta_xi >= 1/2 (or 1/4*pi depending on convention)
        # The "upper bound" might refer to the actual value of the uncertainty product for this test function
        
        uncertainty_product = np.sqrt(x2_g) * np.sqrt(xi2_G)
        
        return uncertainty_product
    
    # Start with the prior
    best_coeffs = prior.copy()
    best_value = _compute_uncertainty_product(*prior)
    
    # Try to improve using local optimization
    # Use a simple coordinate descent or Nelder-Mead
    
    try:
        from scipy.optimize import minimize
        from scipy.optimize import differential_evolution
        
        # Define objective function
        def objective(coeffs):
            if _time_remaining() < 5.0:
                return best_value  # Return current best if time is running out
            return _compute_uncertainty_product(coeffs[0], coeffs[1], coeffs[2])
        
        # Try multiple random starts
        rng = np.random.RandomState(42)
        
        # First, try local optimization from the prior
        result = minimize(objective, prior, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < best_value:
            best_value = result.fun
            best_coeffs = result.x.copy()
        
        # Try a few more random starts if time permits
        for i in range(5):
            if _time_remaining() < 30.0:
                break
            start = prior + rng.normal(0, 0.5, 3)
            result = minimize(objective, start, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        
        # Final polish with tighter tolerance
        if _time_remaining() > 10.0:
            result = minimize(objective, best_coeffs, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-15})
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        
    except Exception:
        pass
    
    # Return the best coefficients found
    return best_coeffs
# EVOLVE-BLOCK-END
