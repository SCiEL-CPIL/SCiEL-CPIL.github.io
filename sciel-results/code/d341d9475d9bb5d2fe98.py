"""
Uncertainty inequality: tune Hermite-Gaussian test function coefficients to minimize the upper bound on C4.

We parameterize g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
where a12 is derived from the constraint g(0) = 0.

The score is the upper bound on A(f)*A(F(f)) (or C4-related constant).
We use scipy.optimize to minimize this bound over (a0, a4, a8).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import eval_hermite
from scipy.optimize import minimize, differential_evolution

DEADLINE_FRACTION = 0.90  # safety margin before 180s (or 210s)


def _hermite_poly(n, x):
    """Compute nth (physicists') Hermite polynomial H_n(x) efficiently."""
    return eval_hermite(n, x)


# Precompute grid and Hermite polynomials for speed
_GRID_X = None
_GRID_H = None

def _init_grid():
    """Initialize the computation grid and precomputed Hermite values."""
    global _GRID_X, _GRID_H
    if _GRID_X is None:
        _GRID_X = np.linspace(-12, 12, 40000)
        _GRID_H = {
            0: np.ones_like(_GRID_X),
            4: eval_hermite(4, _GRID_X),
            8: eval_hermite(8, _GRID_X),
            12: eval_hermite(12, _GRID_X)
        }
        _GRID_H['exp'] = np.exp(-0.5 * _GRID_X**2)

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on the uncertainty product for g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12)*exp(-x^2/2)
    where a12 is chosen so g(0) = 0.
    
    We use numerical quadrature to compute the relevant integrals.
    The bound on A(f)*A(F(f)) involves:
      A(f) = ||f||_inf / ||f||_2  (or related norm ratio)
      A(F(f)) = ||F(f)||_inf / ||F(f)||_2
    
    For the Hermite-Gaussian framework, the uncertainty product bound relates to
    the L-infinity norms of g and its Fourier transform relative to L2 norms.
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n * H_n(x) * exp(-x^2/2) (up to normalization).
    So |F(g)| has the same L-infinity/L2 ratio structure as g.
    
    Let's compute:
      ||g||_2^2 = integral of g(x)^2 dx
      ||g||_inf = max |g(x)|
      ||F(g)||_2^2 = ||g||_2^2 (Parseval, with appropriate normalization)
      ||F(g)||_inf = max |F(g)(xi)|
    
    The product bound is (||g||_inf / ||g||_2) * (||F(g)||_inf / ||F(g)||_2)
    """
    _init_grid()
    
    # Derive a12 from g(0) = 0
    # H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # g(0) = (a0*1 + a4*(-3) + a8*105 + a12*(-10395)) * 1 = 0
    # a12 = (a0 - 3*a4 + 105*a8) / 10395
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    # Build g(x) and compute norms via numerical integration
    # g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    # Fourier transform: F(g)(xi) = (1/sqrt(2*pi)) * integral g(x) exp(-i*xi*x) dx
    # For Hermite functions: FT of H_n(x)*exp(-x^2/2) = (-i)^n * H_n(x) * exp(-x^2/2) (with 1/sqrt(2*pi) convention)
    # Actually, with the convention FT(f)(xi) = (1/sqrt(2*pi)) int f(x) e^{-ixi x} dx:
    #   FT[H_n(x) e^{-x^2/2}](xi) = (-i)^n H_n(xi) e^{-xi^2/2}
    
    # So F(g)(xi) = (a0*1 + a4*(-1)^2*1 + a8*(-1)^4*1 + a12*(-1)^6*1) * H_0(xi)*e^{-xi^2/2}
    # Wait, (-i)^n for even n: (-i)^0=1, (-i)^4=1, (-i)^8=1, (-i)^12=1
    # So F(g)(xi) = (a0 + a4 + a8 + a12) * e^{-xi^2/2} * [H0(xi) terms...]
    # Actually: FT[g] = sum_k a_k * (-i)^k * H_k(xi) * e^{-xi^2/2}
    # For even k: (-i)^k = 1
    # So FT[g](xi) = (a0*H0(xi) + a4*H4(xi) + a8*H8(xi) + a12*H12(xi)) * e^{-xi^2/2}
    # This is the SAME polynomial combination as g!
    
    # So ||F(g)||_inf = ||g||_inf and ||F(g)||_2 = ||g||_2
    # The product is just (||g||_inf / ||g||_2)^2
    
    # Let's compute this numerically using precomputed grid
    x = _GRID_X
    H0 = _GRID_H[0]
    H4 = _GRID_H[4]
    H8 = _GRID_H[8]
    H12 = _GRID_H[12]
    exp_term = _GRID_H['exp']
    
    poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    g = poly * exp_term
    
    # ||g||_2^2
    g2 = g**2
    norm2_sq = np.trapz(g2, x)
    
    if norm2_sq <= 0:
        return 1e10  # invalid
    
    norm2 = np.sqrt(norm2_sq)
    norm_inf = np.max(np.abs(g))
    
    bound = (norm_inf / norm2)**2
    
    return bound


def construct():
    start_time = time.time()
    deadline = 180.0 * DEADLINE_FRACTION  # 162s
    
    best_val = float('inf')
    best_x = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])  # seed
    
    # Evaluate seed
    try:
        seed_val = _compute_bound(best_x[0], best_x[1], best_x[2])
        if seed_val < best_val:
            best_val = seed_val
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x
    
    # Try a smaller, smarter grid search for initialization
    try:
        a0_range = np.linspace(-3, 3, 12)
        a4_range = np.linspace(-0.3, 0.3, 8)
        a8_range = np.linspace(-0.005, 0.005, 6)
        
        grid_best = best_val
        grid_best_x = best_x.copy()
        
        for i, a0 in enumerate(a0_range):
            for a4 in a4_range:
                for a8 in a8_range:
                    if time.time() - start_time > deadline - 25:
                        raise TimeoutError
                    try:
                        v = _compute_bound(a0, a4, a8)
                        if v < grid_best:
                            grid_best = v
                            grid_best_x = np.array([a0, a4, a8])
                    except Exception:
                        pass
        
        if grid_best < best_val:
            best_val = grid_best
            best_x = grid_best_x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return best_x
    
    # Refine with scipy.optimize using multiple methods
    try:
        # First: Nelder-Mead for local refinement
        result = minimize(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-12}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 15:
        return best_x
    
    # Second round: try BFGS with numerical gradient if we have time
    try:
        if time.time() - start_time < deadline - 10:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-10}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_x
    
    # Third round: multiple local starts from perturbed best point
    try:
        for _ in range(5):
            if time.time() - start_time > deadline - 8:
                break
            x0 = best_x + 0.005 * np.random.randn(3)
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-13}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    # Final high-precision polish pass
    try:
        if time.time() - start_time < deadline - 5:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-11, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
