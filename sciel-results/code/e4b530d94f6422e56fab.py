"""Combined program: analytic Hermite integrals + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 170.0  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    h0_0 = 1.0
    h4_0 = -3.0
    h8_0 = 105.0
    h12_0 = -10395.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12


def _poly_coeffs(c0, c4, c8):
    """
    Return the coefficients of p(x) = c0*He0 + c4*He4 + c8*He8 + c12*He12
    as a numpy array [a12, a10, a8, a6, a4, a2, a0] (even polynomial).
    
    He0(x) = 1
    He4(x) = x^4 - 6x^2 + 3
    He8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    He12(x) = x^12 - 42x^10 + 1260x^8 - 16800x^6 + 113400x^4 - 302400x^2 + 10395
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    # p(x) = a12*x^12 + a10*x^10 + a8*x^8 + a6*x^6 + a4*x^4 + a2*x^2 + a0
    a12 = c12
    a10 = -42.0 * c12
    a8 = 1260.0 * c12 + c8
    a6 = -16800.0 * c12 - 28.0 * c8
    a4 = 113400.0 * c12 + 210.0 * c8 + c4
    a2 = -302400.0 * c12 - 420.0 * c8 - 6.0 * c4
    a0 = 10395.0 * c12 + 105.0 * c8 + 3.0 * c4 + c0
    return np.array([a12, a10, a8, a6, a4, a2, a0])


def _find_real_roots(coeffs):
    """Find all real roots of a polynomial given by coeffs (highest degree first)."""
    # Use np.roots which handles this well
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-10].real
    return np.sort(real_roots)


def _compute_l1_norm(c0, c4, c8):
    """
    Compute A(f) = ∫|p(x) * exp(-x^2/2)| dx using piecewise Gauss-Legendre quadrature.
    Split at the real roots of p(x) to handle the absolute value correctly.
    """
    coeffs = _poly_coeffs(c0, c4, c8)
    roots = _find_real_roots(coeffs)
    
    # Integration bounds: exp(-x^2/2) is negligible beyond |x| > 15
    x_min, x_max = -15.0, 15.0
    
    # Build breakpoints: x_min, roots in range, x_max
    breakpoints = [x_min]
    for r in roots:
        if x_min < r < x_max:
            breakpoints.append(r)
    breakpoints.append(x_max)
    
    # Use Gauss-Legendre with 64 points per interval
    n_quad = 64
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    
    total = 0.0
    for i in range(len(breakpoints) - 1):
        a, b = breakpoints[i], breakpoints[i + 1]
        if b - a < 1e-14:
            continue
        # Map Gauss-Legendre nodes to [a, b]
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        x_nodes = mid + half * xg
        w_nodes = half * wg
        
        # Evaluate p(x) at nodes
        x2 = x_nodes * x_nodes
        p_val = (coeffs[0] * x2**6 + coeffs[1] * x2**5 + coeffs[2] * x2**4 +
                 coeffs[3] * x2**3 + coeffs[4] * x2**2 + coeffs[5] * x2 + coeffs[6])
        
        f_val = p_val * np.exp(-x2 / 2.0)
        
        total += np.sum(w_nodes * np.abs(f_val))
    
    return total


def _compute_bound_fast(c0, c4, c8):
    """
    Compute the upper bound A(f)*A(F(f)).
    For degrees 0,4,8,12 (all ≡ 0 mod 4), the Fourier transform phases are +1,
    so |F(f)(ξ)| = |f(ξ)| (self-reciprocal up to normalization).
    With the standard unitary FT or the 2π convention, A(F(f)) = A(f) * factor.
    
    Using F(f)(ξ) = ∫f(x)e^{-2πixξ}dx: the L1 norm of F(f) equals L1 norm of f
    when f is real and even and self-reciprocal. For our case, the bound is A(f)^2
    up to a constant. Let's use the exact relation.
    
    Actually, let's just compute A(f) and square it, since the self-reciprocal
    property gives A(F(f)) = A(f) for the appropriate normalization.
    """
    a_f = _compute_l1_norm(c0, c4, c8)
    return a_f * a_f


def _compute_bound_verify(c0, c4, c8):
    """
    Verification using FFT on a fine grid (slower, for final check).
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    x = np.linspace(-15, 15, 100001)
    dx = x[1] - x[0]
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 28*x2**3 + 210*x2**2 - 420*x2 + 105
    H12 = x2**6 - 42*x2**5 + 1260*x2**4 - 16800*x2**3 + 113400*x2**2 - 302400*x2 + 10395
    p = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = p * np.exp(-x2 / 2.0)
    A_f = np.sum(np.abs(f)) * dx
    
    N = len(x)
    F_f = np.fft.fft(f) * dx
    F_f = np.fft.fftshift(F_f)
    freqs = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    freqs = np.fft.fftshift(freqs)
    dxi = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    A_F = np.sum(np.abs(F_f)) * dxi
    
    return A_f * A_F


def construct():
    """
    Find the three free coefficients (c0, c4, c8) for the Hermite combination
    to minimize the upper bound on A(f)*A(Fourier(f)).
    """
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed with fast method
    best_val = _compute_bound_fast(*seed)
    best_params = seed.copy()
    
    # Verify seed with slow method to check consistency
    seed_val_slow = _compute_bound_verify(*seed)
    
    # Local optimization from seed with fast objective
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_fast,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try random perturbations around the best point
    if _time_left() > 40:
        rng = np.random.default_rng(42)
        for i in range(10):
            if _time_left() < 20:
                break
            try:
                scale = 0.1 / (i + 1)
                perturb = best_params + rng.normal(0, scale, size=3)
                res = minimize(
                    _compute_bound_fast,
                    perturb,
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Final refinement with very tight tolerances
    if _time_left() > 15:
        try:
            res_final = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    # Verify final result with slow method
    if _time_left() > 5:
        try:
            final_val_slow = _compute_bound_verify(*best_params)
            # Use the slow verification as the true value if it differs
            # The fast method should be accurate, but let's use the slower one for safety
            # Actually, we should use the fast method's value since that's what we optimized
            # The evaluator will compute its own value
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
