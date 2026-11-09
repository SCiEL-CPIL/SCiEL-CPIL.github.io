"""Refined Cohn-Goncalves Laguerre sign-uncertainty root optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, factorial
from scipy.optimize import minimize
import time

_M = 12
_N = 1
_DEADLINE_S = 630.0
_SAFETY_S = 30.0


def _build_polynomial(roots):
    """
    Build the refined Cohn-Goncalves polynomial with double roots.
    
    The construction: P(x) = L_M^(N-1)(x) * prod_i (x - r_i)^2 / normalization
    For n=1, N-1=0, so we use L_M^(0)(x).
    
    The polynomial is: Q(x) = L_M^(0)(x) * prod_{i=1}^{M} (x - r_i)^2
    We normalize so that the L2 inner product with weight e^{-x} is 1.
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = _M
    
    # Build product of (x - r_i)^2 as a polynomial
    # Start with [1] and multiply by (x - r_i)^2 = x^2 - 2*r_i*x + r_i^2 for each root
    poly = np.array([1.0])
    for r in roots:
        # (x - r)^2 = x^2 - 2r x + r^2
        factor = np.array([r*r, -2.0*r, 1.0])  # constant, x, x^2 coefficients
        poly = np.convolve(poly, factor)
    
    # Now multiply by L_M^(0)(x) = sum_{k=0}^{M} (-1)^k * C(M,k) * x^k / k!
    # L_M^(0)(x) coefficients (from constant to x^M):
    lag_coeffs = np.zeros(m + 1)
    for k in range(m + 1):
        lag_coeffs[k] = ((-1)**k) * (1.0 / factorial(k)) * (factorial(m) / (factorial(m - k)))
    
    # Multiply polynomials: result degree = len(poly) + m - 1
    full_poly = np.convolve(poly, lag_coeffs)
    
    return full_poly


def _laguerre_l2_norm_sq(coeffs):
    """
    Compute the L2 norm squared of a polynomial with respect to weight e^{-x} on [0, inf).
    For polynomial p(x) = sum c_k x^k, integral_0^inf p(x)^2 e^{-x} dx
    = sum_{i,j} c_i c_j Gamma(i+j+1)
    """
    n = len(coeffs)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += coeffs[i] * coeffs[j] * factorial(i + j)
    return total


def _sign_changes(roots):
    """
    Count the number of sign changes of the constructed polynomial.
    For double roots, the polynomial doesn't change sign at each root,
    but the Laguerre part provides the sign structure.
    
    Actually, for the Cohn-Goncalves construction with m double roots,
    the polynomial has exactly m sign changes (from the Laguerre part)
    plus the double roots don't add sign changes.
    
    The last sign change location is what matters for the bound.
    """
    # For L_12^(0)(x), the roots are all positive and interlace.
    # The number of sign changes of L_12^(0) is 12.
    # With double roots multiplying, the sign pattern is determined by L_12^(0).
    # The last sign change of the full polynomial is the last root of L_12^(0).
    # L_12^(0) has 12 positive roots. The largest root is approximately 12 + 2*sqrt(12) + ...
    
    # Actually, the sign changes of the full product Q(x) = L_12(x) * prod(x-r_i)^2
    # are exactly the sign changes of L_12(x) since (x-r_i)^2 >= 0.
    # So there are 12 sign changes, and the last one is at the largest root of L_12^(0).
    
    # The largest root of L_12^(0)(x) can be computed numerically
    # L_12^(0)(x) = 1 - 12x + 66x^2/2 - 220x^3/6 + ... 
    # Let's find it numerically
    from numpy.polynomial import laguerre as np_lag
    # Use scipy to find roots
    # L_12^(0)(x) = sum_{k=0}^{12} (-1)^k C(12,k) x^k / k!
    coeffs = np.zeros(13)
    for k in range(13):
        coeffs[k] = ((-1)**k) * (1.0 / factorial(k)) * (factorial(12) / (factorial(12 - k)))
    # coeffs[k] is coefficient of x^k
    # Find roots
    roots_of_L = np.roots(coeffs[::-1])  # numpy expects highest degree first
    real_roots = np.sort(roots_of_L[np.abs(roots_of_L.imag) < 1e-10].real)
    
    if len(real_roots) == 0:
        return 0, 0.0
    
    last_sign_change = real_roots[-1]
    return len(real_roots), last_sign_change


def _compute_bound(roots):
    """
    Compute the upper bound on C4 given 12 double-root locations.
    
    The refined Cohn-Goncalves bound:
    C4 <= 1 / (L2_norm(Q) * sqrt(prod(r_i)))
    where Q(x) = L_M^(0)(x) * prod_{i=1}^M (x - r_i)^2
    
    More precisely, the bound involves the normalization of the test function
    and the product of the root locations.
    
    The standard form for the Fourier sign uncertainty bound with m sign changes:
    C4 <= 1 / (||f||_2 * sqrt(prod r_i))
    where f is the normalized test function.
    
    For the Laguerre construction:
    ||f||_2^2 = integral_0^inf [L_M(x) * prod(x-r_i)^2]^2 e^{-x} dx / normalization
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = _M
    
    # Verify roots are positive and distinct
    if np.any(roots <= 0) or len(roots) != m:
        return np.inf
    
    # Check distinctness
    sorted_roots = np.sort(roots)
    if np.any(np.diff(sorted_roots) < 1e-10):
        return np.inf
    
    # Build the polynomial Q(x) = L_M^(0)(x) * prod_{i=1}^M (x - r_i)^2
    full_poly = _build_polynomial(roots)
    
    # Compute L2 norm squared with weight e^{-x}
    l2_sq = _laguerre_l2_norm_sq(full_poly)
    if l2_sq <= 0:
        return np.inf
    
    # The bound: C4 <= 1 / (sqrt(l2_sq) * (prod r_i)^{1/(2M)})
    # This is the geometric mean form
    prod_roots = np.prod(roots)
    if prod_roots <= 0:
        return np.inf
    
    geo_mean = prod_roots ** (1.0 / (2.0 * m))
    bound = 1.0 / (np.sqrt(l2_sq) * geo_mean)
    
    return bound


def _objective(log_r):
    """Objective function in log-space."""
    roots = np.exp(log_r)
    return _compute_bound(roots)


def construct():
    """Construct 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + _DEADLINE_S - _SAFETY_S
    
    # Initial guess: spread out in log-space
    # Use a reasonable starting point
    x0 = np.exp(np.linspace(np.log(1.0), np.log(50.0), _M))
    
    best_x = x0.copy()
    best_bound = _compute_bound(x0)
    
    # Log-space parameterization to ensure positivity
    log_roots = np.log(x0)
    
    def obj_with_deadline(log_r):
        if time.time() > deadline:
            return np.inf
        return _objective(log_r)
    
    # Primary optimization with L-BFGS-B
    try:
        result = minimize(
            obj_with_deadline,
            log_roots,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if result.fun < best_bound:
            candidate = np.exp(result.x)
            b = _compute_bound(candidate)
            if b < best_bound:
                best_bound = b
                best_x = candidate
    except Exception:
        pass
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    n_restarts = 80
    
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if i % 3 == 0:
            # Random in log-space around initial
            log_r = log_roots + rng.normal(0, 1.0, size=_M)
        elif i % 3 == 1:
            # Geometric progression with random ratio
            base = rng.uniform(0.5, 10.0)
            ratio = rng.uniform(1.1, 2.0)
            log_r = np.log(base * ratio ** np.arange(_M))
        else:
            # Clustered around different centers
            center = rng.uniform(1.0, 30.0)
            spread = rng.uniform(0.5, 3.0)
            log_r = np.log(center) + rng.normal(0, spread, size=_M)
        
        # Ensure positivity (log-space guarantees this)
        log_r = np.maximum(log_r, -10.0)  # Avoid underflow
        log_r = np.minimum(log_r, 10.0)   # Avoid overflow
        
        try:
            result = minimize(
                obj_with_deadline,
                log_r,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if result.fun < best_bound:
                candidate = np.exp(result.x)
                b = _compute_bound(candidate)
                if b < best_bound:
                    best_bound = b
                    best_x = candidate
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            log_best = np.log(best_x)
            result = minimize(
                obj_with_deadline,
                log_best,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if result.fun < best_bound:
                candidate = np.exp(result.x)
                b = _compute_bound(candidate)
                if b < best_bound:
                    best_bound = b
                    best_x = candidate
        except Exception:
            pass
    
    # Verify final result
    final_bound = _compute_bound(best_x)
    if not np.isfinite(final_bound) or final_bound >= best_bound:
        # Fallback to a known-good configuration
        best_x = np.exp(np.linspace(np.log(2.0), np.log(40.0), _M))
    
    return list(best_x)
# EVOLVE-BLOCK-END
