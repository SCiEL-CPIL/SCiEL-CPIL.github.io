"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.special import eval_genlaguerre, factorial

# EVOLVE-BLOCK-START
import itertools
import sys

def _laguerre_basis(m, n):
    """Return Laguerre polynomials L_k^{(n)} for k=0..m."""
    return [eval_genlaguerre(k, n, 1.0) for k in range(m + 1)]

def _poly_coeffs(lag_basis, coeffs):
    """
    Given Laguerre basis polynomials (as numpy poly1d or coeff arrays) and
    combination coefficients, return the coefficient array of the combined polynomial.
    We represent Laguerre polynomials as coefficient arrays (highest power first).
    """
    # We'll use numpy's poly operations
    result = np.array([0.0])
    for c, L in zip(coeffs, lag_basis):
        # L is a numpy array of coefficients (highest degree first)
        # Multiply by c and add
        result = np.polyadd(result, c * L)
    return result

def _find_roots(coeffs):
    """Find roots of polynomial given by coefficients (highest power first)."""
    if len(coeffs) == 0:
        return np.array([])
    # Remove leading zeros
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
        coeffs = coeffs[1:]
    if len(coeffs) == 1:
        return np.array([])
    roots = np.roots(coeffs)
    return roots

def _is_valid_roots(roots, m=12, tol=1e-8):
    """Check if roots form m positive double roots."""
    if len(roots) != 2 * m:
        return False
    # Sort by real part
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    if len(real_roots) != 2 * m:
        return False
    real_roots = np.sort(real_roots)
    # Check all positive
    if np.any(real_roots <= 0):
        return False
    # Check for double roots: pair them up
    for i in range(0, 2 * m, 2):
        if abs(real_roots[i] - real_roots[i + 1]) > tol:
            return False
    return True

def _extract_double_roots(roots, m=12):
    """Extract the m double root locations from 2m roots."""
    real_roots = roots[np.abs(roots.imag) < 1e-6].real
    if len(real_roots) != 2 * m:
        return None
    real_roots = np.sort(real_roots)
    double_roots = []
    for i in range(0, 2 * m, 2):
        double_roots.append((real_roots[i] + real_roots[i + 1]) / 2.0)
    return double_roots

def _compute_score(double_roots, m=12, n=1):
    """
    Compute the upper bound on C4 given the double root locations.
    The score is related to the Fourier sign uncertainty upper bound.
    
    For the Cohn-Goncalves construction, the upper bound involves
    evaluating the Laguerre combination at specific points.
    
    A common form: the bound is proportional to 1/(product of root spacings)
    or involves the value of the polynomial at 0 or infinity.
    
    For m=12, n=1, the refined bound C4_upper is typically:
    C4 = (some constant) * f(roots)
    
    Let's use a standard formulation: the upper bound on the number of sign changes
    or the uncertainty constant involves the reciprocal of the product of
    differences between consecutive roots, or the value of the derivative.
    
    For a polynomial P(x) = prod (x - r_i)^2, the relevant quantity for
    uncertainty bounds is often related to ||P'||_2 / ||P||_2 or similar.
    
    Let's use: score = 1 / prod_{i<j} |r_i - r_j|^(1/m) or similar.
    
    Actually, for the Cohn-Goncalves Laguerre construction, the upper bound
    on C4 (the 4th moment or sign change count) is typically:
    
    C4_upper = (2m)! / (prod of certain factors)
    
    Let me use a simpler proxy that is monotonic with the true bound:
    The upper bound is inversely related to the product of pairwise distances.
    
    For 12 double roots, we have 12 distinct locations. The score (upper bound)
    should be minimized by choosing roots that maximize the product of distances.
    
    score = log(prod_{i<j} |r_i - r_j|)  -> we want to MAXIMIZE this
    But the task says "score is the upper bound on C4" and "lower is better".
    
    So score = 1 / prod_{i<j} |r_i - r_j| or similar.
    
    Let's use: score = exp(-log_prod_dist) where log_prod_dist = sum_{i<j} log|r_i - r_j|
    """
    if double_roots is None or len(double_roots) != m:
        return float('inf')
    
    r = np.array(double_roots)
    if np.any(r <= 0):
        return float('inf')
    
    # Compute log of product of pairwise distances
    log_prod = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            d = abs(r[i] - r[j])
            if d < 1e-15:
                return float('inf')
            log_prod += np.log(d)
    
    # The upper bound is inversely related to this product
    # Use a standard normalization
    score = np.exp(-log_prod)
    
    return score

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive double-root locations.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    m = 12
    n = 1
    
    # Precompute Laguerre basis
    # We need L_k^{(1)} for k = 0, 1, ..., 12
    # eval_genlaguerre(k, alpha, x) gives the value, but we need the polynomial coefficients
    # Let's construct them manually or use scipy's poly operations
    
    # Laguerre polynomials L_k^{(alpha)}(x) can be constructed
    # L_0^{(alpha)}(x) = 1
    # L_1^{(alpha)}(x) = -x + alpha + 1
    # Recurrence: (k+1) L_{k+1}^{(alpha)}(x) = (2k + alpha + 1 - x) L_k^{(alpha)}(x) - (k + alpha) L_{k-1}^{(alpha)}(x)
    
    def laguerre_coeffs(k, alpha):
        """Return coefficients of L_k^{(alpha)}(x), highest power first."""
        if k == 0:
            return np.array([1.0])
        if k == 1:
            return np.array([-1.0, alpha + 1.0])
        
        L_km1 = laguerre_coeffs(k - 1, alpha)
        L_km2 = laguerre_coeffs(k - 2, alpha)
        
        # (k) L_k = (2(k-1) + alpha + 1 - x) L_{k-1} - ((k-1) + alpha) L_{k-2}
        # L_k = [(2k - 2 + alpha + 1 - x) L_{k-1} - (k - 1 + alpha) L_{k-2}] / k
        
        # (2k - 1 + alpha - x) L_{k-1}
        # L_{k-1} has degree k-1, so (2k-1+alpha - x) L_{k-1} has degree k
        term1 = np.polyadd(
            np.poly1d([2 * k - 1 + alpha]) * np.poly1d(L_km1),
            -np.poly1d(L_km1) * np.poly1d([1.0, 0.0])
        )
        
        # Wait, let me be more careful.
        # (2k - 1 + alpha) * L_{k-1}(x) - x * L_{k-1}(x)
        
        c1 = 2 * k - 1 + alpha
        # c1 * L_{k-1}
        part1 = c1 * L_km1
        # -x * L_{k-1}: shift coefficients
        part2 = np.concatenate([0.0, -L_km1])
        
        term1 = np.polyadd(part1, part2)
        
        # (k - 1 + alpha) * L_{k-2}
        c2 = k - 1 + alpha
        term2 = c2 * L_km2
        
        # term1 - term2
        result = np.polyadd(term1, -term2)
        
        # Divide by k
        result = result / k
        
        return result
    
    # Build Laguerre basis for n=1
    lag_basis = []
    for k in range(m + 1):
        lag_basis.append(laguerre_coeffs(k, n))
    
    # Now we need to find coefficients c_0, ..., c_12 such that
    # P(x) = sum_{k=0}^{12} c_k L_k^{(1)}(x) has 12 positive double roots.
    
    # A double root at r means P(r) = 0 and P'(r) = 0.
    # So we have 24 conditions (12 roots * 2 conditions) but only 13 coefficients.
    # This is over-determined. The Cohn-Goncalves construction uses a specific
    # ansatz where the polynomial is a product of Laguerre-related factors.
    
    # Alternative approach: The Cohn-Goncalves polynomial for m=12, n=1 is
    # P(x) = [L_m^{(n)}(x)]^2 + c * x * [L_{m-1}^{(n+1)}(x)]^2 or similar.
    
    # Actually, the standard construction is:
    # P(x) = [L_m^{(n)}(x)]^2 + (m+1) * x * [L_{m-1}^{(n+1)}(x)]^2
    # or
    # P(x) = [L_m^{(n)}(x)]^2 - c * x * [L_{m-1}^{(n+1)}(x)]^2
    
    # Let's try the standard Cohn-Goncalves form.
    # For the uncertainty principle, the relevant polynomial is:
    # P(x) = [L_m^{(n)}(x)]^2 + x * [L_{m-1}^{(n+1)}(x)]^2
    
    # Let's construct this and find its roots.
    
    L_m = laguerre_coeffs(m, n)
    L_m_1 = laguerre_coeffs(m - 1, n + 1)
    
    # P(x) = L_m(x)^2 + x * L_{m-1}^{(n+1)}(x)^2
    L_m_sq = np.polysqu(L_m)
    L_m_1_sq = np.polysqu(L_m_1)
    x_term = np.polymul(np.array([1.0, 0.0]), L_m_1_sq)
    
    P_coeffs = np.polyadd(L_m_sq, x_term)
    
    roots = _find_roots(P_coeffs)
    
    double_roots = _extract_double_roots(roots, m)
    
    if double_roots is not None:
        return list(double_roots)
    
    # If the standard construction doesn't give clean double roots, try variations
    # or fall back to a heuristic.
    
    # Try different combinations
    for sign in [1.0, -1.0]:
        P_coeffs = np.polyadd(L_m_sq, sign * x_term)
        roots = _find_roots(P_coeffs)
        double_roots = _extract_double_roots(roots, m)
        if double_roots is not None:
            return list(double_roots)
    
    # Fallback: use the seed program's output
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
