"""Improved program: Optimize roots for Laguerre double-root construction."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize

M = 12          # number of double roots
N = 1           # Laguerre order (n=1)
K = 2 * M + 1   # number of basis Laguerres (degree up to 2M)

def _expand_P_in_monomials(roots):
    """Expand P(x) = prod (x-r_i)^2 in monomial basis, return coeffs ascending."""
    poly_coeffs = np.array([1.0])
    for r in roots:
        # (x - r)^2 = r^2 - 2r*x + x^2, ascending: [r^2, -2r, 1]
        sq = np.array([r * r, -2.0 * r, 1.0])
        poly_coeffs = np.convolve(poly_coeffs, sq)
    return poly_coeffs

def _laguerre_monomial_coeffs(k, alpha):
    """
    Return coefficients of L_k^{(alpha)}(x) in ascending monomial order.
    L_k^{(alpha)}(x) = sum_{j=0}^k (-1)^j * C(k+alpha, k-j) * x^j / j!
    """
    coeffs = np.zeros(k + 1)
    for j in range(k + 1):
        log_binom = gammaln(k + alpha + 1) - gammaln(k - j + 1) - gammaln(j + alpha + 1)
        binom_val = np.exp(log_binom)
        coeff = ((-1) ** j) * binom_val / np.exp(gammaln(j + 1))
        coeffs[j] = coeff
    return coeffs

def _get_laguerre_coefficients(roots):
    """
    Exact computation of Laguerre coefficients using moment integration.
    c_k = 1/h_k * integral_0^inf x^N e^{-x} P(x) L_k^{(N)}(x) dx
    where h_k = Gamma(k+N+1)/k!
    """
    p_coeffs = _expand_P_in_monomials(roots)
    
    coeffs = np.zeros(K)
    for k in range(K):
        l_coeffs = _laguerre_monomial_coeffs(k, N)
        # Product P(x) * L_k^{(N)}(x) in monomial basis
        prod_coeffs = np.convolve(p_coeffs, l_coeffs)
        
        # Multiply by x^N: shift by N positions
        integrand_coeffs = np.zeros(len(prod_coeffs) + N)
        for j in range(len(prod_coeffs)):
            integrand_coeffs[j + N] = prod_coeffs[j]
        
        # Integrate term by term: ∫ x^j x^N e^{-x} dx = Gamma(j+N+1)
        integral = 0.0
        for j in range(len(integrand_coeffs)):
            if integrand_coeffs[j] != 0.0:
                integral += integrand_coeffs[j] * np.exp(gammaln(j + N + 1))
        
        log_h_k = gammaln(k + N + 1) - gammaln(k + 1)
        h_k = np.exp(log_h_k)
        coeffs[k] = integral / h_k
        
    return coeffs

def _validate_roots(roots):
    """Check if roots are valid and coefficients are non-negative."""
    if len(roots) != M:
        return False
    if np.any(roots <= 0):
        return False
    if np.any(np.diff(roots) <= 1e-6):
        return False
    
    coeffs = _get_laguerre_coefficients(roots)
    if coeffs is None:
        return False
    
    if np.any(coeffs < -1e-8):
        return False
        
    return True

def construct():
    # Initial guess: roots of L_M^{(N)}(x)
    try:
        roots0, _ = roots_genlaguerre(M, N)
        roots0 = np.sort(roots0)
    except Exception:
        roots0 = np.array([i * 4.0 for i in range(1, M + 1)])

    def objective(params):
        roots = np.sort(params)
        if np.any(roots <= 0):
            return 1e10
        if np.any(np.diff(roots) <= 1e-6):
            return 1e10
            
        coeffs = _get_laguerre_coefficients(roots)
        if coeffs is None:
            return 1e10
            
        neg_min = np.min(coeffs)
        if neg_min < 0:
            return 1e6 * (-neg_min) + np.max(roots)
            
        return np.max(roots)

    best_roots = roots0.copy()
    best_score = objective(roots0)
    
    # Try multiple initial guesses with different scalings
    np.random.seed(42)
    for trial in range(20):
        scale = 0.3 + 0.15 * trial
        perturbed = roots0 * scale * (1 + 0.05 * np.random.randn(M))
        perturbed = np.sort(perturbed)
        if np.any(perturbed <= 0):
            continue
        score = objective(perturbed)
        if score < best_score:
            best_score = score
            best_roots = perturbed.copy()

    # Refine with Nelder-Mead
    try:
        res = minimize(objective, best_roots, method='Nelder-Mead', 
                       options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        if res.fun < best_score:
            best_roots = res.x.copy()
            best_score = res.fun
    except Exception:
        pass

    # Final validation
    if not _validate_roots(best_roots):
        if _validate_roots(roots0):
            best_roots = roots0
        else:
            best_roots = np.array([i * 4.0 for i in range(1, M + 1)])

    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
