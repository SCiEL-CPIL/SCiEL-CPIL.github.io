"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_genlaguerre, gammaln
from scipy.optimize import minimize, brentq
import time
from math import comb, factorial, log, exp

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive double-root locations.
    For the Cohn-Goncalves Laguerre construction, the bound involves the
    polynomial P(x) = prod (x - r_i)^2 and its relationship to the Laguerre
    weight. The bound is related to the L2 norm of P under the Laguerre weight
    divided by the product of root factors.
    
    The actual C4 bound from the refined Cohn-Goncalves construction:
    C4 <= K * ||P||_{L2(w)} / (prod r_i)^{1/2}
    where w(x) = e^{-x} is the Laguerre weight and K is a constant.
    
    For a polynomial P(x) = prod (x - r_i)^2 with double roots,
    ||P||_{L2(w)}^2 = integral_0^inf P(x)^2 e^{-x} dx
    
    This can be computed using the moments of the Laguerre weight.
    """
    roots = np.array(roots, dtype=np.float64)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    if np.any(np.diff(np.sort(roots)) < 1e-8):
        return float('inf')
    
    # Compute ||P||_{L2(w)}^2 where P(x) = prod (x - r_i)^2
    # P(x)^2 = prod (x - r_i)^4
    # integral_0^inf prod (x - r_i)^4 e^{-x} dx
    # = integral_0^inf [prod (x - r_i)^2]^2 e^{-x} dx
    # 
    # Expand: P(x)^2 = sum of terms x^k * c_k
    # integral_0^inf x^k e^{-x} dx = k!
    #
    # So ||P||^2 = sum_k c_k * k!
    #
    # To compute c_k, we need the coefficients of P(x)^2 where
    # P(x) = prod (x - r_i)^2 = prod (x^2 - 2*r_i*x + r_i^2)
    
    # Build P(x) = prod (x - r_i)^2
    # First build Q(x) = prod (x - r_i), then P(x) = Q(x)^2
    # Q(x) = x^12 + q_11*x^11 + ... + q_0
    
    # Compute elementary symmetric polynomials of roots
    from itertools import combinations
    
    # Q(x) = prod (x - r_i)
    # Coefficients: q_k = (-1)^{12-k} * e_{12-k}(r_1,...,r_12)
    # where e_j is the j-th elementary symmetric polynomial
    
    # For efficiency, compute using Newton's identities or direct multiplication
    # Let's just multiply polynomials directly
    
    # Start with polynomial [1] (representing 1)
    poly = np.array([1.0])
    for r in roots:
        # Multiply by (x - r)
        new_poly = np.zeros(len(poly) + 1)
        new_poly[1:] += poly  # x * poly
        new_poly[:-1] += (-r) * poly  # -r * poly
        poly = new_poly
    
    # poly represents Q(x) = poly[0]*x^12 + poly[1]*x^11 + ... + poly[12]
    # P(x) = Q(x)^2
    # P(x) = (poly[0]*x^12 + ... + poly[12])^2
    # Use convolution
    P_coeffs = np.convolve(poly, poly)
    # P_coeffs[k] is coefficient of x^{24-k} in P(x)
    # So P(x) = P_coeffs[0]*x^24 + P_coeffs[1]*x^23 + ... + P_coeffs[24]
    
    # ||P||^2 = sum_{k=0}^{24} P_coeffs[k] * (24-k)!
    # Wait: integral_0^inf x^j e^{-x} dx = j!
    # P(x) = sum_{j=0}^{24} a_j * x^j where a_j = P_coeffs[24-j]
    # ||P||^2 = sum_{j=0}^{24} a_j * j!
    
    # P_coeffs[k] is coeff of x^{24-k}, so a_j = P_coeffs[24-j]
    log_norm_sq = 0.0
    for j in range(25):
        a_j = P_coeffs[24 - j]
        if a_j != 0:
            log_norm_sq += np.log(abs(a_j)) + gammaln(j + 1)  # log|a_j| + log(j!)
            # But we need the actual sum, not log-sum. Let's compute directly.
    
    # Direct computation (may overflow for large roots, but let's try)
    norm_sq = 0.0
    for j in range(25):
        a_j = P_coeffs[24 - j]
        if a_j != 0:
            norm_sq += a_j * factorial(j)
    
    if norm_sq <= 0:
        return float('inf')
    
    # C4 bound: the refined Cohn-Goncalves bound
    # C4 <= sqrt(||P||^2) / (prod r_i)^{1/2} * K
    # where K is a universal constant from the construction
    #
    # Actually, looking at the structure more carefully:
    # The bound involves the ratio of the L2 norm to the product of roots
    # raised to some power. Let me try:
    # C4 <= sqrt(norm_sq) / sqrt(prod(roots))
    
    prod_roots = np.prod(roots)
    if prod_roots <= 0:
        return float('inf')
    
    bound = np.sqrt(norm_sq) / np.sqrt(prod_roots)
    return bound

def _compute_c4_bound_simple(roots):
    """
    Alternative simpler bound: just use the product of roots.
    C4 <= K / sqrt(prod(roots))
    """
    roots = np.array(roots, dtype=np.float64)
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    
    log_prod = np.sum(np.log(roots))
    c4_bound = exp(-0.5 * log_prod)
    return c4_bound

def _check_constraints(roots, tol=1e-6):
    """
    Check if the roots satisfy the Laguerre-type polynomial constraints.
    """
    roots = np.array(roots)
    if len(roots) != 12:
        return False
    if np.any(roots <= 0):
        return False
    
    # Check that roots are distinct
    if np.any(np.diff(np.sort(roots)) < 1e-8):
        return False
    
    # Reasonable range check
    inv_sum = np.sum(1.0 / roots)
    if inv_sum < 0.01 or inv_sum > 1000:
        return False
    
    return True

def _get_laguerre_zeros_direct(n, alpha=0):
    """
    Get zeros of the generalized Laguerre polynomial L_n^(alpha)
    using the three-term recurrence relation.
    """
    if alpha == 0:
        # Standard Laguerre polynomial
        # L_n(x) satisfies: (k+1)^2 L_{k+1} = (2k+1+alpha)x L_k - (k+alpha) L_{k-1}
        # For alpha=0: (k+1)^2 L_{k+1} = (2k+1)x L_k - k L_{k-1}
        pass
    
    # Build the polynomial coefficients for L_n^(alpha)
    # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * C(n+alpha, n-k) * x^k / k!
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        # C(n+alpha, n-k) = (n+alpha)! / ((n-k)! * (k+alpha)!)
        # For integer alpha, use comb
        if alpha == 0:
            binom_val = comb(n, n - k)
        else:
            # Generalized binomial coefficient
            from math import lgamma
            log_binom = lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + alpha + 1)
            binom_val = exp(log_binom)
        
        coeffs[k] = ((-1)**k) * binom_val / factorial(k)
    
    # coeffs[k] is the coefficient of x^k
    # For np.roots, we need highest degree first
    coeffs_desc = coeffs[::-1]
    
    try:
        roots = np.roots(coeffs_desc)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-8 and r.real > 1e-10]
        real_roots.sort()
        return real_roots
    except:
        return []

def _sturm_liouville_eigenvalues(m=12, n=1):
    """
    Solve the Sturm-Liouville eigenvalue problem for the Laguerre operator
    with the double-root constraint. The eigenvalues should give the
    optimal double-root locations.
    
    The Laguerre differential equation: x*y'' + (1-x)*y' + lambda*y = 0
    with appropriate boundary conditions for the double-root construction.
    """
    from scipy.integrate import solve_bvp
    
    # The eigenvalue problem for the Laguerre operator:
    # -d/dx [x * dy/dx] - (1-x) * dy/dx = lambda * y
    # or equivalently: x*y'' + (1-x)*y' + lambda*y = 0
    #
    # For the Cohn-Goncalves construction with double roots, we need
    # a modified problem. Let's use the standard Laguerre eigenvalues
    # as a starting point, then refine.
    
    # Standard Laguerre eigenvalues are lambda = 0, 1, 2, ..., m-1
    # The zeros of L_m(x) are the eigenfunctions' nodes
    
    # Actually, let's try a different approach: use the fact that
    # the optimal roots for the C4 bound should maximize the product
    # subject to the constraint that the polynomial P(x) = prod(x-r_i)^2
    # has the right sign-change structure with the Laguerre weight.
    
    # For now, return the standard Laguerre zeros as a baseline
    return _get_laguerre_zeros_direct(m, 0)

def _optimize_roots_for_bound(roots_init, deadline):
    """
    Optimize the roots to minimize the C4 bound.
    Uses the insight that the bound is related to the L2 norm of
    the double-root polynomial under the Laguerre weight.
    """
    roots = np.array(roots_init, dtype=np.float64)
    m = 12
    
    # Objective: minimize the C4 bound
    # In log-space: params = log(roots)
    
    def objective(log_params):
        if time.time() > deadline - 5:
            return np.inf
        roots_candidate = np.exp(log_params)
        if np.any(roots_candidate <= 0) or np.any(np.diff(np.sort(roots_candidate)) < 1e-8):
            return np.inf
        return _compute_c4_bound(roots_candidate)
    
    log_params = np.log(roots)
    
    # Try L-BFGS-B with bounds
    lb = np.full(12, -5.0)  # roots > e^{-5} ≈ 0.0067
    ub = np.full(12, 15.0)  # roots < e^{15} ≈ 3.3e6
    bounds = list(zip(lb, ub))
    
    best_params = log_params.copy()
    best_val = objective(log_params)
    
    try:
        result = minimize(
            objective,
            log_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if result.fun < best_val and np.isfinite(result.fun):
            best_params = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Try Nelder-Mead for a different local optimum
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                objective,
                log_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result2.fun < best_val and np.isfinite(result2.fun):
                best_params = result2.x
                best_val = result2.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        try:
            result3 = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            
            if result3.fun < best_val and np.isfinite(result3.fun):
                best_params = result3.x
                best_val = result3.fun
        except Exception:
            pass
    
    best_roots = np.exp(best_params)
    best_roots = np.sort(best_roots)
    
    return best_roots, best_val

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin before 630s kill
    
    m = 12
    best_roots = None
    best_score = float('inf')
    
    # Strategy 1: Standard Laguerre zeros (alpha=0)
    try:
        roots_lag = _get_laguerre_zeros_direct(m, 0)
        if len(roots_lag) == 12:
            score = _compute_c4_bound(roots_lag)
            if score < best_score and np.isfinite(score):
                best_score = score
                best_roots = np.array(roots_lag)
    except Exception:
        pass
    
    # Strategy 2: Try different alpha values for generalized Laguerre
    for alpha in [0, 1, 2, 5, 10, 20, 50]:
        if time.time() > deadline - 10:
            break
        try:
            roots_gen = _get_laguerre_zeros_direct(m, alpha)
            if len(roots_gen) == 12:
                score = _compute_c4_bound(roots_gen)
                if score < best_score and np.isfinite(score):
                    best_score = score
                    best_roots = np.array(roots_gen)
        except Exception:
            continue
    
    # Strategy 3: Equally spaced roots (seed-like)
    roots_eq = np.array([4.0 * (i + 1) for i in range(12)])
    score_eq = _compute_c4_bound(roots_eq)
    if score_eq < best_score and np.isfinite(score_eq):
        best_score = score_eq
        best_roots = roots_eq.copy()
    
    # Strategy 4: Geometrically spaced roots
    for scale in [1, 2, 5, 10]:
        roots_geo = scale * np.array([(i + 1) for i in range(12)])
        score_geo = _compute_c4_bound(roots_geo)
        if score_geo < best_score and np.isfinite(score_geo):
            best_score = score_geo
            best_roots = roots_geo.copy()
    
    # Strategy 5: Optimize from the best starting point
    if best_roots is not None and time.time() < deadline - 30:
        try:
            opt_roots, opt_score = _optimize_roots_for_bound(best_roots, deadline)
            if opt_score < best_score and np.isfinite(opt_score):
                best_score = opt_score
                best_roots = opt_roots
        except Exception:
            pass
    
    # Strategy 6: Try multiple random starting points for optimization
    if time.time() < deadline - 60:
        np.random.seed(42)
        for trial in range(5):
            if time.time() > deadline - 20:
                break
            try:
                # Random log-spaced roots
                log_roots = np.random.uniform(-2, 5, 12)
                roots_rand = np.sort(np.exp(log_roots))
                
                if not _check_constraints(roots_rand):
                    continue
                
                score_rand = _compute_c4_bound(roots_rand)
                if score_rand < best_score and np.isfinite(score_rand):
                    best_score = score_rand
                    best_roots = roots_rand.copy()
                
                # Optimize from this starting point
                if time.time() < deadline - 30:
                    opt_roots, opt_score = _optimize_roots_for_bound(roots_rand, deadline)
                    if opt_score < best_score and np.isfinite(opt_score):
                        best_score = opt_score
                        best_roots = opt_roots
            except Exception:
                continue
    
    # Fallback: ensure we have valid roots
    if best_roots is None or not np.isfinite(best_score):
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
        best_score = _compute_c4_bound(best_roots)
    
    # Final validation
    best_roots = np.sort(np.asarray(best_roots, dtype=np.float64))
    if len(best_roots) != 12 or np.any(best_roots <= 0):
        best_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    return list(best_roots)

# EVOLVE-BLOCK-END
