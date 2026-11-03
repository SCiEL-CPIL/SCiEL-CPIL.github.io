"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize_scalar, differential_evolution, minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Returns a list of 12 positive root locations.
    Score is the upper bound on C4 (lower is better for the evaluator's raw value,
    but the search score is seed/value, so higher search score = better).
    """
    start_time = time.time()
    DEADLINE = 630.0  # seconds wall-clock limit
    SAFETY_MARGIN = 30.0  # return 30s before the kill

    def time_ok():
        return (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN)

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # We want c such that P(x) = sum c_k L_k^{(1)}(x)
    # P(x) = sum p_j x^j
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # So P(x) = sum_j p_j x^j = sum_k c_k (sum_j A[k,j] x^j) = sum_j (sum_k c_k A[k,j]) x^j
    # Thus p_j = sum_k c_k A[k,j] => p = A^T c => c = (A^T)^{-1} p
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            # L_k^{(alpha)}(x) = sum_{j=0}^k (-1)^j * binom(k+alpha, k-j) * x^j / j!
            # For alpha=1: (-1)^j * binom(k+1, k-j) / j!
            # binom(k+1, k-j) = binom(k+1, j+1)
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. We can precompute the inverse of A.T
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_coefficients(roots):
        """Compute Laguerre coefficients c_k for P(x) = prod(x - r_i)"""
        # Build polynomial coefficients in standard basis
        # p[0] + p[1]x + ... + p[12]x^12
        # Roots are positive, so factors are (x - r_i)
        # Let's use np.poly which returns [1, -sum, ...] for monic polynomial
        # np.poly(roots) returns [1, c1, c2, ..., cm] where P(x) = x^m + c1 x^{m-1} + ...
        # We need p in ascending order: p[0] is constant term.
        coeffs_desc = np.poly(roots) # [1, -e1, e2, ..., (-1)^12 e12]
        p = coeffs_desc[::-1] # [(-1)^12 e12, ..., -e1, 1]
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        return c

    def compute_bound(roots):
        """
        Compute the upper bound on C4.
        The bound is related to 1 / ||P||^2_Laguerre.
        We want to minimize the bound, so we want to maximize ||P||^2.
        """
        if any(r <= 0 for r in roots):
            return 1e10
        
        c = compute_coefficients(roots)
        
        # Norm squared in Laguerre basis for n=1:
        # <L_k^{(1)}, L_l^{(1)}> = delta_{kl} * (k+1)! / (1! k!) = delta_{kl} * (k+1)
        norm_sq = sum(c[k]**2 * (k+1) for k in range(13))
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    def get_roots_from_alpha(alpha):
        """
        Get roots of L_12^{(alpha)}(x).
        We use the companion matrix method for high precision.
        """
        # Coefficients of L_m^{(alpha)}(x) in descending order
        # L_m^{(alpha)}(x) = sum_{j=0}^m (-1)^j binom(m+alpha, m-j) x^j / j!
        # Leading coeff (j=0): 1
        # Next (j=1): - (m+alpha)
        # ...
        # Constant term (j=m): (-1)^m binom(m+alpha, 0) / m! = (-1)^m / m!
        
        # np.roots expects coefficients in descending order of powers.
        # P(x) = c_m x^m + ... + c_0
        # Here, the polynomial is monic.
        # Coeff of x^k is (-1)^{m-k} binom(m+alpha, k) / (m-k)! ? No.
        
        # Formula: L_m^{(alpha)}(x) = \sum_{k=0}^m (-1)^k \binom{m+\alpha}{m-k} \frac{x^k}{k!}
        # Let's re-index for descending powers x^m, x^{m-1}, ..., x^0.
        # Term with x^k corresponds to index k in the sum.
        # Power is k.
        # Coeff of x^k is (-1)^k \binom{m+\alpha}{m-k} / k!
        
        # Wait, standard definition:
        # L_n^{(\alpha)}(x) = \sum_{k=0}^n \binom{n+\alpha}{n-k} \frac{(-x)^k}{k!}
        # So coeff of x^k is \binom{n+\alpha}{n-k} (-1)^k / k!
        
        coeffs = []
        for k in range(m, -1, -1):
            # Power of x is k
            # Term in sum where index j = k
            # Coeff = (-1)^k * binom(m+alpha, m-k) / k!
            # Note: binom(m+alpha, m-k) = binom(m+alpha, k)
            val = ((-1)**k) * math.comb(int(m+alpha) if m+alpha==int(m+alpha) else 0, m-k) # This comb logic is wrong for non-integer alpha
            # Use gamma function for general alpha
            # binom(m+alpha, m-k) = Gamma(m+alpha+1) / (Gamma(m-k+1) Gamma(k+alpha+1))
            # Actually, simpler:
            # \binom{n+\alpha}{n-k} = \frac{(n+\alpha)!}{(n-k)! (k+\alpha)!}
            # Let's just compute the term directly using the recurrence or formula.
            # Formula: c_k = (-1)^k / k! * \binom{m+\alpha}{m-k}
            
            # For numerical stability and simplicity, let's use scipy.special.eval_genlaguerre to get coefficients?
            # No, eval_genlaguerre evaluates the polynomial, it doesn't give coefficients easily.
            
            # Let's use the explicit formula with math.gamma
            try:
                binom_val = math.gamma(m + alpha + 1) / (math.gamma(m - k + 1) * math.gamma(k + alpha + 1))
            except (ValueError, OverflowError):
                return None
            
            coeff = ((-1)**k) * binom_val / math.factorial(k)
            coeffs.append(coeff)
            
        # coeffs[0] is coeff of x^m (should be 1)
        # coeffs[-1] is coeff of x^0
        roots = np.roots(coeffs)
        
        # Filter for real positive roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                real_part = r.real
                if real_part > 1e-10:
                    real_roots.append(real_part)
        
        if len(real_roots) != m:
            return None
        
        return np.array(real_roots)

    # Strategy: 1D search over alpha
    # The seed was geometric: 4, 8, ..., 8192.
    # Roots of L_12^{(alpha)} scale roughly with alpha + 12/2 + ...
    # For large alpha, roots are spread out.
    # Let's scan alpha in a range likely to cover the geometric sequence scale.
    # Geometric mean of seed roots: sqrt(4*8192) ~ 181.
    # Roots of L_n^{(alpha)} are roughly in [0, (n+alpha+1)^2].
    # So alpha should be such that scale matches.
    
    best_bound = 1e10
    best_roots = None
    
    # Initial coarse scan
    alpha_min = -10.0
    alpha_max = 200.0
    num_points = 200
    
    for i in range(num_points):
        if not time_ok():
            break
        alpha = alpha_min + (alpha_max - alpha_min) * i / (num_points - 1)
        roots = get_roots_from_alpha(alpha)
        if roots is None:
            continue
        bound = compute_bound(roots)
        if bound < best_bound:
            best_bound = bound
            best_roots = roots.copy()
            best_alpha = alpha

    # Refine around the best alpha found
    if best_alpha is not None:
        # Local refinement
        def alpha_cost(alpha):
            roots = get_roots_from_alpha(alpha)
            if roots is None:
                return 1e10
            return compute_bound(roots)
        
        # Use golden section or bounded minimization
        # Range around best_alpha
        local_min = max(-10.0, best_alpha - 10.0)
        local_max = min(200.0, best_alpha + 10.0)
        
        try:
            res = minimize_scalar(alpha_cost, bounds=(local_min, local_max), method='bounded', options={'xatol': 1e-6})
            if res.success:
                refined_alpha = res.x
                roots = get_roots_from_alpha(refined_alpha)
                if roots is not None:
                    bound = compute_bound(roots)
                    if bound < best_bound:
                        best_bound = bound
                        best_roots = roots.copy()
        except Exception:
            pass

    # If the 1D search didn't find anything good or failed, fallback to seed
    if best_roots is None:
        r0 = 4.0
        q = 2.0
        best_roots = np.array([r0 * (q ** i) for i in range(12)])
        best_bound = compute_bound(best_roots)
    else:
        # Optional: Use the 1D result as a starting point for a short 12D optimization?
        # The prompt says "Keep this design and investigate improvements within its family".
        # The design is now 1D search. Adding 12D might be "refining" but the directive says "Represent ... as nodes ... reducing 12D to 1D".
        # I will stick to the 1D result to respect the "MATERIALLY DIFFERENT" strategy of reducing dimensionality.
        # However, the roots from L_12^{(alpha)} are specific. 
        # Is it possible that a slight perturbation of these roots improves the bound?
        # The task asks to "Tune the refined Cohn-Goncalves ... construction". 
        # The construction *is* the Laguerre polynomial. The roots *are* the nodes.
        # So the 1D search is the direct implementation of the strategy.
        
        pass

    # Ensure roots are positive and sorted? 
    # The task doesn't require sorting, just a list of locations.
    # But double roots implies we just list them.
    
    return list(best_roots)
# EVOLVE-BLOCK-END
