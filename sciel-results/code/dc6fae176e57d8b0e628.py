"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize

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

    # The Laguerre polynomial combination with double roots at x_i:
    # P(x) = prod_{i=1}^{12} (x - x_i)^2
    # This must be expressible as a linear combination of Laguerre polynomials L_k^{(n)}(x) for k=0..m
    # i.e., P(x) = sum_{k=0}^{m} c_k L_k^{(n)}(x)
    # where L_k^{(n)} are generalized Laguerre polynomials with parameter n=1.

    # The constraint is that P(x) lies in the span of {L_0^{(1)}, ..., L_12^{(1)}}.
    # Since P(x) is a degree-24 polynomial and the span of L_0..L_12 is only degree 12,
    # we need a different formulation.

    # Actually, re-reading the task: "12 double roots with m=12 and dimension n=1"
    # The Cohn-Goncalves construction uses a polynomial of degree m with specific properties.
    # Let me reconsider: the polynomial is of degree m=12, and it has 12 roots (each a double root? No, that would be degree 24).

    # Let me re-read: "12 double roots with m=12" - perhaps it means the polynomial has degree m=12
    # and the "double roots" refer to the construction method, not that each root has multiplicity 2.
    # Or perhaps it's a product of 6 double roots (12 total root count with multiplicity).

    # Actually, for the Fourier sign uncertainty problem with the Cohn-Goncalves method:
    # We construct a polynomial Q of degree m such that Q has m positive roots,
    # and the sign changes of the Fourier transform relate to the roots.
    # The "double roots" likely means each root appears with multiplicity 2 in some auxiliary polynomial.

    # Let me use the approach from the knowledge: re-parameterize as geometric sequence r0*q^i
    # and optimize to minimize the upper bound on C4.

    # The upper bound on C4 relates to 1/||psi||^2 where psi is constructed from the roots.
    # From the knowledge: "optimize (log r0, log q) with L-BFGS-B multistart"
    # and "this lets the optimizer descend away from the seed's suboptimal q=2 toward the true minimum of 1/||psi||^2"

    # Let me define the cost function properly.
    # For the Laguerre polynomial construction with m=12, n=1:
    # The polynomial is P(x) = prod_{i=0}^{11} (x - r_i) where r_i are the 12 roots.
    # We express P(x) = sum_{k=0}^{12} c_k * L_k^{(1)}(x)
    # The coefficients c_k are determined by the roots.
    # The uncertainty bound involves the norm of the associated function.

    # A simpler model: the upper bound on C4 is related to the norm of the polynomial
    # in the Laguerre basis. Specifically, if P(x) = sum c_k L_k^{(1)}(x),
    # then ||P||^2 = sum_{k=0}^{12} c_k^2 * <L_k^{(1)}, L_k^{(1)}> = sum c_k^2 * (12+k)!/(1! k!)... 

    # Actually, the orthogonality: <L_k^{(n)}, L_l^{(n)}> = delta_{kl} * (n+k)!/(n! k!)
    # For n=1: <L_k^{(1)}, L_k^{(1)}> = (k+1)!/k! = k+1

    # So ||P||^2 = sum_{k=0}^{12} c_k^2 * (k+1)

    # The upper bound on C4 might be proportional to ||P||^2 or 1/||P||^2 or similar.
    # From the knowledge: "minimum of 1/||psi||^2" - so we want to MINIMIZE 1/||psi||^2
    # which means we want to MAXIMIZE ||psi||^2.

    # But wait, the task says "score is the upper bound on C4" and "raw value (lower is better)".
    # And "search score is seed/value". So if value is the upper bound, lower value is better,
    # and search score = seed_value / value. If value < seed_value, search score > 1.

    # Let me just implement the optimization as described in the knowledge:
    # Parameterize roots as r0 * q^i for i=0..11, optimize (log r0, log q) to minimize the cost.

    def compute_coefficients(roots):
        """Compute Laguerre coefficients c_k for P(x) = prod(x - r_i) = sum c_k L_k^{(1)}(x)"""
        # Build the polynomial P(x) = prod_{i=0}^{11} (x - r_i)
        # Coefficients in standard basis: p[0] + p[1]*x + ... + p[12]*x^12
        p = np.array([1.0])
        for r in roots:
            p = np.convolve(p, np.array([1.0, -r]))
        # p has 13 coefficients: p[0] is constant, p[12] is x^12 coeff
        
        # Convert from standard basis to Laguerre basis L_k^{(1)}(x)
        # L_k^{(1)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+1, j) * x^{k-j} / j!
        # We need to express x^j in terms of L_k^{(1)}.
        # Build the transformation matrix M where x^j = sum_{k=0}^{j} M[j,k] * L_k^{(1)}(x)
        
        # First, express L_k^{(1)}(x) in standard basis:
        # L_k^{(1)}(x) = sum_{j=0}^{k} a_{k,j} x^j
        # a_{k,j} = (-1)^{k-j} * binom(k+1, j) / (k-j)!
        # Wait, let me use the explicit formula:
        # L_k^{(alpha)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+alpha, k-j) * x^j / j!
        # For alpha=1: L_k^{(1)}(x) = sum_{j=0}^{k} (-1)^j * binom(k+1, k-j) * x^j / j!
        # = sum_{j=0}^{k} (-1)^j * binom(k+1, j+1) * x^j / j!  [since binom(k+1,k-j)=binom(k+1,j+1)]
        
        # Let's build the matrix A where A[k,j] is the coefficient of x^j in L_k^{(1)}(x)
        A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}
        for k in range(13):
            for j in range(k + 1):
                # (-1)^j * binom(k+1, k-j) * 1/j!
                # binom(k+1, k-j) = binom(k+1, j+1)
                A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
        
        # Now P(x) = sum_{j=0}^{12} p[j] * x^j = sum_{k=0}^{12} c_k * L_k^{(1)}(x)
        # In matrix form: p = A^T @ c, so c = (A^T)^{-1} @ p = (A^T)^{-1} @ p
        # Or: c = solve(A^T, p)
        try:
            c = np.linalg.solve(A.T, p)
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(A.T, p, rcond=None)[0]
        return c

    def cost_func(params):
        """Compute the upper bound on C4 given (log_r0, log_q)"""
        log_r0, log_q = params
        r0 = math.exp(log_r0)
        q = math.exp(log_q)
        if r0 <= 0 or q <= 0:
            return 1e10
        roots = [r0 * (q ** i) for i in range(12)]
        # Ensure all roots are positive and reasonably well-separated
        if any(r <= 0 for r in roots):
            return 1e10
        c = compute_coefficients(roots)
        # The norm squared in Laguerre basis: ||P||^2 = sum c_k^2 * (k+1) for n=1
        norm_sq = sum(c[k]**2 * (k+1) for k in range(13))
        if norm_sq <= 0:
            return 1e10
        # The upper bound on C4 is related to 1/norm_sq (we want to minimize this)
        # Or it could be norm_sq itself. Let's use 1/norm_sq as the cost to minimize.
        # Actually, from the knowledge: "minimum of 1/||psi||^2" - so cost = 1/norm_sq
        return 1.0 / norm_sq

    # Multistart optimization
    best_value = 1e10
    best_params = None
    
    # The seed is r0=4, q=2, so log_r0=log(4), log_q=log(2)
    # Start from seed and several perturbations
    seed_params = (math.log(4.0), math.log(2.0))
    
    start_points = [seed_params]
    # Add perturbations
    np.random.seed(42)
    for _ in range(50):
        dr0 = np.random.uniform(-2.0, 2.0)
        dq = np.random.uniform(-1.5, 1.5)
        start_points.append((seed_params[0] + dr0, seed_params[1] + dq))
    
    for sp in start_points:
        if not time_ok():
            break
        try:
            result = minimize(cost_func, sp, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < best_value:
                if result.fun < best_value:
                    best_value = result.fun
                    best_params = result.x
        except Exception:
            continue
    
    # If optimization failed, fall back to seed
    if best_params is None:
        best_params = seed_params
    
    log_r0, log_q = best_params
    r0 = math.exp(log_r0)
    q = math.exp(log_q)
    roots = [r0 * (q ** i) for i in range(12)]
    
    # Ensure roots are positive
    roots = [max(r, 1e-10) for r in roots]
    
    return roots
# EVOLVE-BLOCK-END
