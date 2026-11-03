"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution

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

    def cost_func(params):
        """
        Compute the upper bound on C4.
        The bound is related to 1 / ||P||^2_Laguerre.
        We want to minimize the bound, so we want to maximize ||P||^2.
        However, the task says "score is the upper bound on C4" and "lower is better".
        So we return the bound value.
        
        Wait, the search score is seed/value. If value is the bound, and we want to improve,
        we need value < seed_value.
        
        Let's look at the previous program's logic:
        "return 1.0 / norm_sq"
        If norm_sq is large, 1/norm_sq is small.
        So minimizing 1/norm_sq means maximizing norm_sq.
        
        Is the bound proportional to 1/norm_sq or norm_sq?
        Usually uncertainty bounds involve ratios.
        Let's stick to the previous interpretation: Cost = 1 / ||P||^2_Laguerre.
        """
        roots = params
        if any(r <= 0 for r in roots):
            return 1e10
        
        c = compute_coefficients(roots)
        
        # Norm squared in Laguerre basis for n=1:
        # <L_k^{(1)}, L_l^{(1)}> = delta_{kl} * (k+1)! / (1! k!) = delta_{kl} * (k+1)
        norm_sq = sum(c[k]**2 * (k+1) for k in range(13))
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Differential Evolution over 12 independent positive roots
    # Bounds: roots should be positive. 
    # The seed was r0=4, q=2 -> roots: 4, 8, 16, ..., 4*2^11 = 8192.
    # Let's set bounds [0.1, 10000] to be safe.
    bounds = [(0.1, 10000.0)] * 12
    
    # Initial seed for DE
    # We can use the geometric sequence as a hint, but DE doesn't take a single start point.
    # We can constrain the search or just let it explore.
    # To help convergence, we might use a tighter bound if we knew the scale better, 
    # but 0.1 to 10000 covers the seed.
    
    # We run DE with a limited number of iterations to fit within time.
    # popsize=15, maxiter=100. 15*100 = 1500 evaluations.
    # Each evaluation is O(1) mostly, just matrix-vector mult. Very fast.
    # We can afford more iterations if needed, but 100 is usually enough for DE to find a good basin.
    
    try:
        result = differential_evolution(
            cost_func, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8, 
            polish=True,  # Use L-BFGS-B for final polish
            callback=None
        )
        
        if result.success:
            best_roots = result.x
            best_val = result.fun
        else:
            best_roots = result.x
            best_val = result.fun
            
    except Exception as e:
        # Fallback to seed geometric sequence
        r0 = 4.0
        q = 2.0
        best_roots = np.array([r0 * (q ** i) for i in range(12)])
        best_val = cost_func(best_roots)

    # Ensure roots are positive
    best_roots = np.maximum(best_roots, 1e-6)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
