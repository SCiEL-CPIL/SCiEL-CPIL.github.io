"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import eval_genlaguerre, roots_laguerre

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Minimizes the Cohn-Goncalves upper bound on C4 by optimizing the parameter alpha
    of the generalized Laguerre polynomial L_12^(alpha)(x).
    """
    deadline = time.time() + 570
    
    # The "value" (upper bound on C4) is hypothesized to be the largest root
    # of the polynomial used in the construction.
    # The construction uses a Laguerre polynomial of degree 12.
    # By using the generalized Laguerre polynomial L_12^(alpha)(x), we can
    # shift the roots. We search for alpha that minimizes the largest root.
    
    def get_laguerre_roots(alpha):
        """
        Compute roots of L_12^(alpha)(x).
        L_n^(alpha)(x) is defined for alpha > -1.
        We use numpy to find roots of the polynomial.
        """
        n = 12
        # Coefficients of L_n^(alpha)(x)
        # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
        # Or use scipy.special.eval_genlaguerre to evaluate, and find roots via brentq?
        # Better: use numpy.polynomial.laguerre.lag2orth? No.
        # Let's use the explicit formula to get coefficients.
        
        # coeffs are for x^0 ... x^n
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            # term is (-1)^k * binom(n+alpha, n-k) / k! * x^k
            # binom(n+alpha, n-k) = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
            # This can be numerically unstable.
            # Alternative: use scipy.special.genlaguerre?
            # scipy.special.genlaguerre(n, alpha) returns a poly1d object?
            # No, it returns a function.
            
            # Let's use the recurrence or just evaluate on a grid and find roots?
            # Grid search for roots is robust enough for n=12.
            pass
            
        # More robust: use scipy.optimize.brentq on intervals.
        # Roots of L_n^(alpha)(x) are positive and simple.
        # We can bracket them.
        # L_12^(alpha)(x) oscillates.
        # Let's just use the roots of the standard Laguerre polynomial if alpha=0.
        # For alpha != 0, we need to compute.
        
        # Using numpy.polynomial.laguerre.Laguerre
        # There is a function `numpy.polynomial.laguerre.laggauss` for weights, not roots of L_n^alpha.
        
        # Let's use the explicit coefficients from the formula:
        # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * (Gamma(n+alpha+1) / (Gamma(k+alpha+1) * (n-k)!)) * x^k / k! ?
        # Actually, standard formula:
        # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
        
        # Let's use log-gamma for stability.
        from math import lgamma, exp
        
        coeffs = np.zeros(n + 1)
        for k in range(n + 1):
            # coeff of x^k is (-1)^k * binom(n+alpha, n-k) / k!
            # binom(n+alpha, n-k) = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
            # log_binom = lgamma(n+alpha+1) - lgamma(n-k+1) - lgamma(k+alpha+1)
            # log_term = log_binom - lgamma(k+1)
            # term = (-1)^k * exp(log_term)
            
            # Note: alpha must be such that k+alpha+1 > 0 for all k.
            # If alpha > -1, this is fine.
            
            log_binom = lgamma(n + alpha + 1) - lgamma(n - k + 1) - lgamma(k + alpha + 1)
            log_fact = lgamma(k + 1)
            log_val = log_binom - log_fact
            val = exp(log_val)
            if k % 2 == 1:
                val = -val
            coeffs[k] = val
            
        # Find roots of polynomial with coefficients coeffs[0] ... coeffs[n]
        # numpy.roots expects highest degree first.
        poly_coeffs = coeffs[::-1]
        roots = np.roots(poly_coeffs)
        
        # Filter real, positive roots
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        real_roots = real_roots[real_roots > 0]
        real_roots.sort()
        
        # Should have n roots
        if len(real_roots) != n:
            # Fallback to standard Laguerre if computation fails
            std_roots, _ = roots_laguerre(n)
            return std_roots
            
        return real_roots

    def objective(alpha):
        try:
            if alpha <= -0.9: # Avoid singularities
                return 1e10
            roots = get_laguerre_roots(alpha)
            return np.max(roots)
        except Exception:
            return 1e10

    # Search for optimal alpha
    # Roots shift left as alpha decreases?
    # Let's check alpha in [-0.5, 5.0]
    res = minimize_scalar(objective, bounds=(-0.5, 5.0), method='bounded')
    
    if res.success:
        best_alpha = res.x
    else:
        best_alpha = 0.0
        
    # Get the roots for the best alpha
    best_roots = get_laguerre_roots(best_alpha)
    
    result = list(best_roots)
    
    # Verify
    assert len(result) == 12
    assert all(r > 0 for r in result)
    assert all(result[i] < result[i+1] for i in range(11))
    
    return result
# EVOLVE-BLOCK-END
