import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time

def construct():
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    m = 12  # number of double roots
    n = 1   # dimension
    
    def compute_bound(roots):
        """Compute the C4 upper bound given 12 double-root locations."""
        # roots: array of 12 positive values, sorted
        roots = np.sort(roots)
        
        # P(x) = prod_{i=1}^{12} (x - x_i)^2
        # P(x)^2 = prod_{i=1}^{12} (x - x_i)^4
        # We need integral_0^inf P(x)^2 e^{-x} dx
        # P(x)^2 is a polynomial of degree 48
        # integral = sum_k c_k * k! where c_k is coefficient of x^k
        
        # Build P(x)^2 as a polynomial
        # Start with 1, multiply by (x - x_i)^4 for each i
        poly = np.array([1.0])
        for x_i in roots:
            # (x - x_i)^4 = x^4 - 4*x_i*x^3 + 6*x_i^2*x^2 - 4*x_i^3*x + x_i^4
            factor = np.array([1.0, -4*x_i, 6*x_i**2, -4*x_i**3, x_i**4])
            poly = np.convolve(poly, factor)
        
        # poly[k] is coefficient of x^k (poly[0] = constant term)
        # integral = sum_{k=0}^{48} poly[k] * k!
        num = 0.0
        for k in range(len(poly)):
            num += poly[k] * factorial(k)
        
        # Denominator: prod_{i<j} (x_j - x_i)^4
        denom = 1.0
        for i in range(m):
            for j in range(i+1, m):
                diff = roots[j] - roots[i]
                if diff <= 0:
                    return np.inf
                denom *= diff**4
        
        if denom <= 0 or not np.isfinite(denom):
            return np.inf
        
        return num / denom
    
    def objective(roots):
        """Objective function for minimization."""
        # Ensure positivity
        roots = np.maximum(roots, 1e-6)
        val = compute_bound(roots)
        if not np.isfinite(val):
            return 1e30
        return val
    
    # Asymptotic initialization: zeros of Laguerre polynomial L_12^(0)(x)
    # For large m, zeros of L_m^(α)(x) are approximately distributed
    # according to a known distribution. For m=12, let's use known values.
    
    # The zeros of L_12(x) (α=0) can be approximated.
    # For the double-root construction, the optimal roots should be near
    # the zeros of the Laguerre polynomial or related quantities.
    
    # Let's use the actual zeros of L_12^(0)(x) as initial guess
    from scipy.special import roots_laguerre
    lag_zeros = roots_laguerre(m)[0]
    
    # Multi-start optimization
    best_val = np.inf
    best_roots = lag_zeros.copy()
    
    # Several starting points
    starts = [
        lag_zeros.copy(),
        lag_zeros * 1.1,
        lag_zeros * 0.9,
        np.linspace(0.5, 50, m),
        np.linspace(1.0, 40, m),
        lag_zeros + np.random.RandomState(42).randn(m) * 0.5,
    ]
    
    for start in starts:
        if time.time() > deadline - 60:
            break
        
        start = np.maximum(start, 0.1)
        start = np.sort(start)
        
        try:
            result = minimize(
                objective,
                start,
                method='L-BFGS-B',
                bounds=[(0.01, 1000)] * m,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_val:
                val = result.fun
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_roots = np.sort(result.x)
        except Exception:
            continue
    
    # Verify the result
    final_roots = np.sort(best_roots)
    final_val = compute_bound(final_roots)
    
    if not np.isfinite(final_val):
        # Fallback to seed
        return [4.0 * (i + 1) for i in range(m)]
    
    return [float(x) for x in final_roots]
