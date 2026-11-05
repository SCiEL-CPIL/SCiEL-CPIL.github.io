import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize

def construct():
    # The task specifies m=12. We start with the roots of the Laguerre polynomial of order 12
    # as an initial guess, but we "tune" them by optimizing a proxy for the uncertainty constant.
    roots, _ = roots_laguerre(12)
    
    def cost_function(r):
        # Ensure roots are positive
        if np.any(r <= 0):
            return 1e10
        r = np.sort(r)
        
        # Proxy for uncertainty: Variance of the distribution |P(x)|^2 e^{-x}
        # Use numerical integration
        x = np.linspace(0, 150, 2000)
        
        # Calculate log|P(x)| to handle large numbers
        # P(x) = prod(x - r_i)
        # log|P(x)| = sum(log|x - r_i|)
        # Handle x == r_i carefully (log 0 is -inf, exp(-inf) is 0, which is correct for P(x)=0)
        # But we integrate P(x)^2, so it's fine.
        
        diff = x[:, None] - r[None, :]
        # Avoid log(0) by clipping
        abs_diff = np.abs(diff)
        abs_diff[abs_diff < 1e-10] = 1e-10
        
        log_p = np.sum(np.log(abs_diff), axis=1)
        p_sq = np.exp(2 * log_p)
        w = p_sq * np.exp(-x)
        
        norm = np.trapz(w, x)
        if norm < 1e-10:
            return 1e10
            
        m1 = np.trapz(x * w, x) / norm
        m2 = np.trapz(x**2 * w, x) / norm
        variance = m2 - m1**2
        
        return variance

    # Optimize the roots
    # Bounds: roots must be positive. Laguerre roots are roughly in [0, 4*m].
    # We use L-BFGS-B with bounds.
    bounds = [(1e-6, 100.0)] * 12
    result = minimize(cost_function, roots, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 500, 'ftol': 1e-8})
    
    # Return as a list of floats
    return [float(r) for r in result.x]
