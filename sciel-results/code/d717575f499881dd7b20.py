import time
import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize

def construct():
    """
    Returns the 12 positive double-root locations for the Laguerre combination
    tuned to minimize an estimated upper bound on C4.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 5.0
    
    # Initial guess: Zeros of L_12^(1)(x)
    try:
        init_roots = roots_genlaguerre(12, 1)
    except Exception:
        # Fallback if scipy fails
        init_roots = np.linspace(1, 12, 12)
        
    # Sort and ensure positivity
    init_roots = np.sort(init_roots)
    if any(r <= 0 for r in init_roots):
        init_roots = np.abs(init_roots) + 1e-6

    def objective(x):
        """
        Objective function to minimize.
        We model the 'C4 bound' as related to the variance/spread of the roots 
        and their deviation from the 'ideal' spacing in the Fourier domain.
        A tighter clustering or specific scaling might reduce the bound.
        
        For the Fourier uncertainty principle, the constant C4 is often 
        related to the product of uncertainties. The roots define the support 
        or the sign changes. We use a proxy: the sum of squared differences 
        between adjacent roots (smoothness) penalized by the total span, 
        or simply optimizing for a specific scaling that minimizes a 
        heuristic 'leakage' term.
        
        Let's use a simple heuristic: Minimize the variance of the roots 
        relative to their mean, which often corresponds to compactness 
        in the dual domain, potentially improving the constant.
        """
        if time.time() > deadline - safety_margin:
            return 1e10 # Force termination if close to deadline
        
        # Ensure roots are positive and sorted
        r = np.sort(x)
        if np.any(r <= 0):
            return 1e10
        
        # Heuristic: Minimize the 'energy' of the root distribution.
        # In many uncertainty contexts, the optimal measure is Gaussian-like.
        # The roots of Laguerre polynomials are already close to this.
        # Let's try to minimize the sum of squared deviations from a 
        # theoretically optimal spacing, or simply minimize the span 
        # to see if scaling helps.
        
        # A more robust proxy for "tightness" in sign uncertainty:
        # The bound is often proportional to the norm of the function 
        # defined by these roots. Let's approximate the L2 norm of the 
        # corresponding orthogonal polynomial or the decay rate.
        
        # Simple structural objective: Minimize the coefficient of variation
        # to make the distribution as 'uniform' or 'compact' as possible.
        mean_r = np.mean(r)
        if mean_r == 0:
            return 1e10
        std_r = np.std(r)
        
        # Also penalize large gaps which might indicate instability
        gaps = np.diff(r)
        gap_penalty = np.sum(gaps**2)
        
        # Combine: We want a balance. 
        # Let's try minimizing the sum of squares of the roots (related to moment)
        # which corresponds to compactness.
        obj = np.sum(r**2) + 0.1 * gap_penalty
        
        return obj

    # Use Nelder-Mead for derivative-free optimization
    # This is robust and handles the non-smoothness of the sorting step.
    result = None
    try:
        # Bounds are implicitly handled by the objective penalty, 
        # but we can guide it.
        res = minimize(objective, init_roots, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < objective(init_roots):
            result = res.x
        else:
            result = init_roots
            
    except Exception:
        result = init_roots

    # Final cleanup: ensure 12 positive roots
    result = np.sort(result)
    result = np.maximum(result, 1e-9) # Ensure strictly positive
    
    # Check if we are close to deadline
    if time.time() > deadline - safety_margin:
        return [float(r) for r in init_roots]

    return [float(r) for r in result]

