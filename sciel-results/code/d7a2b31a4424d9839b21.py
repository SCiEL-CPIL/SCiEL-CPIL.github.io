"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import factorial

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better). We minimize this bound.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Problem setup
    m = 12
    n = 1
    
    # We need to find 12 positive roots r_1, ..., r_12
    # The Laguerre polynomial combination has these as double roots.
    
    # The C4 upper bound is related to the last sign change and the root configuration.
    # For the refined Cohn-Goncalves construction, the key quantity involves
    # the product of (1 - r_i) terms or similar expressions.
    
    # Let's define the objective: we want to minimize the upper bound on C4.
    # Based on the structure, a good starting point is to use the roots of 
    # a related orthogonal polynomial or optimize directly.
    
    # The score is seed/value where value is the upper bound.
    # Current seed returns [4, 8, 12, ..., 48] with score 1.0, meaning value = seed_value.
    # We need to find roots that give a smaller value.
    
    # For Laguerre polynomials with double roots, the optimal configuration
    # often relates to the zeros of associated Laguerre polynomials or
    # can be found by minimizing a potential energy functional.
    
    # Let's define the objective function as the C4 upper bound.
    # A common form involves the product of distances or a norm of the polynomial.
    
    # Strategy: Use optimization to minimize the upper bound.
    # The upper bound on C4 for the Fourier sign uncertainty with these roots
    # is typically computed from the Laguerre combination's properties.
    
    # Let's define a surrogate objective that we believe correlates with C4.
    # For double roots r_i, the Laguerre combination L(x) has L(r_i) = L'(r_i) = 0.
    # The C4 bound often involves sum or product of 1/(1-r_i) or similar.
    
    # Let's try: objective = sum of (1 + r_i)^2 / r_i or similar potential.
    # Actually, let's think about what minimizes the uncertainty bound.
    
    # The Fourier sign uncertainty upper bound C4 is related to the L1 norm
    # of the Fourier transform of the sign function, which depends on the
    # root configuration.
    
    # A reasonable objective to minimize is the product of (1 + r_i) / r_i
    # or the sum of r_i, since smaller roots (closer to 0) might give better bounds.
    
    # But roots must be positive and distinct (or can be equal? "double roots" means
    # each r_i is a root of multiplicity 2 in the polynomial).
    
    # Let's use a simple objective: minimize the sum of r_i, subject to r_i > 0.
    # This would push roots toward 0, but we need to check if that's valid.
    
    # Actually, for the Cohn-Goncalves construction, the roots are typically
    # the zeros of a specific polynomial. Let's use the zeros of the Laguerre
    # polynomial L_{12}^{(0)}(x) as a starting point, then optimize.
    
    from numpy.polynomial.laguerre import laggauss
    
    # Get the 12 nodes of Gauss-Laguerre quadrature (zeros of L_{12})
    x_nodes, _ = laggauss(m)
    x_nodes = np.sort(x_nodes)  # Sort in increasing order
    
    # These are positive and distinct. Use as initial guess.
    x0 = x_nodes.copy()
    
    # Define the objective function.
    # The C4 upper bound for the refined inequality likely involves:
    # C4 <= some function of the roots.
    
    # Let's define objective as the product of (1 + r_i) which is a common
    # term in such bounds. Minimizing this would give smaller roots.
    # But we need to ensure the roots are valid for the Laguerre combination.
    
    # Alternative: The bound might be proportional to the norm of the
    # Laguerre combination, which we can compute.
    
    # Let's use a simple, smooth objective that we can optimize:
    # f(r) = sum of r_i^2 / (1 + r_i)
    # This is smooth, positive, and minimized when r_i are small.
    
    def objective(r):
        r = np.maximum(r, 1e-10)  # Ensure positivity
        return np.sum(r**2 / (1 + r))
    
    # Run optimization
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple optimization methods and initial points
    methods = ['Nelder-Mead', 'Powell', 'COBYLA']
    
    for method in methods:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, x0, method=method, 
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Ensure roots are positive and sorted
    best_x = np.maximum(best_x, 1e-6)
    best_x = np.sort(best_x)
    
    # Final check: ensure all roots are positive
    if np.any(best_x <= 0):
        best_x = x0  # Fallback to initial guess
    
    # Convert to list of floats
    result = [float(x) for x in best_x]
    
    return result
# EVOLVE-BLOCK-END
