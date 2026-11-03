<<<<<<< SEARCH
import numpy as np
from numpy.polynomial import laguerre

def construct():
    # Compute the roots of the Laguerre polynomial L_12(x)
    # laggauss returns points and weights for integration, but we need roots of the polynomial
    # We can get the roots using the eigenvalues of the companion matrix or numpy.roots
    # However, numpy.polynomial.laguerre.laggauss gives the Gauss-Laguerre quadrature points,
    # which are the roots of L_m(x) for the standard weight e^-x.
    
    # For m=12, the roots of L_12(x) are the nodes of the Gauss-Laguerre quadrature.
    x, w = laguerre.laggauss(12)
    
    # The roots are positive.
    # Ensure they are sorted (laggauss returns them sorted)
    roots = list(x)
    
    return roots
=======
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial guess: roots of L_12
    x0, _ = laguerre.laggauss(12)
    x0 = np.array(x0, dtype=float)
    
    # Objective function: Minimize the sum of squares of the roots
    # This heuristic assumes that concentrating the roots reduces the "size" 
    # and thus the upper bound on the uncertainty constant.
    def objective(roots):
        # Ensure positivity
        roots = np.maximum(roots, 1e-6)
        return np.sum(roots ** 2)
    
    # Constraints: roots must be positive
    bounds = [(1e-6, None) for _ in range(12)]
    
    best_result = None
    best_score = np.inf
    
    # Run optimization
    try:
        # L-BFGS-B is efficient for this
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-10})
        if res.success:
            best_result = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # If optimization fails or doesn't improve, fallback to initial
    if best_result is None or best_score > np.sum(x0**2):
        best_result = x0
    
    # Check time
    if time.time() > deadline - 10:
        return list(best_result)
        
    # Try a second phase: local refinement with Nelder-Mead around the L-BFGS-B result
    # Nelder-Mead can sometimes find better local minima if the objective is non-smooth (due to clipping)
    try:
        res_nm = minimize(objective, best_result, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res_nm.success and res_nm.fun < best_score:
            best_result = res_nm.x
            best_score = res_nm.fun
    except Exception:
        pass
    
    # Ensure all roots are strictly positive
    best_result = np.maximum(best_result, 1e-6)
    
    return list(best_result)
>>>>>>> REPLACE
