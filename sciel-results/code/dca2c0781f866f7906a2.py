import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_genlaguerre, roots_laguerre
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin before 600s limit

    # Strategy:
    # 1. Use generalized Laguerre roots L_12^(alpha) as a structural representation.
    #    This captures the "spread vs decay" balance better than uniform spacing.
    # 2. Optimize the parameter alpha (1D search) to find the best family of roots.
    # 3. Refine the specific root locations (12D optimization) starting from the best alpha family.
    # 4. Use the refined roots as the final output.

    # Precompute factorials for the objective function
    fact = np.ones(49)
    for i in range(1, 49):
        fact[i] = fact[i-1] * i

    def objective(params):
        # params are the 12 roots
        if not np.all(params > 0):
            return 1e10
        # Sort roots for stability (though poly doesn't care, it helps with convergence)
        roots = np.sort(params)
        
        # Construct polynomial Q(x) = prod (x - r_i)
        # Coefficients in descending power order
        q_coeffs = np.poly(roots)
        
        # Compute Q(x)^4
        q2 = np.convolve(q_coeffs, q_coeffs)
        q4 = np.convolve(q2, q2)
        
        # Integral of e^-x * Q(x)^4 from 0 to inf
        # Sum c_k * k! where c_k is coeff of x^k
        # q4 is descending: q4[0] is x^48, ..., q4[48] is x^0
        powers = np.arange(48, -1, -1)
        return np.sum(q4 * fact[powers])

    best_x = None
    best_val = np.inf

    # Phase 1: 1D optimization over alpha
    # Roots of L_12^(alpha) are positive for alpha > -1
    # We expect alpha to be in a reasonable range, e.g., [-1, 10]
    
    def score_alpha(alpha):
        try:
            if alpha <= -1:
                return 1e10
            roots, _ = roots_genlaguerre(12, alpha)
            if not np.all(roots > 0):
                return 1e10
            return objective(roots)
        except Exception:
            return 1e10

    # Scan a range of alpha values to find a good starting point
    # L_12^0 roots are around 1 to 30.
    # Let's try alpha from -0.5 to 5.0
    alphas = np.linspace(-0.5, 5.0, 100)
    best_alpha = 0.0
    best_alpha_val = np.inf
    
    for a in alphas:
        val = score_alpha(a)
        if val < best_alpha_val:
            best_alpha_val = val
            best_alpha = a

    # Refine alpha using minimize_scalar
    from scipy.optimize import minimize_scalar
    try:
        res_alpha = minimize_scalar(score_alpha, bounds=(best_alpha - 1.0, best_alpha + 1.0), method='bounded')
        if res_alpha.success and res_alpha.fun < best_alpha_val:
            best_alpha_val = res_alpha.fun
            best_alpha = res_alpha.x
    except Exception:
        pass

    # Get the roots for the best alpha
    try:
        roots_alpha, _ = roots_genlaguerre(12, best_alpha)
        if np.all(roots_alpha > 0):
            best_x = roots_alpha
            best_val = best_alpha_val
    except Exception:
        # Fallback to standard Laguerre roots
        roots_lag, _ = roots_laguerre(12)
        best_x = roots_lag
        best_val = objective(best_x)

    # Check time
    if time.time() > deadline - 20:
        return [float(x) for x in np.sort(best_x)]

    # Phase 2: 12D optimization to refine the specific root locations
    # Start from the best Laguerre-family roots
    
    # Try Nelder-Mead first for local refinement
    try:
        res_nm = minimize(objective, best_x, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res_nm.fun < best_val:
            best_x = res_nm.x
            best_val = res_nm.fun
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 10:
        best_x = np.maximum(best_x, 1e-6)
        return [float(x) for x in np.sort(best_x)]

    # Try L-BFGS-B for further refinement with bounds
    bounds = [(1e-6, None)] * 12
    try:
        res_lbfgs = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 1000})
        if res_lbfgs.fun < best_val:
            best_x = res_lbfgs.x
            best_val = res_lbfgs.fun
    except Exception:
        pass

    # Check time
    if time.time() > deadline - 5:
        best_x = np.maximum(best_x, 1e-6)
        return [float(x) for x in np.sort(best_x)]

    # Optional: Try Differential Evolution if time permits? 
    # DE is stochastic and might find a better global minimum, but it's slow.
    # Given the structure, the Laguerre roots are likely close to optimal.
    # Let's stick to the local refinement results to ensure speed and reliability.
    
    # Ensure positive and sorted
    best_x = np.maximum(best_x, 1e-6)
    best_x = np.sort(best_x)
    
    return [float(x) for x in best_x]
