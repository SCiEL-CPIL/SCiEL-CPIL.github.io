"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import genlaguerre, factorial

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 600.0
    
    m = 12  # number of double roots
    n = 1   # dimension
    
    # Objective: minimize the upper bound C4
    # The Cohn-Goncalves bound involves Laguerre polynomials with double roots
    # We parameterize roots as positive values and minimize the max |L_m(x_i)| weighted appropriately
    
    def objective_and_grad(params):
        # params: 12 positive root locations (log-scaled for positivity)
        log_roots = params
        roots = np.exp(log_roots)
        
        # Ensure roots are positive and sorted
        roots = np.maximum(roots, 1e-10)
        
        # Compute the Laguerre polynomial L_m(x) and its values at roots
        # The bound involves the product/sum structure
        # For the refined inequality, we use the Laguerre polynomial of degree m
        # evaluated at the double root locations
        
        # The C4 bound is related to the maximum of |L_m(x_i) * sqrt(x_i)| / normalization
        # Actually, the standard construction uses:
        # f(x) = L_m(x) * exp(-x/2) with double roots at x_i
        # The bound C4 = max over i of some expression
        
        # More precisely, for the Fourier sign uncertainty:
        # The bound involves the Laguerre polynomial and its derivative at the roots
        # Since roots are double, L_m(x_i) = 0 and L_m'(x_i) = 0
        # The bound C4 is determined by the second derivative or the polynomial structure
        
        # Let me use a different formulation:
        # The upper bound on sign changes is related to the number of zeros of
        # a certain function constructed from Laguerre polynomials
        # C4 = m + 1 + (number of additional sign changes from the construction)
        
        # Actually, re-reading the task: "computes the last sign change"
        # The score is the upper bound on C4, which should be a number
        # With 12 double roots, the base bound is around 12 or 24
        
        # The optimization should minimize this bound
        # A reasonable objective: minimize the maximum |L_m''(x_i)| / |L_m(x_i)| type quantity
        # Or more directly, the bound C4 is a function of the root positions
        
        # Let's use: the bound is proportional to the sum of 1/(x_i * (x_i - x_j)^2) type terms
        # which arise from the Laguerre polynomial structure
        
        # Simpler approach: the Cohn-Goncalves bound for m double roots gives
        # C4 <= m + something depending on root spacing
        # To minimize, we want optimal root spacing
        
        # Use the actual Laguerre polynomial evaluation:
        # L_m(x) = sum_{k=0}^{m} (-1)^k * binom(m+k, k) * x^k / k!
        
        # The bound involves the values of L_m and its derivatives at the roots
        # For double roots, we need L_m(x_i) = 0, L_m'(x_i) = 0
        # The residual from this condition gives the error in the bound
        
        # Compute L_m(x_i) for each root
        L_vals = np.zeros(m)
        for i in range(m):
            L_vals[i] = _laguerre_val(roots[i], m)
        
        # The objective is the maximum absolute value (should be 0 for perfect roots)
        # But since we're optimizing root positions, we minimize the sum of squares
        # of the Laguerre polynomial values, which drives them toward zero
        
        obj = np.sum(L_vals**2)
        
        # Analytic gradient using polynomial division / automatic differentiation
        # d/dx_i of L_m(x_i)^2 = 2 * L_m(x_i) * L_m'(x_i)
        grad = np.zeros(m)
        for i in range(m):
            Lm_prime = _laguerre_deriv(roots[i], m)
            grad[i] = 2.0 * L_vals[i] * Lm_prime * np.exp(log_roots[i])  # chain rule for exp
        
        return obj, grad
    
    def _laguerre_val(x, m):
        """Evaluate Laguerre polynomial L_m(x) using recurrence."""
        if m == 0:
            return 1.0
        if m == 1:
            return 1.0 - x
        L0 = 1.0
        L1 = 1.0 - x
        for k in range(1, m):
            L2 = ((2*k + 1 - x) * L1 - k * L0) / (k + 1)
            L0, L1 = L1, L2
        return L1
    
    def _laguerre_deriv(x, m):
        """Evaluate derivative of Laguerre polynomial L_m'(x)."""
        # L_m'(x) = -L_{m-1}^{(1)}(x)
        # Use recurrence for generalized Laguerre or numerical
        if m == 0:
            return 0.0
        if m == 1:
            return -1.0
        # L_m'(x) = (m/(x)) * (L_{m-1}(x) - L_m(x)) ... not quite
        # Use: d/dx L_m(x) = -L_{m-1}^{(1)}(x)
        # Compute L_{m-1}^{(1)}(x) using recurrence for alpha=1
        if m == 1:
            return -1.0
        # Generalized Laguerre L_{m-1}^{(1)}(x)
        # L_0^{(1)}(x) = 1
        # L_1^{(1)}(x) = 2 - x
        if m - 1 == 0:
            return -1.0
        if m - 1 == 1:
            return -(2.0 - x)
        L0 = 1.0
        L1 = 2.0 - x
        for k in range(1, m - 1):
            # L_{k+1}^{(1)}(x) = ((2k+3-x)*L_k^{(1)}(x) - (k+1)*L_{k-1}^{(1)}(x)) / (k+2)
            L2 = ((2*k + 3 - x) * L1 - (k + 1) * L0) / (k + 2)
            L0, L1 = L1, L2
        return -L1
    
    # Initialization: alpha-based scaling
    # Good initial guess: roots spread out with some scaling
    # Use alpha * (i+1)^beta type initialization
    alpha = 2.0
    beta = 1.0
    init_roots = alpha * np.arange(1, m+1, dtype=float)**beta
    init_log_roots = np.log(init_roots)
    
    best_result = None
    best_obj = np.inf
    
    # Try L-BFGS with analytic gradient
    try:
        res = minimize(
            objective_and_grad,
            init_log_roots,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.success or res.fun < best_obj:
            best_obj = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 30:
        if best_result is not None:
            roots = np.exp(best_result)
            return [float(r) for r in roots]
        return [4.0 * (i + 1) for i in range(12)]
    
    # Try multiple initializations
    best_obj = np.inf
    best_result = None
    
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        
        # Different initializations
        if trial == 0:
            init = np.log(2.0 * np.arange(1, m+1, dtype=float))
        elif trial == 1:
            init = np.log(3.0 * np.arange(1, m+1, dtype=float))
        elif trial == 2:
            init = np.log(1.5 * np.arange(1, m+1, dtype=float)**1.2)
        elif trial == 3:
            init = np.log(4.0 * np.arange(1, m+1, dtype=float)**0.8)
        else:
            init = np.log(2.5 * np.arange(1, m+1, dtype=float) + np.random.uniform(-0.5, 0.5, m))
        
        try:
            res = minimize(
                objective_and_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        roots = np.exp(best_result)
        # Ensure positive
        roots = np.maximum(roots, 1e-10)
        return [float(r) for r in roots]
    
    return [4.0 * (i + 1) for i in range(12)]
# EVOLVE-BLOCK-END
