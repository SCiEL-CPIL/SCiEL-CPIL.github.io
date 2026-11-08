"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_genlaguerre
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find 12 positive double-root locations for the Laguerre combination (m=12, n=1)
    that minimize the upper bound on C4.
    Returns: list of 12 positive double-root locations
    """
    start_time = time.time()
    deadline = 630.0 - 30.0  # 30s safety margin
    
    m = 12
    n = 1
    
    # Degree of polynomial with m double roots is 2m = 24
    deg = 2 * m
    
    # Gauss-Laguerre quadrature nodes and weights for computing Laguerre coefficients
    # We need enough nodes for degree 24 polynomial times Laguerre polynomials
    n_quad = 100
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    def compute_laguerre_coeffs(roots):
        """Compute Laguerre expansion coefficients of P(x) = prod(x - r_i)^2."""
        # Build polynomial coefficients (in standard power basis)
        # P(x) = prod_i (x - r_i)^2
        poly = np.array([1.0])
        for r in roots:
            # (x - r)^2 = x^2 - 2r*x + r^2
            quad_poly = np.array([1.0, -2.0 * r, r * r])
            poly = np.convolve(poly, quad_poly)
        
        # Evaluate P at quadrature nodes
        P_vals = np.polyval(poly, x_quad)
        
        # Compute Laguerre coefficients: c_k = integral P(x) L_k(x) e^{-x} dx / k!
        # Using quadrature: sum_j w_j P(x_j) L_k(x_j) / k!
        coeffs = np.zeros(deg + 1)
        for k in range(deg + 1):
            L_k_vals = eval_genlaguerre(k, 0, x_quad)  # Standard Laguerre L_k(x)
            coeffs[k] = np.sum(w_quad * P_vals * L_k_vals) / np.math.factorial(k)
        
        return coeffs
    
    def compute_c4_bound(roots):
        """Compute the upper bound on C4 from the Laguerre coefficients."""
        coeffs = compute_laguerre_coeffs(roots)
        
        # The Cohn-Goncalves bound for C4 with n=1 typically involves
        # the norm of the polynomial in the Laguerre space.
        # For the refined construction, the bound is related to the 
        # weighted sum of squared coefficients.
        
        # The L2 norm in Laguerre space: ||P||^2 = sum_k |c_k|^2 * k!
        # But the C4 bound specifically involves certain coefficient combinations.
        
        # For the standard Cohn-Goncalves bound on the 4th moment/C4,
        # with n=1 (first moment constraint), the bound is typically:
        # C4_bound = sum_{k=0}^{deg} |c_k|^2 * k! * (some weight depending on n)
        
        # A common form: the bound equals the squared norm of the projection
        # onto the orthogonal complement of the first n Laguerre polynomials.
        
        # With n=1, we project out L_0 (and possibly L_1 depending on convention).
        # Let's use: bound = sum_{k=1}^{deg} |c_k|^2 * k!
        # (excluding k=0 which corresponds to the mass/zeroth moment)
        
        # Actually, for the Fourier sign uncertainty, the relevant bound involves
        # the ratio or specific coefficient combinations. Let me use the full norm
        # as a reasonable proxy that should correlate with the true C4 bound.
        
        # The most standard: C4 upper bound = ||P||_{L2(e^{-x})}^2 = sum_k |c_k|^2 * k!
        norm_sq = np.sum(np.abs(coeffs) ** 2 * np.array([np.math.factorial(k) for k in range(deg + 1)]))
        
        return norm_sq
    
    # Initial guess: spread roots logarithmically or use known optimal patterns
    # For Laguerre-type problems, roots often cluster near certain values
    # Start with evenly spaced roots in a reasonable range
    roots_init = np.linspace(0.5, 10.0, m)
    
    best_roots = roots_init.copy()
    best_bound = compute_c4_bound(roots_init)
    
    # Try multiple optimization strategies
    strategies = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}),
        ('Nelder-Mead', {'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}),
    ]
    
    for method, opts in strategies:
        if time.time() > deadline:
            break
        
        try:
            # Convert to log-space for positivity constraint
            def objective(log_roots):
                roots = np.exp(log_roots)
                # Check for NaN or inf
                if not np.all(np.isfinite(roots)):
                    return 1e10
                return compute_c4_bound(roots)
            
            log_roots_init = np.log(roots_init)
            
            result = minimize(objective, log_roots_init, method=method, options=opts)
            
            if result.success or result.fun < best_bound:
                new_roots = np.exp(result.x)
                if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    new_bound = result.fun
                    if new_bound < best_bound:
                        best_bound = new_bound
                        best_roots = new_roots.copy()
        except Exception:
            pass
    
    # Refine with more iterations if time permits
    if time.time() < deadline - 60:
        try:
            def objective(log_roots):
                roots = np.exp(log_roots)
                if not np.all(np.isfinite(roots)):
                    return 1e10
                return compute_c4_bound(roots)
            
            log_roots_init = np.log(best_roots)
            result = minimize(objective, log_roots_init, method='L-BFGS-B', 
                            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_bound:
                new_roots = np.exp(result.x)
                if np.all(new_roots > 0) and np.all(np.isfinite(new_roots)):
                    best_roots = new_roots.copy()
                    best_bound = result.fun
        except Exception:
            pass
    
    # Ensure all roots are positive and finite
    best_roots = np.maximum(best_roots, 1e-10)
    
    return list(best_roots)
# EVOLVE-BLOCK-END

