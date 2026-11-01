import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

START_TIME = time.time()
TIME_LIMIT = 195.0  # Leave 15s safety margin for 210s total

def construct():
    """
    Finds coefficients c0, c4, c8 for the Hermite combination
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    such that g(0)=0 and the upper bound on A(g)*A(hat{g}) is minimized.
    Returns np.array([c0, c4, c8]).
    """
    
    # Precompute Gauss-Hermite nodes and weights for fast integration
    N = 300
    u, w = np.polynomial.hermite.hermgauss(N)
    sqrt2 = np.sqrt(2.0)
    u_scaled = u * sqrt2
    u4 = u**4
    
    # Precompute Hermite polynomials at nodes for speed
    x2 = u_scaled * u_scaled
    x4 = x2 * x2
    x6 = x4 * x2
    x8 = x4 * x4
    x10 = x8 * x2
    x12 = x6 * x6
    
    H0_nodes = np.ones_like(u_scaled)
    H4_nodes = x4 - 6.0 * x2 + 3.0
    H8_nodes = x8 - 28.0 * x6 + 210.0 * x4 - 420.0 * x2 + 105.0
    H12_nodes = x12 - 84.0 * x10 + 2520.0 * x8 - 30240.0 * x6 + 151200.0 * x4 - 221760.0 * x2 + 10395.0
    
    # Weight vector for the integral
    w_u4 = w * u4 * 2.0 * sqrt2
    
    def get_c12(c0, c4, c8):
        return -(c0 + 3.0*c4 + 105.0*c8) / 945.0
    
    def compute_p_at_nodes(c0, c4, c8, c12):
        return c0 * H0_nodes + c4 * H4_nodes + c8 * H8_nodes + c12 * H12_nodes
    
    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        p_vals = compute_p_at_nodes(c0, c4, c8, c12)
        abs_p = np.abs(p_vals)
        
        integral_val = np.sum(w_u4 * abs_p)
        
        return 2.0 * integral_val
    
    def objective_grad(coeffs):
        """
        Compute objective and gradient using subgradient of abs.
        For abs(x), the subgradient is sign(x), with 0 at x=0.
        """
        c0, c4, c8 = coeffs
        c12 = get_c12(c0, c4, c8)
        
        p_vals = compute_p_at_nodes(c0, c4, c8, c12)
        abs_p = np.abs(p_vals)
        
        integral_val = np.sum(w_u4 * abs_p)
        obj = 2.0 * integral_val
        
        # Subgradient: sign(p) for p != 0, 0 for p == 0
        signs = np.sign(p_vals)
        
        # Gradient with respect to c0, c4, c8
        # d/dc0 = 2 * sum(w_u4 * signs * H0)
        # d/dc4 = 2 * sum(w_u4 * signs * H4)
        # d/dc8 = 2 * sum(w_u4 * signs * H8)
        # Note: c12 depends on c0, c4, c8, but since we're using subgradient
        # and the objective is convex in the coefficients (sum of abs of linear functions),
        # we can compute the subgradient directly.
        
        grad0 = 2.0 * np.sum(w_u4 * signs * H0_nodes)
        grad4 = 2.0 * np.sum(w_u4 * signs * H4_nodes)
        grad8 = 2.0 * np.sum(w_u4 * signs * H8_nodes)
        
        return obj, np.array([grad0, grad4, grad8])
    
    # Seed coefficients
    best_score = np.inf
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    seed_score = objective(best_coeffs)
    if seed_score < best_score:
        best_score = seed_score
        best_coeffs = best_coeffs.copy()
    
    np.random.seed(42)
    
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # Phase 1: Differential evolution for global search
    if time.time() < TIME_LIMIT - 30:
        try:
            de_res = differential_evolution(
                objective, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='sobol',
                updating='deferred'
            )
            if de_res.fun < best_score:
                best_score = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead for local refinement
    num_starts = 50
    for i in range(num_starts):
        if time.time() > TIME_LIMIT - 15:
            break
            
        if i == 0:
            x0 = best_coeffs.copy()
        else:
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1])
            ])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B polish with subgradient
    if time.time() < TIME_LIMIT - 10:
        try:
            res2 = minimize(objective, best_coeffs, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeffs = res2.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish with very tight tolerances
    if time.time() < TIME_LIMIT - 5:
        try:
            res3 = minimize(objective, best_coeffs, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            
            if res3.fun < best_score:
                best_score = res3.fun
                best_coeffs = res3.x.copy()
        except Exception:
            pass
    
    return best_coeffs
