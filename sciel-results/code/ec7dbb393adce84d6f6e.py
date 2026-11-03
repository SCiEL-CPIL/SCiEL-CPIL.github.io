"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# Precompute Gauss-Laguerre quadrature nodes and weights
_N_QUAD = 400
_x_quad, _w_quad = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_log_norm_and_grad(roots):
    """
    Compute the log of the L2 norm squared of P(x) = prod (x - r_i)^2
    with weight e^{-x}, and its gradient with respect to roots.
    We want to MAXIMIZE this norm.
    
    Returns (log_norm_sq, grad) where grad[i] = d(log_norm_sq)/d(roots[i])
    """
    # log P(x)^2 = 4 * sum_i log|x - r_i|
    # d/d(r_i) log P(x)^2 = -4 / (x - r_i)  (for x > r_i, using |x-r_i|)
    # Actually d/d(r_i) log|x - r_i| = -1/(x - r_i) for x > r_i, +1/(x-r_i) for x < r_i
    # Since we're integrating over x >= 0 and roots are positive, most x > r_i
    
    log_p_sq = np.zeros_like(_x_quad)
    grad_terms = np.zeros((len(roots), len(_x_quad)))
    
    for i, r in enumerate(roots):
        diff = _x_quad - r
        abs_diff = np.abs(diff)
        abs_diff[abs_diff < 1e-300] = 1e-300
        log_p_sq += 4.0 * np.log(abs_diff)
        # d/d(r_i) log|x - r_i| = -sign(x - r_i) / |x - r_i|
        grad_terms[i] = -np.sign(diff) / abs_diff
        grad_terms[i][~np.isfinite(grad_terms[i])] = 0.0
    
    # Integrand: e^{-x} * P(x)^2
    log_integrand = -_x_quad + log_p_sq
    
    max_log = np.max(log_integrand)
    if not np.isfinite(max_log):
        return -np.inf, np.zeros_like(roots)
    
    # Compute integral and gradient
    exp_shifted = np.exp(log_integrand - max_log)
    weights = _w_quad * exp_shifted
    
    sum_val = np.sum(weights)
    if sum_val <= 0 or not np.isfinite(sum_val):
        return -np.inf, np.zeros_like(roots)
    
    log_norm_sq = max_log + np.log(sum_val)
    
    # Gradient: d(log_norm_sq)/d(r_i) = 4 * integral(e^{-x} P(x)^2 * d/d(r_i) log|x-r_i|) / integral(e^{-x} P(x)^2)
    # = 4 * sum_j w_j e^{-x_j} P(x_j)^2 * (-sign(x_j - r_i)/|x_j - r_i|) / sum_j w_j e^{-x_j} P(x_j)^2
    # = 4 * sum_j weights_j * grad_terms[i,j] / sum_val
    # But weights already includes exp(log_integrand - max_log), so:
    # numerator_i = sum_j weights_j * grad_terms[i,j]
    # grad[i] = 4 * numerator_i / sum_val
    
    # Actually, let me recompute more carefully:
    # I = integral e^{-x} P(x)^2 dx = sum_j w_j e^{-x_j} P(x_j)^2
    # dI/dr_i = integral e^{-x} P(x)^2 * 4 * d/d(r_i) log|x-r_i| dx
    #         = sum_j w_j e^{-x_j} P(x_j)^2 * 4 * (-sign(x_j-r_i)/|x_j-r_i|)
    # d(log I)/dr_i = (dI/dr_i) / I
    
    # Using our shifted representation:
    # I = exp(max_log) * sum_val
    # dI/dr_i = exp(max_log) * sum_j weights_j * 4 * grad_terms[i,j]
    # d(log I)/dr_i = 4 * sum_j weights_j * grad_terms[i,j] / sum_val
    
    grad = np.zeros_like(roots)
    for i in range(len(roots)):
        numerator = np.sum(weights * grad_terms[i])
        grad[i] = 4.0 * numerator / sum_val
    
    return log_norm_sq, grad

def _compute_log_norm(roots):
    """Compute only the log norm (faster when gradient not needed)."""
    log_norm_sq, _ = _compute_log_norm_and_grad(roots)
    return log_norm_sq

def _roots_from_geometric(mu, sigma):
    """Generate 12 roots from geometric progression in log-space."""
    i = np.arange(12)
    return np.exp(mu + sigma * i)

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction (m=12, n=1).
    """
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_roots = None
    best_log_norm = -np.inf
    
    def objective_geometric(params):
        """Objective function for geometric parameterization (minimize negative log norm)."""
        if time.time() > deadline - 10:
            return 1e10
        mu, sigma = params
        roots = _roots_from_geometric(mu, sigma)
        if not np.all(np.isfinite(roots)) or np.any(roots <= 0) or np.any(roots > 5000):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    # Phase 1: Global search over (mu, sigma) with wider bounds
    bounds = [(-1.0, 6.0), (-1.0, 1.5)]
    
    try:
        de_result = differential_evolution(
            objective_geometric, 
            bounds, 
            seed=42, 
            maxiter=300, 
            tol=1e-10,
            popsize=20,
            init='sobol',
            polish=False
        )
        
        if de_result.fun < 1e9:
            mu_opt, sigma_opt = de_result.x
            best_roots = _roots_from_geometric(mu_opt, sigma_opt)
            best_log_norm = _compute_log_norm(best_roots)
            
    except Exception:
        pass
    
    # Phase 2: Local refinement on 12 independent roots using gradient
    if best_roots is None:
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        best_log_norm = _compute_log_norm(best_roots)
    
    def objective_full(params):
        """Objective function for full 12-dimensional root optimization."""
        if time.time() > deadline - 5:
            return 1e10
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 5000):
            return 1e10
        log_norm = _compute_log_norm(roots)
        if not np.isfinite(log_norm):
            return 1e10
        return -log_norm
    
    def gradient_full(params):
        """Gradient for full 12-dimensional root optimization."""
        if time.time() > deadline - 5:
            return np.zeros(12)
        roots = np.sort(params)
        if np.any(roots <= 1e-6) or np.any(roots > 5000):
            return np.zeros(12)
        _, grad = _compute_log_norm_and_grad(roots)
        # grad[i] is d(log_norm)/d(roots[i]), we want d(-log_norm)/d(params[i])
        # Since roots = sort(params), the mapping is complicated, so we just negate
        return -grad
    
    try:
        result = minimize(
            objective_full, 
            best_roots, 
            jac=gradient_full,
            method='L-BFGS-B',
            bounds=[(1e-6, 5000)] * 12,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.fun < -best_log_norm:
            best_roots = np.sort(result.x)
            best_log_norm = _compute_log_norm(best_roots)
            
    except Exception:
        pass
    
    # Phase 3: High-precision polish with tighter tolerances
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                objective_full, 
                best_roots, 
                jac=gradient_full,
                method='L-BFGS-B',
                bounds=[(1e-6, 5000)] * 12,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 50}
            )
            
            if result2.fun < -best_log_norm:
                best_roots = np.sort(result2.x)
                best_log_norm = _compute_log_norm(best_roots)
                
        except Exception:
            pass
    
    # Final check
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0) or not np.all(np.isfinite(best_roots)):
        best_roots = np.array([4.0 * i for i in range(1, 13)])
        
    return best_roots.tolist()
# EVOLVE-BLOCK-END
