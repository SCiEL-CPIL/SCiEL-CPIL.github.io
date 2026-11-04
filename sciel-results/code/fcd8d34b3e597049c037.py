"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600.0  # seconds
    
    m = 12  # number of double roots
    n = 1   # dimension
    
    def compute_C4(roots):
        """Compute the C4 upper bound given 12 positive double roots."""
        roots = np.asarray(roots, dtype=float)
        if np.any(roots <= 0):
            return 1e10
        if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        
        # Sort roots
        roots = np.sort(roots)
        
        # P(x) = prod (x - r_i)^2, degree 2m = 24
        # Compute using log-sum for stability
        # C4 bound: use Laguerre-weighted moment ratio
        # ||x * P||^2 / ||P||^2 with weight e^{-x}
        
        # Use Gauss-Laguerre quadrature for integration
        # For degree 24 polynomial squared = degree 48, need enough nodes
        x_nodes, x_weights = np.polynomial.laguerre.laggauss(80)
        
        # Compute P(x) at quadrature nodes
        # P(x) = prod_i (x - r_i)^2
        # log|P(x)| = 2 * sum_i log|x - r_i|
        # But we need P(x)^2 for the norm, so |P(x)|^2 = prod_i (x - r_i)^4
        
        # Compute |P(x)|^2 * e^{-x} = prod_i (x - r_i)^4 * e^{-x}
        # log of this = 4 * sum_i log|x - r_i| - x
        
        log_abs_P2 = 4.0 * np.sum(np.log(np.abs(x_nodes[:, None] - roots[None, :])), axis=1) - x_nodes
        
        # Handle overflow/underflow
        max_log = np.max(log_abs_P2)
        if max_log > 700:
            log_abs_P2 = log_abs_P2 - max_log
        
        abs_P2 = np.exp(log_abs_P2)
        
        # ||P||^2 = sum w_j * |P(x_j)|^2
        norm_P2 = np.sum(x_weights * abs_P2)
        
        # ||x*P||^2 = sum w_j * x_j * |P(x_j)|^2
        norm_xP2 = np.sum(x_weights * x_nodes * abs_P2)
        
        if norm_P2 <= 0:
            return 1e10
        
        # C4 bound: ratio of moments
        C4 = norm_xP2 / norm_P2
        
        return C4
    
    def objective(params):
        """Objective function: C4 in log-space."""
        roots = np.exp(params)
        return compute_C4(roots)
    
    # Initial guess: spread out roots
    # Seed was 4, 8, 12, ..., 48
    initial_roots = np.array([4.0 * (i + 1) for i in range(m)])
    initial_params = np.log(initial_roots)
    
    best_roots = initial_roots.copy()
    best_C4 = compute_C4(initial_roots)
    
    elapsed = time.time() - start_time
    
    # Try L-BFGS-B first (fast)
    if elapsed < deadline - 60:
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000}
            )
            elapsed = time.time() - start_time
            if result.fun < best_C4:
                best_C4 = result.fun
                best_roots = np.exp(result.x)
        except Exception:
            pass
    
    # Try differential evolution (global, slower)
    if elapsed < deadline - 30:
        try:
            # Bounds in log-space: roots from 0.1 to 100
            bounds = [(np.log(0.1), np.log(100.0))] * m
            
            def de_objective(params):
                return compute_C4(np.exp(params))
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                polish=True,
                update_interval=1
            )
            elapsed = time.time() - start_time
            if de_result.fun < best_C4:
                best_C4 = de_result.fun
                best_roots = np.exp(de_result.x)
        except Exception:
            pass
    
    # Final polish with L-BFGS-B from best found
    if elapsed < deadline - 10:
        try:
            final_params = np.log(best_roots)
            result = minimize(
                objective,
                final_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000}
            )
            elapsed = time.time() - start_time
            if result.fun < best_C4:
                best_C4 = result.fun
                best_roots = np.exp(result.x)
        except Exception:
            pass
    
    # Ensure roots are positive and sorted
    best_roots = np.sort(best_roots)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
