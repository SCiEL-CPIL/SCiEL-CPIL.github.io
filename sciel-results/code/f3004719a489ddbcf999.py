"""Rewritten program for the suite task; optimized quadrature rule."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    return abs(approx - exact) / exact * 100.0

def _worst_error(nodes, weights, z_samples, models):
    """Compute worst-case relative error across all samples."""
    max_err = 0.0
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline (5s margin before 90s kill)
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
    ]
    
    # Z samples for evaluation
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    
    # Start with Gauss-Legendre nodes as initial guess
    best_nodes = np.polynomial.legendre.leggauss(n)[0]
    best_weights = np.polynomial.legendre.leggauss(n)[1] / 2.0  # Normalize to sum to 1
    
    # Ensure nodes are in [0,1] and strictly increasing
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    best_score = _worst_error(best_nodes, best_weights, z_samples, models)
    
    # Optimize both nodes and weights simultaneously using smooth min-max
    # We'll use a soft-min/soft-max approach via log-sum-exp for the objective
    
    def objective(params):
        """Objective function: minimize worst-case relative error."""
        # Extract nodes and weights from params
        log_nodes = params[:n]
        log_weights = params[n:2*n]
        
        # Convert from log-space to ensure positivity
        nodes = np.exp(log_nodes)
        weights = np.exp(log_weights)
        
        # Check if nodes are in [0,1] and strictly increasing
        if nodes[0] < 1e-6 or nodes[-1] > 1 - 1e-6:
            return 1e10
        if np.any(np.diff(nodes) < 1e-8):
            return 1e10
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Compute worst-case error using soft-max for smoothness
        errors = []
        for z in z_samples:
            for wa, wb, w0 in models:
                err = _relative_error(z, nodes, weights, wa, wb, w0)
                errors.append(err)
        
        errors = np.array(errors)
        
        # Use soft-max to get a smooth approximation of max error
        beta = 50.0  # Temperature parameter for soft-max
        max_err = np.max(errors)
        soft_max = max_err + np.log(np.sum(np.exp((errors - max_err) / beta))) / beta
        
        return soft_max
    
    # Initial parameters in log-space
    log_nodes_init = np.log(best_nodes)
    log_weights_init = np.log(best_weights)
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Optimize using Nelder-Mead
    try:
        result = minimize(
            objective,
            params_init,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-10,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.fun < best_score:
            log_nodes_opt = result.x[:n]
            log_weights_opt = result.x[n:2*n]
            
            nodes_opt = np.exp(log_nodes_opt)
            weights_opt = np.exp(log_weights_opt)
            
            # Validate and normalize
            if nodes_opt[0] > 1e-6 and nodes_opt[-1] < 1 - 1e-6 and np.all(np.diff(nodes_opt) > 1e-8):
                weights_opt = weights_opt / np.sum(weights_opt)
                score_opt = _worst_error(nodes_opt, weights_opt, z_samples, models)
                if score_opt < best_score:
                    best_nodes = nodes_opt
                    best_weights = weights_opt
                    best_score = score_opt
    except Exception as e:
        pass
    
    # Polish pass: fine-tune weights for the best nodes found
    if time.time() < deadline - 5.0:
        try:
            # Optimize weights only for fixed nodes
            def weight_objective(log_weights):
                weights = np.exp(log_weights)
                weights = weights / np.sum(weights)
                return _worst_error(best_nodes, weights, z_samples, models)
            
            log_weights_init = np.log(best_weights)
            result_w = minimize(
                weight_objective,
                log_weights_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 100,
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if result_w.fun < best_score:
                log_weights_opt = result_w.x
                weights_opt = np.exp(log_weights_opt)
                weights_opt = weights_opt / np.sum(weights_opt)
                score_opt = _worst_error(best_nodes, weights_opt, z_samples, models)
                if score_opt < best_score:
                    best_weights = weights_opt
                    best_score = score_opt
        except Exception as e:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
