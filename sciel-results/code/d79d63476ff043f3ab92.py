"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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

def _gauss_legendre_nodes(n):
    """Generate Gauss-Legendre nodes and weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    # Map from [-1, 1] to [0, 1]
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    # Sort in increasing order
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    # Normalize weights to sum to 1
    w = w / np.sum(w)
    return x, w

def _log_param_to_nodes(log_nodes):
    """Convert log-parameterized nodes to actual nodes in (0,1)."""
    # Use tanh to map from real line to (0,1)
    nodes = 0.5 * (1.0 + np.tanh(log_nodes))
    # Ensure strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    # Ensure strictly increasing
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1] + 1e-12:
            nodes[i] = nodes[i-1] + 1e-10
    return nodes

def _log_param_to_weights(log_weights):
    """Convert log-parameterized weights to positive weights summing to 1."""
    # Use softmax
    log_w = log_weights - np.max(log_weights)
    w = np.exp(log_w)
    w = w / np.sum(w)
    return w

def _objective(params, z_samples, models):
    """Objective: worst-case relative error."""
    n = len(params) // 2
    log_nodes = params[:n]
    log_weights = params[n:]
    
    nodes = _log_param_to_nodes(log_nodes)
    weights = _log_param_to_weights(log_weights)
    
    max_err = 0.0
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline (5s margin)
    
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
    
    # Start with Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = _gauss_legendre_nodes(n)
    
    # Convert to log-parameterization
    # Inverse of tanh: atanh(2*x - 1)
    log_nodes_init = np.arctanh(2.0 * gl_nodes - 1.0)
    # Inverse of softmax: just use log of weights (up to constant)
    log_weights_init = np.log(gl_weights)
    
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Evaluate initial score
    best_params = params_init.copy()
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # L-BFGS-B optimization
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective, params_init,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception as e:
            pass
    
    # Differential evolution with multiple restarts
    remaining = deadline - time.time()
    if remaining > 30:
        bounds = [(-5, 5)] * n + [(-5, 5)] * n
        
        def de_objective(params):
            return _objective(params, z_samples, models)
        
        # Multiple restarts
        n_restarts = 3
        time_per_restart = remaining / n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                result = differential_evolution(
                    de_objective, bounds,
                    maxiter=50, popsize=15,
                    tol=1e-6, seed=42 + r,
                    polish=True,
                    workers=1,
                    time_limit=time_per_restart - 2
                )
                
                if result.fun < best_score:
                    best_params = result.x
                    best_nodes = _log_param_to_nodes(result.x[:n])
                    best_weights = _log_param_to_weights(result.x[n:])
                    best_score = result.fun
            except Exception as e:
                pass
    
    # Final refinement with L-BFGS-B
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective, best_params,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception as e:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
