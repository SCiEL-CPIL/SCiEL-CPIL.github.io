import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Define E(a) for CPL dark energy in flat universe
    # w(z) = w0 + w1*z, a = 1/(1+z)
    # E^2(a) = Omega_m/a^3 + Omega_de * a^(-3*(1+w0)) * exp(3*w1*(1-a)/a)
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        # Avoid overflow in exp for large w1 or small a
        exp_term = 3.0 * w1 * (1.0 - a_safe) / a_safe
        # Clip exp_term to prevent overflow
        exp_term = np.clip(exp_term, -500, 500)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(exp_term))

    def _true_distance(z_max, w0, w1):
        """True luminosity distance via adaptive quadrature."""
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) / _E(a, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val

    # Test cosmologies: diverse CPL models
    # (w0, w1) pairs
    test_cosmos = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL w1>0
        (-1.0, -0.5),     # CPL w1<0
        (-0.9, 0.3),      # Quintessence-like
        (-0.95, -0.3),    # Another variant
        (-1.2, 0.0),      # Phantom
        (-0.8, 0.5),      # Strong evolution
        (-1.0, 1.0),      # Strong CPL
        (-1.1, 0.4),      # Additional
        (-0.85, -0.4),    # Additional
        (-0.7, 0.8),      # Very strong evolution
        (-1.3, -0.2),     # Phantom with slight evolution
        (-1.0, 0.0),      # LambdaCDM again for weight? No, unique.
        (-0.95, 0.0),     # Constant w
        (-1.05, 0.0),     # Constant w
    ]
    
    # Redshifts to test
    test_zs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    # Pre-compute true distances for all test cases
    # This is crucial for speed during optimization
    true_distances = {}
    num_cases = 0
    for (w0, w1) in test_cosmos:
        for z in test_zs:
            key = (w0, w1, z)
            if key not in true_distances:
                true_distances[key] = _true_distance(z, w0, w1)
                num_cases += 1
                
    # Helper to compute errors for a given set of nodes and weights
    def _compute_errors(nodes, weights):
        errors = np.zeros(num_cases)
        idx = 0
        for (w0, w1) in test_cosmos:
            for z in test_zs:
                key = (w0, w1, z)
                true_val = true_distances[key]
                
                # Quadrature approximation
                z_nodes = z * nodes
                a_nodes = 1.0 / (1.0 + z_nodes)
                vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
                approx = z * np.sum(weights * vals)
                
                rel_err = abs(approx - true_val) / true_val
                errors[idx] = rel_err
                idx += 1
        return errors

    # Smooth worst-case error using log-sum-exp
    def _smooth_worst_error(nodes, weights, tau):
        # Check constraints
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        weights = weights / np.sum(weights)
        
        errors = _compute_errors(nodes, weights)
        
        # Log-sum-exp trick for numerical stability
        max_err = np.max(errors)
        if max_err > 100: # If errors are huge, just return max
            return max_err
            
        # J = tau * log(sum(exp((E - max)/tau))) + max
        scaled_errors = (errors - max_err) / tau
        # Clip to avoid underflow in exp (not strictly necessary for sum, but good practice)
        # exp(-large) -> 0
        sum_exp = np.sum(np.exp(scaled_errors))
        
        if sum_exp == 0:
            return max_err
            
        return max_err + tau * np.log(sum_exp)

    # Start with Gauss-Legendre as initial guess
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    
    # Calculate initial error with hard max for reporting
    init_errors = _compute_errors(best_nodes, best_weights)
    best_err = np.max(init_errors)
    
    # Objective function wrapper for optimizer
    # We optimize nodes directly and log-weights
    def _objective(params, tau):
        if time.time() > deadline:
            return 1e10
            
        nodes = params[:8]
        log_weights = params[8:]
        
        # Softmax for weights
        # Subtract max for stability
        max_lw = np.max(log_weights)
        weights = np.exp(log_weights - max_lw)
        weights = weights / np.sum(weights)
        
        return _smooth_worst_error(nodes, weights, tau)

    # Initial parameters
    x0 = np.concatenate([nodes_init, np.log(weights_init)])
    
    # Stage 1: Coarse optimization with larger tau
    tau1 = 0.1
    try:
        result1 = minimize(_objective, x0, args=(tau1,), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        if time.time() < deadline - 10 and result1.fun < 1e9:
            nodes_c = result1.x[:8]
            log_w_c = result1.x[8:]
            max_lw = np.max(log_w_c)
            weights_c = np.exp(log_w_c - max_lw)
            weights_c = weights_c / np.sum(weights_c)
            
            # Verify constraints
            if np.all(nodes_c[1:] > nodes_c[:-1]) and np.all(nodes_c > 0) and np.all(nodes_c < 1):
                err_c = np.max(_compute_errors(nodes_c, weights_c))
                if err_c < best_err:
                    best_nodes = nodes_c.copy()
                    best_weights = weights_c.copy()
                    best_err = err_c
    except Exception as e:
        pass

    # Stage 2: Fine optimization with smaller tau
    if time.time() < deadline - 5:
        tau2 = 0.01
        x0 = np.concatenate([best_nodes, np.log(best_weights)])
        try:
            result2 = minimize(_objective, x0, args=(tau2,), method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            if time.time() < deadline - 2 and result2.fun < 1e9:
                nodes_f = result2.x[:8]
                log_w_f = result2.x[8:]
                max_lw = np.max(log_w_f)
                weights_f = np.exp(log_w_f - max_lw)
                weights_f = weights_f / np.sum(weights_f)
                
                if np.all(nodes_f[1:] > nodes_f[:-1]) and np.all(nodes_f > 0) and np.all(nodes_f < 1):
                    err_f = np.max(_compute_errors(nodes_f, weights_f))
                    if err_f < best_err:
                        best_nodes = nodes_f.copy()
                        best_weights = weights_f.copy()
                        best_err = err_f
        except Exception:
            pass

    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    # Ensure strict increase
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
