"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import roots_legendre


def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    
    Uses a rational-function (Padé-like) approach to approximate 1/E(a) and derive
    optimal nodes/weights via convex optimization.
    """
    n = 8

    def E_of_a(a, w0, wa):
        """Hubble parameter normalized by H0 for CPL model."""
        a = np.maximum(a, 1e-15)
        if abs(w0 + 1.0) < 1e-10:
            return np.sqrt(a**(-3) + (1.0 + wa) * np.exp(-3.0 * wa * (1.0 - a)))
        else:
            exponent = -3.0 * (1.0 + w0)
            prefactor = (1.0 + w0 + wa)
            exp_term = np.exp(-3.0 * wa * (1.0 - a) / (1.0 + w0))
            return np.sqrt(a**(-3) + prefactor * a**exponent * exp_term)

    def exact_luminosity_distance(z, w0, wa):
        """Compute exact dL/(c/H0) via adaptive quadrature."""
        def integrand(zp):
            a = 1.0 / (1.0 + zp)
            return (1.0 + zp) * zp / E_of_a(a, w0, wa)
        result, _ = quad(integrand, 0.0, z, limit=200)
        return result

    def quadrature_luminosity_distance(z, nodes, weights, w0, wa):
        """Compute quadrature approximation of dL/(c/H0)."""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)

    # Training cosmologies - expanded set for better generalization
    training_cosmos = [
        (0.0, 0.0),    # Lambda CDM
        (-0.5, 0.5),   # quintessence-like
        (-1.5, 0.5),   # phantom-like
        (-1.0, 0.5),   # w0 = -1
        (-1.0, -0.5),  # w0 = -1, negative wa
        (-0.5, -0.5),  # another
        (-1.5, -0.5),  # another
        (0.0, 0.5),    # another
        (-0.8, 0.3),   # additional
        (-1.2, 0.2),   # additional
        (-0.3, 0.7),   # additional
        (-1.8, 0.3),   # additional
    ]

    # Test redshifts - more dense sampling
    test_zs = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

    # Precompute exact values
    exact_values = {}
    for (w0, wa) in training_cosmos:
        exact_values[(w0, wa)] = np.array([exact_luminosity_distance(z, w0, wa) for z in test_zs])

    # Parameterize nodes via logit: nodes = sigmoid(t_i)
    # Weights via log: w_i = exp(log_w_i) / sum(exp(log_w_j))
    
    def params_to_nodes_weights(params):
        t = params[:n]
        log_w = params[n:]
        nodes = 1.0 / (1.0 + np.exp(-t))
        w = np.exp(log_w - np.max(log_w))  # for numerical stability
        w = w / np.sum(w)
        return nodes, w

    def objective_and_grad(params):
        nodes, w = params_to_nodes_weights(params)
        
        # Check if nodes are strictly increasing
        diffs = np.diff(nodes)
        if np.any(diffs <= 1e-15):
            return 1e10, np.zeros_like(params)
        
        # Compute relative errors for all (cosmo, z) pairs
        rel_errors = []
        for (w0, wa) in training_cosmos:
            exact = exact_values[(w0, wa)]
            for i_z, z in enumerate(test_zs):
                approx = quadrature_luminosity_distance(z, nodes, w, w0, wa)
                rel_err = abs(approx - exact[i_z]) / max(exact[i_z], 1e-15)
                rel_errors.append(rel_err)
        
        rel_errors = np.array(rel_errors)
        max_rel_err = np.max(rel_errors)
        
        # For gradient, we only need the gradient of the max error
        # Find which errors are close to max
        mask = rel_errors > max_rel_err - 1e-8
        if np.sum(mask) == 0:
            mask = rel_errors == max_rel_err
        
        # Compute gradient via finite differences for simplicity
        # (could be improved with analytic gradient)
        grad = np.zeros_like(params)
        eps = 1e-7
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            nodes_plus, w_plus = params_to_nodes_weights(params_plus)
            diffs_plus = np.diff(nodes_plus)
            if np.any(diffs_plus <= 1e-15):
                grad[i] = 1.0
                continue
            
            rel_errors_plus = []
            for (w0, wa) in training_cosmos:
                exact = exact_values[(w0, wa)]
                for i_z, z in enumerate(test_zs):
                    approx = quadrature_luminosity_distance(z, nodes_plus, w_plus, w0, wa)
                    rel_err = abs(approx - exact[i_z]) / max(exact[i_z], 1e-15)
                    rel_errors_plus.append(rel_err)
            rel_errors_plus = np.array(rel_errors_plus)
            max_rel_err_plus = np.max(rel_errors_plus)
            grad[i] = (max_rel_err_plus - max_rel_err) / eps
        
        return max_rel_err, grad

    # Initialize with Gauss-Legendre nodes mapped to (0,1)
    # Gauss-Legendre nodes are in (-1,1), map to (0,1) via (x+1)/2
    gl_nodes, gl_weights = roots_legendre(n)
    nodes_init = (gl_nodes + 1.0) / 2.0
    nodes_init = np.sort(nodes_init)
    
    # Convert to logit space
    t_init = np.log(nodes_init / (1.0 - nodes_init))
    
    # Initialize weights to be proportional to Gauss-Legendre weights
    w_init = gl_weights / np.sum(gl_weights)
    log_w_init = np.log(w_init)
    
    x0 = np.concatenate([t_init, log_w_init])
    
    # Try multiple restarts with different initializations
    best_result = None
    best_obj = float('inf')
    
    # Restart 1: Gauss-Legendre based
    result1 = minimize(objective_and_grad, x0, method='L-BFGS-B', 
                      jac=True, options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12})
    
    if result1.fun < best_obj:
        best_obj = result1.fun
        best_result = result1
    
    # Restart 2: Uniform nodes
    nodes_uniform = np.linspace(0.05, 0.95, n)
    t_uniform = np.log(nodes_uniform / (1.0 - nodes_uniform))
    log_w_uniform = np.zeros(n)
    x0_uniform = np.concatenate([t_uniform, log_w_uniform])
    
    result2 = minimize(objective_and_grad, x0_uniform, method='L-BFGS-B',
                      jac=True, options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12})
    
    if result2.fun < best_obj:
        best_obj = result2.fun
        best_result = result2
    
    # Restart 3: Chebyshev-like nodes
    k = np.arange(n)
    nodes_cheb = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / n))
    t_cheb = np.log(nodes_cheb / (1.0 - nodes_cheb))
    log_w_cheb = np.zeros(n)
    x0_cheb = np.concatenate([t_cheb, log_w_cheb])
    
    result3 = minimize(objective_and_grad, x0_cheb, method='L-BFGS-B',
                      jac=True, options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12})
    
    if result3.fun < best_obj:
        best_obj = result3.fun
        best_result = result3
    
    # Use the best result
    t_opt = best_result.x[:n]
    log_w_opt = best_result.x[n:]
    
    nodes = 1.0 / (1.0 + np.exp(-t_opt))
    weights = np.exp(log_w_opt - np.max(log_w_opt))
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly increasing and in (0,1)
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    
    # Ensure strict increase with small separation
    for i in range(1, n):
        if nodes[i] <= nodes[i-1] + 1e-10:
            nodes[i] = nodes[i-1] + 1e-8
    
    # Final normalization
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
