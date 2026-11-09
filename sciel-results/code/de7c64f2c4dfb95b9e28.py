import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Representative cosmologies: (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (w1 positive)
        (0.3, -1.0, -0.5),  # Evolving DE (w1 negative)
        (0.2, -1.0, 0.0),   # Low matter density
        (0.4, -1.0, 0.0),   # High matter density
        (0.3, -1.5, 0.3),   # Strongly evolving
        (0.3, -0.5, -0.3),  # Mild quintessence with evolution
        (0.35, -1.1, 0.2),  # Slightly matter-dominated
    ]
    
    # Test redshifts
    z_test = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    
    def E_func(z, Om, w0, w1):
        OL = 1.0 - Om
        # E^2 = Om(1+z)^3 + OL(1+z)^(3(1+w0+w1*z))
        # Handle z=0: (1+0)=1, so 1^anything = 1
        return np.sqrt(Om * (1.0 + z)**3 + OL * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Om, w0, w1):
        """Compute the true integral using adaptive quadrature."""
        def integrand(u):
            return 1.0 / E_func(z * u, Om, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100)
        return res
    
    # Precompute true integrals for all test cases
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
    
    # Check time after precomputation
    if time.time() > deadline:
        # Fallback to Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}
    
    def _nodes_from_params(node_params):
        """Map unconstrained params to strictly increasing nodes in (0,1) via logit differences."""
        # Use differences: node[0] = sigmoid(p0), node[i] = node[i-1] + sigmoid(p_i) * (1 - node[i-1])
        # This guarantees 0 < n1 < n2 < ... < n8 < 1
        nodes = np.zeros(8)
        nodes[0] = 1.0 / (1.0 + np.exp(-node_params[0]))
        for i in range(1, 8):
            step = 1.0 / (1.0 + np.exp(-node_params[i]))
            nodes[i] = nodes[i-1] + step * (1.0 - nodes[i-1])
        return nodes
    
    def objective(params):
        """
        params: 16 floats
        first 8: node parameters (unconstrained)
        last 8: weight parameters (unconstrained, mapped via softmax)
        """
        node_params = params[:8]
        weight_params = params[8:]
        
        nodes = _nodes_from_params(node_params)
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10
        
        # Map weight_params to weights summing to 1 and positive
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        # Calculate worst-case relative error using soft-max for smoothness
        errors = np.zeros((len(cosmo_params), len(z_test)))
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(weights, integrand_vals)
                
                true_val = true_vals[i, j]
                if true_val > 0:
                    errors[i, j] = abs(quad_val - true_val) / true_val
                else:
                    errors[i, j] = 0.0
        
        # Use log-sum-exp for smooth approximation of max
        flat_errors = errors.flatten()
        max_e = np.max(flat_errors)
        if max_e < 1e-15:
            return max_e
        # Soft-max with temperature
        temp = 0.01
        log_sum_exp = max_e + temp * np.log(np.sum(np.exp((flat_errors - max_e) / temp)))
        return log_sum_exp / temp
    
    # Start from Gauss-Legendre rule
    n = 8
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    gl_weights = 0.5 * gl_weights
    gl_weights = gl_weights / gl_weights.sum()
    
    # Invert the sigmoid mapping to get initial node_params
    # For node[0] = sigmoid(p0) => p0 = log(node[0] / (1 - node[0]))
    # For node[i] = node[i-1] + sigmoid(p_i) * (1 - node[i-1])
    #   => sigmoid(p_i) = (node[i] - node[i-1]) / (1 - node[i-1])
    #   => p_i = log((node[i] - node[i-1]) / (1 - node[i] + node[i-1]))
    
    node_params_init = np.zeros(8)
    node_params_init[0] = np.log(gl_nodes[0] / (1.0 - gl_nodes[0]))
    for i in range(1, 8):
        ratio = (gl_nodes[i] - gl_nodes[i-1]) / (1.0 - gl_nodes[i-1])
        ratio = np.clip(ratio, 1e-10, 1.0 - 1e-10)
        node_params_init[i] = np.log(ratio / (1.0 - ratio))
    
    # Invert softmax for weights: w_i = exp(p_i) / sum(exp(p_j))
    # We can set p_i = log(w_i) since softmax is invariant to adding constant
    weight_params_init = np.log(gl_weights + 1e-15)
    
    initial_params = np.concatenate([node_params_init, weight_params_init])
    
    # Run L-BFGS-B optimization with multiple restarts if time allows
    best_result = None
    best_obj = np.inf
    
    try:
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # If we have time, try a second run with slightly perturbed initial point
    if time.time() < deadline - 15.0 and best_result is not None:
        try:
            perturbed = best_result + 0.01 * np.random.randn(16)
            result2 = minimize(
                objective,
                perturbed,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_result = result2.x
        except Exception:
            pass
    
    if best_result is None:
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    # Check time
    if time.time() > deadline:
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    # Extract best params
    node_params = best_result[:8]
    weight_params = best_result[8:]
    
    # Reconstruct nodes and weights
    nodes = _nodes_from_params(node_params)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    
    # Map weight_params to weights
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Polish: do a final high-precision local optimization on the raw nodes/weights
    # using the true max error (not soft-max)
    def final_objective(p):
        """Direct optimization on nodes (8) and weights (8) with constraints handled by penalty."""
        n_params = p[:8]
        w_params = p[8:]
        
        # Nodes: use same sigmoid mapping
        nds = _nodes_from_params(n_params)
        nds = np.clip(nds, 1e-10, 1.0 - 1e-10)
        for i in range(1, 8):
            if nds[i] <= nds[i-1]:
                nds[i] = nds[i-1] + 1e-10
        
        # Weights via softmax
        ew = np.exp(w_params - np.max(w_params))
        wts = ew / np.sum(ew)
        
        # True max relative error
        max_err = 0.0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_u = z * nds
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(wts, integrand_vals)
                true_val = true_vals[i, j]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Polish with tighter tolerance
    try:
        polish_result = minimize(
            final_objective,
            best_result,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        if polish_result.fun <= best_obj + 1e-10:
            best_result = polish_result.x
    except Exception:
        pass
    
    # Reconstruct final nodes and weights
    node_params = best_result[:8]
    weight_params = best_result[8:]
    
    nodes = _nodes_from_params(node_params)
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10
    
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Final validation
    if not (len(nodes) == 8 and len(weights) == 8):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(np.diff(nodes) > 0):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if not np.all(weights > 0):
        return {"nodes": gl_nodes, "weights": gl_weights}
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
