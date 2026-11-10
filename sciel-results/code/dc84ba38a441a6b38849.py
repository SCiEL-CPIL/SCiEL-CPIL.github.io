import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    # Cosmology parameters
    Om_m = 0.3
    Om_de = 0.7
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cosmologies = [(w0, wa) for w0 in w0_vals for wa in wa_vals]
    
    # z values for validation/optimization
    # Log spacing covers low and high z well
    z_vals = np.logspace(-2, 1, 40) # 0.01 to 10

    def E_of_z(z, w0, wa):
        """Compute E(z) = H(z)/H0 for CPL model."""
        if z == 0:
            return 1.0
        # E^2 = Om_m (1+z)^3 + Om_de (1+z)^(3(1+w0)) exp(3 wa z/(1+z))
        # Handle numerical stability for large z
        term1 = Om_m * (1+z)**3
        # For large z, (1+z)^(3(1+w0)) might overflow if 1+w0 > 0 and z is huge, 
        # but z<=10 is safe.
        exp_term = np.exp(3 * wa * z / (1+z))
        term2 = Om_de * (1+z)**(3*(1+w0)) * exp_term
        return np.sqrt(term1 + term2)

    def true_distance(z, w0, wa):
        """High-precision reference distance."""
        if z == 0:
            return 0.0
        integrand = lambda zp: 1.0 / E_of_z(zp, w0, wa)
        # quad is adaptive and accurate
        val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-10, epsrel=1e-10)
        return (1+z) * val

    def quadrature_distance(z, nodes, weights, w0, wa):
        """Compute distance using the quadrature rule."""
        if z == 0:
            return 0.0
        # a_i = 1/(1+z*nodes[i])
        # E(a_i) = E(z * nodes[i])
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1+z) * z * integral_approx

    def decode_params(params):
        """
        Decode raw parameters to nodes and weights.
        params[0:8] -> nodes via exp-cumsum
        params[8:16] -> weights via softmax
        """
        x = params[:8]
        y = params[8:16]
        
        # Nodes: exp(x) -> cumsum -> normalize to [0,1]
        # Adding a small offset to x prevents underflow if x is very negative
        x_safe = x - np.min(x) + 1.0 
        exp_x = np.exp(x_safe)
        cum_exp = np.cumsum(exp_x)
        nodes = cum_exp / cum_exp[-1]
        
        # Ensure strictly increasing and in (0,1)
        # The exp-cumsum normalization guarantees nodes in (0,1) and increasing.
        # However, node[0] might be very close to 0 and node[-1] is 1.
        # The task requires nodes in [0,1] and strictly increasing.
        # "Strictly increasing in [0,1]" usually implies 0 < n1 < ... < n8 < 1 or similar.
        # The formula uses a_i = 1/(1+z*nodes[i]). If node=1, a=1/(1+z). If node=0, a=1.
        # The integral is from 0 to 1 in the u-substitution.
        # Standard quadrature nodes are in (0,1).
        
        # Weights: softmax
        y_safe = y - np.max(y)
        exp_y = np.exp(y_safe)
        weights = exp_y / np.sum(exp_y)
        
        return nodes, weights

    def objective_smooth(params):
        """
        Smooth surrogate objective: Sum of squared relative errors.
        Used for L-BFGS-B initialization.
        """
        if time.time() > deadline:
            return 1e10
        nodes, weights = decode_params(params)
        
        total_err = 0.0
        # Sample a subset of cosmologies/z for speed during optimization
        # Use all if fast enough, else subsample
        for w0, wa in cosmologies:
            for z in z_vals:
                d_true = true_distance(z, w0, wa)
                if d_true == 0: continue
                d_quad = quadrature_distance(z, nodes, weights, w0, wa)
                rel_err = abs(d_quad - d_true) / d_true
                total_err += rel_err**2
                
        return total_err

    def objective_minimax(params):
        """
        Minimax objective: Max relative error.
        Used for Nelder-Mead refinement.
        """
        if time.time() > deadline:
            return 1e10
        nodes, weights = decode_params(params)
        
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_vals:
                d_true = true_distance(z, w0, wa)
                if d_true == 0: continue
                d_quad = quadrature_distance(z, nodes, weights, w0, wa)
                rel_err = abs(d_quad - d_true) / d_true
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # Initial Guesses
    # 1. Gauss-Legendre mapped to (0,1)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes_01 = (gl_nodes + 1) / 2
    gl_weights_01 = gl_weights / 2
    
    # Invert parameterization for GL
    # Nodes: cumsum(exp(x))/sum(exp(x)) = gl_nodes_01
    # This is tricky to invert exactly, so we just use a simple parameterization 
    # that maps to GL nodes approximately, or start with x=0 (uniform nodes)
    
    # Let's start with uniform nodes (x=0) and uniform weights (y=0)
    params_init_1 = np.zeros(16)
    
    # Start 2: Slightly perturbed
    params_init_2 = np.random.normal(0, 0.1, 16)
    
    # Start 3: Based on GL nodes approximation
    # If we set x such that exp(x) are proportional to diffs of GL nodes
    diffs = np.diff(gl_nodes_01)
    diffs = np.concatenate(([gl_nodes_01[0]], diffs)) # First node is just the first value?
    # Actually, cumsum(exp(x)) = C * gl_nodes_01
    # exp(x_i) = C * (gl_nodes_01[i] - gl_nodes_01[i-1]) for i>0, and C*gl_nodes_01[0] for i=0
    # So x_i = log(C * diff_i) = log(diff_i) + log(C)
    # Since C cancels in normalization, we can just use x_i = log(diff_i)
    # But diff_0 = gl_nodes_01[0].
    diffs_for_x = np.zeros(8)
    diffs_for_x[0] = gl_nodes_01[0]
    diffs_for_x[1:] = np.diff(gl_nodes_01)
    # Ensure positive
    diffs_for_x = np.maximum(diffs_for_x, 1e-10)
    x_init_gl = np.log(diffs_for_x)
    
    # Weights: softmax(y) = gl_weights_01
    # y_i = log(gl_weights_01[i])
    y_init_gl = np.log(gl_weights_01)
    
    params_init_gl = np.concatenate([x_init_gl, y_init_gl])
    
    best_params = params_init_1
    best_score = objective_smooth(params_init_1)
    
    inits = [params_init_1, params_init_2, params_init_gl]
    
    # Phase 1: L-BFGS-B on smooth objective
    for p_init in inits:
        if time.time() > deadline - 20: break
        try:
            res = minimize(objective_smooth, p_init, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 500})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Phase 2: Nelder-Mead on Minimax objective
    # Initialize NM with the best L-BFGS-B result
    # NM requires a simplex. We create one by perturbing the best point.
    x0 = best_params
    simplex = [x0]
    for i in range(16):
        x_pert = x0.copy()
        x_pert[i] += 0.1
        simplex.append(x_pert)
        
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(objective_minimax, x0, method='Nelder-Mead', 
                              options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
            if res_nm.fun < objective_minimax(best_params):
                best_params = res_nm.x
        except Exception:
            pass

    # Final check and extraction
    if time.time() > deadline:
        # Emergency exit
        pass

    nodes, weights = decode_params(best_params)
    
    # Post-processing: Ensure strict validity
    # Nodes must be strictly increasing in [0,1]
    # Weights positive and sum to 1
    
    # The decode function guarantees this mathematically, but floating point might cause issues.
    # Re-normalize weights
    weights = weights / np.sum(weights)
    
    # Ensure nodes are within [0,1] and strictly increasing
    # If any node is 0 or 1, clip slightly inside
    eps = 1e-12
    nodes = np.clip(nodes, eps, 1.0 - eps)
    # Ensure strict increase
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + eps * (1.0 - nodes[i-1])
    
    return {"nodes": nodes, "weights": weights}
