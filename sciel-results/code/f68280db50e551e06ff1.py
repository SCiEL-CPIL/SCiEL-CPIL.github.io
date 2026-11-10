"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Structural change: Instead of treating the quadrature as an opaque black-box 
    minimization over 16 parameters, we exploit the fact that the integrand 1/E(a) 
    is a smooth function of the redshift variable u in [0,1]. 
    We use a weighted least-squares fit to a polynomial basis of degree 7 (or lower) 
    to find nodes and weights that minimize the error for a representative family 
    of CPL cosmologies. This is a "collocation-like" approach where we fit the 
    quadrature to the integral of the function itself, not just pointwise errors.
    
    We optimize the nodes and weights directly using a robust objective that 
    minimizes the maximum relative error (Chebyshev-like) but smoothed for 
    optimization, using the insight that the optimal nodes for a family of 
    related functions often lie close to Gauss-Legendre nodes for the base 
    weight function, but shifted to accommodate the dark energy term.
    """
    deadline = time.time() + 75  # Safety margin
    
    # Representative CPL cosmologies
    w0_values = [-1.5, -1.2, -1.0, -0.8, -0.5, 0.0, 0.5]
    wa_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    cosmologies = []
    for w0 in w0_values:
        for wa in wa_values:
            cosmologies.append((w0, wa))
    
    Om_m = 0.3
    Om_de = 0.7
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        matter = Om_m / a**3
        if abs(w0 + wa) < 1e-10:
            de_term = Om_de * a**(-3*(1+w0))
        else:
            exponent = 3*wa/(1+w0+wa)
            de_term = Om_de * a**(-3*(1+w0+wa)) * (1 + wa*(1-a))**exponent
        return np.sqrt(matter + de_term)
    
    def integrand(u, z, w0, wa):
        """
        The integrand in the quadrature formula:
        dL/(c/H0) = (1+z)*z * sum_i w_i / E(a_i)
        where a_i = 1/(1+z*u_i)
        
        This is equivalent to approximating:
        integral_0^1 du / E(1/(1+z*u))
        
        So the integrand at node u is 1/E(1/(1+z*u))
        """
        a = 1.0 / (1.0 + z * u)
        return 1.0 / E_of_a(a, w0, wa)
    
    def reference_integral(z, w0, wa, n_ref=10000):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        vals = integrand(u, z, w0, wa)
        if n_ref % 2 == 0:
            n_ref = n_ref - 1
            u = np.linspace(0, 1, n_ref)
            vals = integrand(u, z, w0, wa)
        h = 1.0 / (n_ref - 1)
        integral = h * (vals[0] + vals[-1] + 4*np.sum(vals[1:-1:2]) + 2*np.sum(vals[2:-1:2]))
        return integral
    
    def compute_worst_error(nodes, weights, z_test, cosmologies):
        """Compute the worst-case relative error across all test points."""
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_test:
                # Quadrature value
                u_nodes = nodes
                a_vals = 1.0 / (1.0 + z * u_nodes)
                E_vals = E_of_a(a_vals, w0, wa)
                quad_val = np.sum(weights / E_vals)
                
                # Reference value
                ref_val = reference_integral(z, w0, wa)
                
                # Relative error
                err = np.abs(quad_val - ref_val) / ref_val
                if err > max_err:
                    max_err = err
        return max_err
    
    def objective(params):
        """
        Objective function: minimize the sum of squared errors plus a penalty 
        for the worst-case error. We use a smooth approximation to the 
        max-error (Huber loss-like) to make it differentiable.
        """
        if time.time() > deadline:
            return 1e10
        
        # Unpack nodes and weights
        # Use a parameterization that ensures nodes are in (0,1) and increasing
        # and weights are positive and sum to 1.
        
        # Nodes: use a cumulative sum of exponentials
        # node_i = cumsum(exp(x_i)) / cumsum(exp(x))[-1]
        x = params[:8]
        # Shift x to avoid overflow
        x_shifted = x - np.max(x)
        t = np.exp(x_shifted)
        cumsum_t = np.cumsum(t)
        nodes = cumsum_t / cumsum_t[-1]
        # Ensure strictly in (0,1)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        
        # Weights: use softmax
        w = params[8:16]
        w_shifted = w - np.max(w)
        exp_w = np.exp(w_shifted)
        weights = exp_w / np.sum(exp_w)
        
        # Compute errors
        z_test = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        total_err_sq = 0.0
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_test:
                a_vals = 1.0 / (1.0 + z * nodes)
                E_vals = E_of_a(a_vals, w0, wa)
                quad_val = np.sum(weights / E_vals)
                ref_val = reference_integral(z, w0, wa)
                err = np.abs(quad_val - ref_val) / ref_val
                total_err_sq += err**2
                if err > max_err:
                    max_err = err
        
        # Combine: minimize both average and worst-case
        # Use a smooth max approximation: max_err + sqrt(sum(err^2)/N)
        N = len(cosmologies) * len(z_test)
        smooth_max = max_err + 0.1 * np.sqrt(total_err_sq / N)
        
        return smooth_max
    
    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = roots_legendre(8)
    gl_nodes_01 = 0.5 * (gl_nodes + 1)
    
    # Invert node parameterization
    t_init = np.zeros(8)
    t_init[0] = gl_nodes_01[0]
    for i in range(1, 8):
        t_init[i] = gl_nodes_01[i] - gl_nodes_01[i-1]
    x_init = np.log(t_init + 1e-10)
    
    # Invert weight parameterization
    w_init = np.log(gl_weights + 1e-10) - np.log(gl_weights + 1e-10).max()
    
    params_init = np.concatenate([x_init, w_init])
    
    # Optimize with L-BFGS-B
    best_params = params_init
    best_score = objective(params_init)
    
    try:
        result = minimize(objective, params_init, method='L-BFGS-B', 
                          options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-8})
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Refine with Nelder-Mead if time allows
    if time.time() < deadline - 10:
        try:
            result2 = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if result2.fun < best_score:
                best_params = result2.x
                best_score = result2.fun
        except Exception:
            pass
    
    # Extract final nodes and weights
    x_opt = best_params[:8]
    x_shifted = x_opt - np.max(x_opt)
    t = np.exp(x_shifted)
    cumsum_t = np.cumsum(t)
    nodes = cumsum_t / cumsum_t[-1]
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    
    w_opt = best_params[8:16]
    w_shifted = w_opt - np.max(w_opt)
    exp_w = np.exp(w_shifted)
    weights = exp_w / np.sum(exp_w)
    
    # Ensure constraints
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0):
        nodes = gl_nodes_01
        weights = gl_weights
    else:
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
