"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 8
    
    # Define the cosmology models we'll optimize against
    # CPL model: w(z) = w0 + w1 * z, with E(a) = (1/a)^(3(1+w0+w1)/2) * (1/a - 1)^(3w1/2)
    # More precisely: E(a) = a^(-3(1+w0+w1)/2) * (1-a)^(3w1/2)
    # Actually: H(a)/H0 = (a/1)^(-3(1+w0+w1)/2) * ((1-a)/(1-1))^(3w1/2) ... let me be careful.
    
    # For CPL: w(a) = w0 + w1*(1-a)
    # H^2/H0^2 = a^(-3(1+w0+w1)) * (1-a)^(3w1)
    # E(a) = H(a)/H0 = a^(-3(1+w0+w1)/2) * (1-a)^(3w1/2)
    
    def make_E(w0, w1):
        def E(a):
            # Handle a near 0 or 1 carefully
            a_safe = np.clip(a, 1e-15, 1.0 - 1e-15)
            return a_safe**(-1.5*(1.0 + w0 + w1)) * (1.0 - a_safe)**(1.5*w1)
        return E
    
    # Representative set of CPL cosmologies for optimization
    # Include LambdaCDM (w0=-1, w1=0), and various others
    cpl_params = [
        (-1.0, 0.0),    # LambdaCDM
        (-1.0, 0.5),    # CPL with positive w1
        (-1.0, -0.5),   # CPL with negative w1
        (-0.9, 0.3),    
        (-1.1, 0.5),    
        (-0.8, -0.4),   
        (-1.2, 0.2),    
        (-0.7, 0.6),    
    ]
    
    # The integrand for luminosity distance:
    # dL/(c/H0) = (1+z) * z * integral_0^z dz'/E(a(z'))
    # where a = 1/(1+z), so z = 1/a - 1, dz = -da/a^2
    # 
    # In terms of a: integral from a=1 to a=1/(1+z) of (1/a - 1) * (1/a) * da/a^2 ... 
    # Let me redo this.
    #
    # dL/(c/H0) = (1+z) * integral_0^z dz'/E(a(z'))
    # a = 1/(1+z), so z = 1/a - 1
    # dz = -da/a^2
    # When z=0, a=1; when z=z_max, a=1/(1+z_max)
    #
    # integral_0^z dz'/E(a') = integral_{a=1}^{a=1/(1+z)} [-da/a^2] / E(a)
    #                         = integral_{a=1/(1+z)}^{a=1} da / (a^2 * E(a))
    #
    # So dL/(c/H0) = (1+z) * integral_{1/(1+z)}^{1} da / (a^2 * E(a))
    #
    # Now change variable to x = 1 - a, so a = 1-x, da = -dx
    # When a = 1/(1+z), x = 1 - 1/(1+z) = z/(1+z)
    # When a = 1, x = 0
    #
    # integral = integral_{x=z/(1+z)}^{0} [-dx] / ((1-x)^2 * E(1-x))
    #          = integral_0^{z/(1+z)} dx / ((1-x)^2 * E(1-x))
    #
    # Hmm, but the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    #
    # So the quadrature is approximating: integral_0^z dz' / E(a(z'))
    # using nodes in z-space? Let me re-read.
    #
    # "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    #
    # So it seems like the integral being approximated is:
    # integral_0^1 f(t) dt where f(t) = 1/E(a(t)) and a(t) = 1/(1+z*t)
    #
    # Because: integral_0^1 dt/E(1/(1+z*t)) = integral_0^1 dt / E(a(t))
    #
    # Let's verify: let u = z*t, du = z*dt, so dt = du/z
    # integral_0^1 dt/E(1/(1+z*t)) = (1/z) * integral_0^z du/E(1/(1+u))
    #
    # So dL/(c/H0) = (1+z) * integral_0^z dz'/E(a(z'))
    #              = (1+z) * z * integral_0^1 dt/E(a(t))
    #
    # where a(t) = 1/(1+z*t). Yes!
    #
    # So the quadrature approximates: integral_0^1 g(t) dt ≈ sum_i w_i * g(t_i)
    # where g(t) = 1/E(a(t)) = 1/E(1/(1+z*t))
    #
    # The nodes t_i are in [0,1], and they're universal (don't depend on z or the cosmology).
    
    def compute_error(nodes, weights, z_vals, cpl_params_list):
        """Compute worst relative error across all z and cosmologies."""
        max_rel_err = 0.0
        for w0, w1 in cpl_params_list:
            E_func = make_E(w0, w1)
            for z in z_vals:
                # Reference: adaptive integration
                from scipy.integrate import quad
                def integrand(t):
                    a_t = 1.0 / (1.0 + z * t)
                    return 1.0 / E_func(a_t)
                
                ref_val, _ = quad(integrand, 0, 1, limit=200)
                
                # Quadrature approximation
                a_nodes = 1.0 / (1.0 + z * nodes)
                quad_val = np.sum(weights / E_func(a_nodes))
                
                if abs(ref_val) > 1e-15:
                    rel_err = abs(quad_val - ref_val) / abs(ref_val)
                    max_rel_err = max(max_rel_err, rel_err)
        return max_rel_err
    
    # z values to test
    z_test = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    # Initialize with Gauss-Legendre nodes
    gl_nodes, gl_weights = roots_legendre(n)
    # Map from [-1,1] to [0,1]
    x0 = 0.5 * (gl_nodes + 1.0)
    w0_init = 0.5 * gl_weights
    
    best_nodes = x0.copy()
    best_weights = w0_init.copy()
    best_error = float('inf')
    
    # Objective function for optimization
    def objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        nodes = params[:n]
        log_weights = params[n:]
        
        # Ensure nodes are in (0,1) and strictly increasing
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(np.diff(nodes) <= 0):
            return 1e10
        
        weights = np.exp(log_weights)
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        if np.any(weights <= 0):
            return 1e10
        
        return compute_error(nodes, weights, z_test, cpl_params)
    
    # Local minimax optimization using Nelder-Mead
    x_init = np.concatenate([x0, np.log(w0_init)])
    
    try:
        result = minimize(objective, x_init, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 
                                  'adaptive': True})
        elapsed = time.time() - start_time
        if elapsed < deadline:
            nodes_opt = result.x[:n]
            log_w_opt = result.x[n:]
            weights_opt = np.exp(log_w_opt)
            weights_opt = weights_opt / np.sum(weights_opt)
            
            # Validate
            if (np.all(nodes_opt > 0) and np.all(nodes_opt < 1) and 
                np.all(np.diff(nodes_opt) > 0) and np.all(weights_opt > 0)):
                err = compute_error(nodes_opt, weights_opt, z_test, cpl_params)
                if err < best_error:
                    best_error = err
                    best_nodes = nodes_opt
                    best_weights = weights_opt
    except Exception:
        pass
    
    # If optimization didn't improve, use initial
    if best_error == float('inf'):
        best_nodes = x0
        best_weights = w0_init
    
    # Final validation: ensure constraints
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure strictly increasing with small epsilon if needed
    for i in range(1, n):
        if best_nodes[i] - best_nodes[i-1] < 1e-12:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    
    # Ensure nodes in (0,1)
    best_nodes[0] = max(best_nodes[0], 1e-10)
    best_nodes[-1] = min(best_nodes[-1], 1.0 - 1e-10)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
