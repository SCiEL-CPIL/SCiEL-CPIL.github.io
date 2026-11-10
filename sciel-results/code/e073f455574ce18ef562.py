import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    t_start = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # CPL model: w(z) = w0 + wa*z/(1+z)
    # E(z) = H(z)/H0 for flat universe
    # E(z) = (1+z)^(3/2) * exp(-3/2 * (w0*z + wa*ln(1+z)))
    # In terms of a = 1/(1+z): E(a) = a^(-3/2) * exp(-3/2*(w0*(1-a)/a + wa*ln(a)))
    # Simplify: E(a) = a^(-3/2) * exp(-3/2*w0*(1-a)/a) * a^(-3/2*wa)
    #           = a^(-3/2 - 3*wa/2) * exp(-3/2*w0*(1-a)/a)
    
    def E_of_a(a, w0, wa):
        """E(a) = H(a)/H0 for flat CPL cosmology"""
        # a in (0, 1], z = (1-a)/a
        # E(z) = (1+z)^(3/2) * exp(-3/2*(w0*z + wa*log(1+z)))
        # 1+z = 1/a, z = (1-a)/a, log(1+z) = -log(a)
        # E = (1/a)^(3/2) * exp(-3/2*(w0*(1-a)/a + wa*(-log(a))))
        #   = a^(-3/2) * exp(-3*w0*(1-a)/(2*a) + 3*wa*log(a)/2)
        #   = a^(-3/2) * a^(3*wa/2) * exp(-3*w0*(1-a)/(2*a))
        #   = a^(-3/2 + 3*wa/2) * exp(-3*w0*(1-a)/(2*a))
        #   = a^(-3(1-wa)/2) * exp(-3*w0*(1-a)/(2*a))
        return a**(-1.5 + 1.5*wa) * np.exp(-1.5*w0*(1.0-a)/a)
    
    def reference_integral(z, w0, wa):
        """Compute dL/(c/H0) = integral_0^z (1+z')/E(z') dz' via adaptive quad"""
        def integrand(zp):
            a = 1.0/(1.0+ zp)
            return (1.0 + zp) / E_of_a(a, w0, wa)
        val, _ = quad(integrand, 0, z, limit=100, epsabs=1e-12, epsrel=1e-12)
        return val
    
    def quadrature_value(z, nodes, weights, w0, wa):
        """Compute (1+z)*z*sum_i w_i/E(a_i)"""
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return (1.0 + z) * z * np.sum(weights / E_vals)
    
    # Generate representative cosmologies
    # w0 in [-1.5, 0.5], wa in [-1.0, 1.0]
    np.random.seed(42)
    n_cosmo = 25
    w0_vals = np.linspace(-1.2, 0.3, 5)
    wa_vals = np.linspace(-0.8, 0.8, 5)
    cosmologies = []
    for w0 in w0_vals:
        for wa in wa_vals:
            cosmologies.append((w0, wa))
    
    # z values to test
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    # Precompute reference values
    ref_data = []  # (z, w0, wa, ref_val)
    for (w0, wa) in cosmologies:
        for z in z_test:
            rv = reference_integral(z, w0, wa)
            ref_data.append((z, w0, wa, rv))
    
    def objective(params):
        """Minimize sum of squared relative errors"""
        nodes = params[:8]
        log_weights = params[8:]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        # Check nodes are in (0,1) and increasing
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        
        total_err = 0.0
        for (z, w0, wa, rv) in ref_data:
            qv = quadrature_value(z, nodes, weights, w0, wa)
            rel_err = (qv - rv) / rv
            total_err += rel_err**2
        return total_err
    
    # Initial guess: Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    # Map from [-1,1] to [0,1]
    init_nodes = (gl_nodes + 1.0) / 2.0
    init_weights = gl_weights / 2.0  # Jacobian
    init_log_w = np.log(init_weights)
    init_params = np.concatenate([init_nodes, init_log_w])
    
    # Optimize
    best_params = init_params
    best_obj = objective(init_params)
    
    t_check = time.time()
    if t_check - t_start < DEADLINE - 10:
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
    
    t_check = time.time()
    if t_check - t_start < DEADLINE - 5:
        try:
            res2 = minimize(objective, best_params, method='Powell',
                           options={'maxiter': 30, 'ftol': 1e-14, 'xtol': 1e-12})
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    t_check = time.time()
    if t_check - t_start < DEADLINE - 3:
        # Try a few random restarts if time permits
        for _ in range(3):
            t_check = time.time()
            if t_check - t_start > DEADLINE - 3:
                break
            try:
                # Perturb from best
                noise = np.random.randn(16) * 0.05
                pert_params = best_params + noise
                # Ensure nodes stay in (0,1)
                pert_params[:8] = np.clip(pert_params[:8], 0.01, 0.99)
                # Enforce increasing order
                for i in range(1, 8):
                    if pert_params[i] <= pert_params[i-1]:
                        pert_params[i] = pert_params[i-1] + 0.01
                pert_params[:8] = np.clip(pert_params[:8], 0.01, 0.99)
                
                obj_pert = objective(pert_params)
                if obj_pert < best_obj:
                    res3 = minimize(objective, pert_params, method='L-BFGS-B',
                                   options={'maxiter': 30, 'ftol': 1e-12})
                    if res3.fun < best_obj:
                        best_obj = res3.fun
                        best_params = res3.x
            except Exception:
                pass
    
    # Extract final result
    final_nodes = best_params[:8]
    final_log_w = best_params[8:]
    final_weights = np.exp(final_log_w)
    final_weights = final_weights / np.sum(final_weights)
    
    # Ensure nodes are strictly increasing in (0,1)
    final_nodes = np.sort(final_nodes)
    final_nodes = np.clip(final_nodes, 1e-6, 1.0 - 1e-6)
    # Ensure strictly increasing
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-6
    final_nodes = np.clip(final_nodes, 1e-6, 1.0 - 1e-6)
    
    # Verify weights sum to 1
    final_weights = final_weights / np.sum(final_weights)
    
    return {"nodes": final_nodes, "weights": final_weights}
