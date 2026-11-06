"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # The integrand for luminosity distance in dimensionless form:
    # dL/(c/H0) = (1+z) * ∫_0^z dz' / E(z')
    # where E(z) = H(z)/H0
    # 
    # In terms of a = 1/(1+z), with z from 0 to z_max, a from 1 to 1/(1+z_max)
    # But the task uses nodes in [0,1] as a fraction of z range.
    # The formula given: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i])
    # 
    # So the quadrature approximates: ∫_0^z dz'/E(z') ≈ z * sum_i w_i / E(1/(1+z*x_i))
    # where x_i are the nodes in [0,1]
    # 
    # This means we're approximating f(t) = 1/E(1/(1+z*t)) for t in [0,1]
    # using quadrature: ∫_0^1 f(t) dt ≈ sum_i w_i f(x_i)
    # Then dL/(c/H0) = (1+z) * z * ∫_0^1 dt/E(1/(1+z*t))
    
    # We need to find 8 nodes x_i in (0,1) strictly increasing and positive weights w_i summing to 1
    # that minimize worst-case relative error across different CPL models.
    
    # CPL model: w(z) = w0 + wa*z/(1+z)
    # E(z)^2 = Omega_m*(1+z)^3 + Omega_k*(1+z)^2 + Omega_Lambda*(1+z)^{3(1+w0+wa)}
    # where Omega_m + Omega_k + Omega_Lambda = 1
    
    # For spatially flat: Omega_k = 0, so Omega_Lambda = 1 - Omega_m
    # E(z)^2 = Omega_m*(1+z)^3 + (1-Omega_m)*(1+z)^{3(1+w0+wa)}
    
    def compute_E_sq(z, Om, w0, wa):
        """Compute E(z)^2 for CPL model."""
        # w_eff(z) = w0 + wa*z/(1+z)
        # E^2 = Om*(1+z)^3 + (1-Om)*(1+z)^{3(1+w0+wa)}
        # Note: for varying w, the exponent is 3(1+w0+wa) only if we use the 
        # standard CPL approximation. Actually for CPL:
        # E(z)^2 = Om*(1+z)^3 + (1-Om)*exp(3*wa*((1+z)/(w0+1) - 1 + w0/(w0+1)*log(1+z)))
        # But a simpler approximation often used:
        # E(z)^2 = Om*(1+z)^3 + (1-Om)*(1+z)^{3(1+w0+wa)}
        # Let's use the exact CPL formula:
        # H(z)/H0 = sqrt(Om*(1+z)^3 + (1-Om)*exp(3*wa*(1 - (1+z)/(w0+1) + w0*log(1+z)/(w0+1))))
        # Actually the standard CPL:
        # E(z)^2 = Om*(1+z)^3 + (1-Om)*(1+z)^3 * exp(3*wa*(1 - (1+z)/(1+w0) + w0/(1+w0)*log(1+z)))
        # Let me use a cleaner form. For CPL w(z) = w0 + wa*z/(1+z):
        # ∫ w(z)/(1+z) dz from 0 to z = wa*(1 - 1/(1+z) + w0/(1+w0)*log(1+z))  ... no
        # The standard result:
        # E(z)^2 = Om*(1+z)^3 + (1-Om)*(1+z)^3 * exp(3*wa*(1 - (1+z)/(1+w0) + w0/(1+w0)*np.log(1+z)))
        # Let me just use the power-law approximation which is simpler and commonly used:
        return Om * (1 + z)**3 + (1 - Om) * (1 + z)**(3*(1 + w0 + wa))
    
    def compute_E(z, Om, w0, wa):
        return np.sqrt(compute_E_sq(z, Om, w0, wa))
    
    def quadrature_error(z, nodes, weights, Om, w0, wa):
        """Compute relative error of quadrature for given z and model."""
        # True integral: ∫_0^z dz'/E(z')
        # Using adaptive quadrature as reference
        from scipy.integrate import quad
        integrand = lambda zp: 1.0 / compute_E(zp, Om, w0, wa)
        true_val, _ = quad(integrand, 0, z, limit=200)
        
        # Quadrature estimate
        a_vals = 1.0 / (1.0 + z * nodes)
        z_vals = (1.0 / a_vals) - 1.0  # z' = 1/a - 1
        E_vals = compute_E(z_vals, Om, w0, wa)
        quad_val = z * np.sum(weights / E_vals)
        
        if abs(true_val) < 1e-15:
            return 0.0
        return abs(quad_val - true_val) / abs(true_val) * 100.0  # percent
    
    # Define a set of test cosmologies (CPL parameters)
    # Om in [0.2, 0.4], w0 in [-1.5, -0.5], wa in [-1.0, 1.0]
    test_oms = [0.25, 0.30, 0.35, 0.40]
    test_w0s = [-1.5, -1.0, -0.5]
    test_was = [-1.0, -0.5, 0.0, 0.5, 1.0]
    test_zs = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Build list of (Om, w0, wa, z) test cases
    test_cases = []
    for Om in test_oms:
        for w0 in test_w0s:
            for wa in test_was:
                for z in test_zs:
                    test_cases.append((Om, w0, wa, z))
    
    def worst_case_error(params):
        """Compute worst-case relative error (percent) for given nodes and weights."""
        # params: 16 values = 8 nodes + 8 weights
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce constraints: nodes in (0,1) strictly increasing, weights positive, sum to 1
        # We'll use a penalty approach
        penalty = 0.0
        
        # Nodes must be in (0,1) and strictly increasing
        for i in range(8):
            if nodes[i] <= 0.001 or nodes[i] >= 0.999:
                penalty += 1e6 * (max(0, 0.001 - nodes[i]) + max(0, nodes[i] - 0.999))
        for i in range(7):
            if nodes[i+1] - nodes[i] < 0.001:
                penalty += 1e6 * (0.001 - (nodes[i+1] - nodes[i]))
        
        # Weights positive and sum to 1
        for i in range(8):
            if weights[i] < 0.0001:
                penalty += 1e6 * (0.0001 - weights[i])
        weight_sum = np.sum(weights)
        penalty += 1e6 * (weight_sum - 1.0)**2
        
        if penalty > 1e3:
            return penalty
        
        # Compute worst-case error
        max_err = 0.0
        for (Om, w0, wa, z) in test_cases:
            err = quadrature_error(z, nodes, weights, Om, w0, wa)
            if err > max_err:
                max_err = err
            if time.time() > deadline - 5:
                break
        return max_err + penalty
    
    # Initial guess: Gauss-Legendre-like nodes and equal weights
    # Use Gauss-Legendre nodes and weights for 8 points
    from numpy.polynomial.legendre import leggauss
    gl_nodes, gl_weights = leggauss(8)
    # Map from [-1,1] to [0,1]
    init_nodes = (gl_nodes + 1) / 2
    init_weights = gl_weights / 2  # scale for the change of variables
    
    x0 = np.concatenate([init_nodes, init_weights])
    
    best_params = x0.copy()
    best_err = worst_case_error(x0)
    
    # Try Nelder-Mead optimization
    if time.time() < deadline - 10:
        try:
            result = minimize(worst_case_error, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6,
                                     'adaptive': True})
            if result.fun < best_err:
                best_params = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        # Random perturbation
        rand_nodes = np.sort(np.random.uniform(0.05, 0.95, 8))
        rand_weights = np.random.dirichlet(np.ones(8))
        x_rand = np.concatenate([rand_nodes, rand_weights])
        
        try:
            result = minimize(worst_case_error, x_rand, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6,
                                     'adaptive': True})
            if result.fun < best_err:
                best_params = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Extract final nodes and weights
    final_nodes = best_params[:8]
    final_weights = best_params[8:]
    
    # Ensure strict positivity and sum to 1
    final_weights = np.maximum(final_weights, 1e-10)
    final_weights = final_weights / np.sum(final_weights)
    
    # Ensure nodes are strictly increasing in (0,1)
    final_nodes = np.clip(final_nodes, 0.001, 0.999)
    for i in range(7):
        if final_nodes[i+1] - final_nodes[i] < 0.001:
            final_nodes[i+1] = final_nodes[i] + 0.001
    # Re-normalize if needed
    final_nodes = np.clip(final_nodes, 0.001, 0.999)
    
    # Final check: weights sum to 1 within 1e-12
    final_weights = final_weights / np.sum(final_weights)
    
    return {"nodes": final_nodes, "weights": final_weights}
# EVOLVE-BLOCK-END
