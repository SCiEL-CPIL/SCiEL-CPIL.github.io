import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time
import itertools

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    The integrand is f(z) = z / E(a) where a = 1/(1+z), and the full distance is:
    dL/(c/H0) = (1+z) * integral_0^z f(z') dz'
    
    We are approximating integral_0^z f(z') dz' with a fixed 8-node rule on [0,1]
    where the variable is u = z'/z, so z' = z*u, a = 1/(1+z*u), and:
    integral_0^z f(z') dz' = z * integral_0^1 f(z*u) du
    
    So the quadrature approximates:
    integral_0^1 [ (z*u) / E(1/(1+z*u)) ] du
    
    with fixed nodes u_i in [0,1] and weights w_i.
    """
    start_time = time.time()
    time_limit = 85.0  # Leave 5 seconds margin
    
    # Define a family of cosmologies to optimize against.
    # We'll use a diverse set of CPL (CPL) models: w0, wa parameters.
    # E(a) = (Omega_m * a^-3 + Omega_r * a^-4 + Omega_k * a^-2 + Omega_L * a^(-3(1+w)))^(1/2)
    # For flat: Omega_k = 0, Omega_L = 1 - Omega_m
    # CPL: w = w0 + wa * (1 - a)
    # For CPL, the integral form of the DE term is:
    # int a^(-3(1+w)) da = a^(-3(1+w0-wa)) / (-3(1+w0-wa)) if w0 != wa
    # But we just need E(a) at specific points, so we can compute it directly.
    
    # Actually, for the integrand we just need E(a) at the 8 nodes for various z and various cosmologies.
    # Let's define E(a; params) directly.
    
    def E_of_a(a, Omega_m, w0, wa):
        """
        Compute E(a) = H(a)/H0 for flat CPL model.
        a is scale factor.
        """
        # Matter: Omega_m * a^-3
        # Radiation: negligible for high-z structure, but let's include a small amount? 
        # Actually, the task says "spatially flat, time-varying dark-energy models". 
        # Typically these are matter + dark energy, maybe no radiation for simplicity 
        # or radiation is negligible at the z ranges where we care.
        # Let's assume matter + dark energy only.
        # E^2 = Omega_m * a^-3 + (1 - Omega_m) * a^(-3(1+w(a)))
        # where w(a) = w0 + wa * (1 - a)
        
        # To compute a^(-3(1+w(a))) we need to integrate the exponent if w varies with a.
        # But E(a) is the *value* of H/H0 at scale factor a, which depends on the 
        # integrated effect of w from a to 1.
        # H^2/H0^2 = Omega_m a^-3 + Omega_L * exp(-3 * integral_a^1 (1+w(a'))/a' da')
        # For CPL: integral_a^1 (1+w(a'))/a' da' = integral_a^1 [1+w0+wa(1-a')]/a' da'
        # = integral_a^1 (1+w0+wa)/a' da' - wa * integral_a^1 da'
        # = (1+w0+wa) * ln(1/a) - wa * (1-a)
        
        log_factor = (1.0 + w0 + wa) * np.log(1.0 / a) - wa * (1.0 - a)
        # The factor for dark energy is exp(-3 * log_factor)
        # But wait, the standard CPL form is:
        # Omega_L(a) = Omega_L0 * a^(-3(1+w0+wa)) * exp(3*wa*(1-a))
        # So E^2 = Omega_m * a^-3 + Omega_L0 * a^(-3(1+w0+wa)) * exp(3*wa*(1-a))
        
        # Let's use this explicit form.
        term_m = Omega_m * a**(-3)
        term_L = (1.0 - Omega_m) * a**(-3*(1+w0+wa)) * np.exp(3*wa*(1-a))
        return np.sqrt(term_m + term_L)
    
    def integrand_z(z, u, Omega_m, w0, wa):
        """
        The integrand in the variable u = z'/z.
        z' = z * u
        a = 1 / (1 + z') = 1 / (1 + z*u)
        integrand = z' / E(a) = z*u / E(1/(1+z*u))
        """
        a = 1.0 / (1.0 + z * u)
        E = E_of_a(a, Omega_m, w0, wa)
        return z * u / E
    
    def quadrature_error(nodes, weights, z, Omega_m, w0, wa):
        """
        Compute relative error of the quadrature for a given z and cosmology.
        """
        # Quadrature approximation of integral_0^1 f(z*u) du
        # where f(z*u) = z*u / E(1/(1+z*u))
        # The actual integral is:
        # I_exact = integral_0^1 [z*u / E(1/(1+z*u))] du
        
        # Our approximation:
        I_quad = np.sum(weights * [integrand_z(z, u, Omega_m, w0, wa) for u in nodes])
        
        # Exact integral using adaptive quadrature:
        def f_integrand(u):
            return integrand_z(z, u, Omega_m, w0, wa)
        
        I_exact, _ = quad(f_integrand, 0, 1, limit=200)
        
        if abs(I_exact) < 1e-15:
            return 0.0
        return abs(I_quad - I_exact) / abs(I_exact)
    
    # Define a set of validation cosmologies and redshifts.
    # We want to minimize the worst-case relative error.
    
    # Cosmologies: (Omega_m, w0, wa)
    # Typical values: Omega_m ~ 0.2-0.4, w0 ~ -1 to -0.5, wa ~ -1 to 1
    cosmologies = [
        (0.27, -1.0, 0.0),   # LambdaCDM
        (0.27, -0.9, 0.1),   # Mildly varying w
        (0.30, -1.1, 0.5),   # Phantom-ish, varying
        (0.25, -0.5, -0.5),  # Quintessence-like
        (0.35, -1.0, 0.3),   # Higher matter, varying
        (0.27, -1.0, -0.3),  # Varying w, negative wa
        (0.30, -0.8, 0.2),   # Moderate w0, positive wa
        (0.28, -1.2, 0.8),   # More phantom, strong variation
    ]
    
    # Redshifts to test: we need to cover a range.
    # The quadrature is for integral_0^1 in u, but the actual z can vary.
    # The error depends on z because the shape of f(z*u) changes with z.
    redshifts = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute a few test points for fast evaluation during optimization.
    # We'll use a subset for the optimization and check against all at the end.
    
    def objective(params):
        """
        params: array of 16 values: 8 nodes and 8 weights.
        We'll use a transformation to ensure nodes are in (0,1) and increasing,
        and weights are positive.
        """
        # Transform params to valid nodes and weights
        # nodes: use cumulative sum of exponentials to ensure positive and increasing
        # Actually, let's use a different parameterization.
        # Let nodes be determined by 7 internal values in (0,1) plus boundaries.
        # Or use a simpler approach: optimize the 8 nodes directly with bounds,
        # and use log-transform for weights.
        
        # For now, let's use a simple parameterization:
        # First 8 params: log-spaced or something? 
        # Let's just use raw values and project.
        
        # Better: use the fact that nodes should be in [0,1] and increasing.
        # Let's parameterize the 8 nodes as cumulative sums of 8 positive numbers.
        # s_i > 0, nodes_i = sum_{j=1}^i s_j / sum_{j=1}^8 s_j
        # This ensures nodes are in (0,1) and increasing, with node_8 = 1.
        # But we want node_0 = 0 too? The integral is from 0 to 1.
        # Actually, the nodes don't have to include 0 and 1 necessarily,
        # but for a quadrature of integral_0^1, it's often good to have some nodes near 0 and 1.
        
        # Let's use a different approach: optimize the 8 node positions directly
        # with the constraint 0 < n_1 < n_2 < ... < n_8 < 1.
        # And 8 weights > 0, sum to 1.
        
        # Parameterization:
        # params[0:8] = log of raw node positions (to ensure positive)
        # params[8:16] = log of raw weights (to ensure positive)
        
        # But this is complex. Let's use a simpler approach:
        # Use scipy.optimize with bounds.
        
        # For the objective function, we'll take 16 parameters:
        # First 8: node positions in [0,1] (we'll sort them)
        # Last 8: log-weights (to ensure positive)
        
        n_params = 16
        if len(params) != n_params:
            return 1e10
        
        node_raw = params[0:8]
        log_w = params[8:16]
        
        # Sort nodes to ensure increasing order
        nodes = np.sort(node_raw)
        
        # Ensure nodes are in [0,1]
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        
        # Weights from log
        w = np.exp(log_w)
        w = w / np.sum(w)  # Normalize to sum to 1
        
        # Compute worst-case error over a subset of cosmologies and redshifts
        # For speed, use a subset during optimization
        max_err = 0.0
        for om, w0, wa in cosmologies[:5]:  # Use subset
            for z in redshifts[:5]:  # Use subset
                err = quadrature_error(nodes, w, z, om, w0, wa)
                if err > max_err:
                    max_err = err
        
        return max_err
    
    # Initial guess: Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x_gl + 1.0)
    w_gl = 0.5 * w_gl
    
    # Parameterize: nodes directly, log-weights
    p0 = np.concatenate([nodes_gl, np.log(w_gl)])
    
    # Bounds
    bounds = []
    for i in range(8):
        bounds.append((1e-10, 1.0 - 1e-10))  # Nodes
    for i in range(8):
        bounds.append((-10, 10))  # Log-weights
    
    # Optimize using differential evolution (global) then Powell (local)
    best_params = p0.copy()
    best_obj = objective(p0)
    
    # Step 1: Differential evolution
    try:
        if time.time() - start_time < time_limit - 30:
            de_res = differential_evolution(
                objective, bounds,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False  # We'll do our own polish
            )
            if de_res.fun < best_obj:
                best_obj = de_res.fun
                best_params = de_res.x
    except Exception:
        pass
    
    # Step 2: Powell refinement
    try:
        if time.time() - start_time < time_limit - 10:
            pow_res = minimize(
                objective, best_params,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            if pow_res.fun < best_obj:
                best_obj = pow_res.fun
                best_params = pow_res.x
    except Exception:
        pass
    
    # Extract final nodes and weights
    nodes_final = np.sort(best_params[0:8])
    nodes_final = np.clip(nodes_final, 1e-10, 1.0 - 1e-10)
    w_final = np.exp(best_params[8:16])
    w_final = w_final / np.sum(w_final)
    
    # Ensure strict increasing order (should already be sorted)
    # And ensure weights are positive (should be from exp)
    
    # Check time and return
    return {"nodes": nodes_final, "weights": w_final}
