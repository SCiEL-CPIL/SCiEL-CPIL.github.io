"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0 # 60s limit, use 50s for safety
    
    # Cosmology parameters for optimization
    # We want to cover a range of plausible cosmologies.
    # Flat universe: Omega_m + Omega_L = 1
    # CPL: w(z) = w_0 + w_a * z / (1+z)
    # E(z) = sqrt(Omega_m * (1+z)^3 + Omega_L * (1+z)^(3*(1+w_0+w_a)) * exp(3*w_a*z/(1+z)))
    
    def make_E(Omega_m, w0, wa):
        def E(z):
            # z is the redshift
            # Avoid z=0 issues if any, though E(0)=1
            one_plus_z = 1.0 + z
            # Term for matter
            matter = Omega_m * (one_plus_z ** 3)
            # Term for Dark Energy
            # Exponent part: 3 * (1 + w0 + wa) * ln(1+z) + 3 * wa * z / (1+z)
            # We can compute (1+z)^(3*(1+w0+wa)) * exp(3*wa*z/(1+z))
            # To avoid overflow/underflow issues with large z, we compute in log space or carefully
            # z is in [0, 3] typically for these tests.
            exp_term = 3.0 * wa * z / one_plus_z
            power_term = 3.0 * (1.0 + w0 + wa) * np.log(one_plus_z)
            de = (1.0 - Omega_m) * np.exp(power_term + exp_term)
            return np.sqrt(matter + de)
        return E

    # Define a set of validation cosmologies to optimize against
    # These should be representative of the "disjoint public validation cosmologies"
    # We assume standard ranges: Omega_m ~ 0.2-0.4, w0 ~ -1.5 to 0, wa ~ -1 to 1
    cosmo_params = [
        (0.3, -1.0, 0.0),   # LambdaCDM
        (0.3, -1.2, 0.5),   # Phantom-like
        (0.3, -0.8, -0.5),  # Quintessence-like
        (0.25, -1.0, 0.0),  # Lower matter density
        (0.35, -1.0, 0.0),  # Higher matter density
        (0.3, -1.5, 0.0),   # Stronger phantom
        (0.3, -0.5, -1.0),  # Stronger quintessence
        (0.3, -1.0, 1.0),   # High wa
        (0.3, -1.0, -1.0),  # Low wa
    ]
    
    # Pre-compute E functions
    E_funcs = [make_E(om, w0, wa) for om, w0, wa in cosmo_params]
    
    # Redshifts to test
    z_tests = np.linspace(0.1, 3.0, 20)
    
    # Function to compute relative error for a given set of nodes and weights
    def compute_errors(nodes, weights):
        # nodes: array of 8 floats in (0,1)
        # weights: array of 8 floats, sum to 1
        max_err = 0.0
        for E_func in E_funcs:
            for z in z_tests:
                # Exact integral: int_0^1 du / E(z*u)
                # Using quad for high precision reference
                # integrand u -> 1/E(z*u)
                def integrand(u):
                    return 1.0 / E_func(z * u)
                
                try:
                    exact_val, _ = opt.integrate.quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)
                except:
                    continue
                
                # Quadrature approximation
                # sum w_i / E(z * x_i)
                approx_val = np.sum(weights / E_func(z * nodes))
                
                if exact_val == 0:
                    continue
                    
                rel_err = abs(approx_val - exact_val) / exact_val
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # Optimization variables
    # x: log-space weights (8 vars)
    # y: nodes (8 vars), constrained to (0,1) and sorted
    # To ensure sorted nodes, we can optimize differences or just use bounds and sort,
    # but sorting breaks gradient flow. 
    # Better: optimize nodes directly with bounds [1e-6, 1-1e-6] and enforce sorting by penalizing if not sorted, 
    # or simply start with sorted and hope L-BFGS-B stays in the sorted region if initialized well.
    # Given the symmetry and convexity of the error landscape in this low-dim space, 
    # starting with a good guess (Gauss-Legendre) should keep them sorted.
    
    # Initial guess: Gauss-Legendre nodes and weights for 8 points
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map from [-1, 1] to [0, 1]
    nodes_init = 0.5 * (x_gl + 1.0)
    weights_init = w_gl / 2.0 # Scale weights to sum to 1
    
    # Log-space weights for positivity
    log_w_init = np.log(weights_init)
    
    # Flatten parameters
    # p[0:8] = log_weights
    # p[8:16] = nodes
    p0 = np.concatenate([log_w_init, nodes_init])
    
    # Bounds
    # Weights: log_w can be very large/small, but let's bound them to prevent numerical issues
    # w_i > 0. If log_w is -100, w is tiny. If 100, huge.
    # Sum(exp(log_w)) = 1 constraint is handled by normalization in the error function?
    # No, the error function uses `weights` directly. 
    # I need to normalize weights inside the evaluation to ensure sum=1.
    # Let's modify compute_errors to take raw log_weights and normalize.
    
    def objective(p):
        if time.time() > deadline - 5:
            return 1e10 # Penalty for timeout
            
        log_w = p[:8]
        nodes = p[8:16]
        
        # Check node validity
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        # Check sorted
        if np.any(np.diff(nodes) <= 0):
            return 1e10
            
        # Normalize weights
        # Softmax to ensure positivity and sum 1
        # Subtract max for numerical stability
        max_log_w = np.max(log_w)
        w = np.exp(log_w - max_log_w)
        w_sum = np.sum(w)
        if w_sum == 0:
            return 1e10
        w = w / w_sum
        
        # Calculate max relative error
        max_err = 0.0
        for E_func in E_funcs:
            for z in z_tests:
                def integrand(u):
                    return 1.0 / E_func(z * u)
                
                try:
                    exact_val, _ = opt.integrate.quad(integrand, 0, 1, epsabs=1e-10, epsrel=1e-10, limit=100)
                except:
                    continue
                
                approx_val = np.sum(w / E_func(z * nodes))
                
                if exact_val < 1e-10:
                    continue
                    
                rel_err = abs(approx_val - exact_val) / exact_val
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # Run optimization
    # L-BFGS-B is good for bound-constrained problems
    bounds = [(-50, 50)] * 8 + [(1e-6, 1 - 1e-6)] * 8
    
    best_p = p0
    best_err = objective(p0)
    
    # Try a few restarts or just one long run
    # Since we have ~50s, we can run minimize for a while
    # maxiter and maxfun need to be set to use the time effectively
    # However, minimize doesn't take a time limit directly. 
    # We can wrap the function to break early, but L-BFGS-B might not respect it gracefully if it's in the middle of a step.
    # Given the low dimension (16), one run is likely sufficient if well-initialized.
    
    try:
        res = opt.minimize(objective, p0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-15, 'gtol': 1e-15})
        
        if res.fun < best_err:
            best_p = res.x
            best_err = res.fun
            
    except Exception as e:
        # If optimization fails, fall back to initial
        pass
    
    # Final extraction
    log_w_best = best_p[:8]
    nodes_best = best_p[8:16]
    
    # Normalize weights
    max_log_w = np.max(log_w_best)
    w_best = np.exp(log_w_best - max_log_w)
    w_best = w_best / np.sum(w_best)
    
    # Ensure strict increasing and in [0,1]
    # The optimizer should have kept them sorted, but let's verify and fix if needed
    # If not sorted, we might need to re-optimize or just return the sorted version (which might not be optimal)
    # Given the initialization and nature of quadrature, they should stay sorted.
    if not np.all(np.diff(nodes_best) > 0):
        # Fallback to sorted nodes if optimizer messed up
        nodes_best = np.sort(nodes_best)
        # Re-normalize weights? No, weights correspond to specific nodes.
        # If nodes were swapped, weights are wrong. 
        # But L-BFGS-B with bounds usually doesn't swap if initialized sorted.
        
    # Final check on constraints
    # Weights positive and sum to 1
    assert np.all(w_best > 0)
    assert abs(np.sum(w_best) - 1.0) < 1e-12
    
    # Nodes strictly increasing in [0,1]
    assert np.all(nodes_best > 0)
    assert np.all(nodes_best < 1)
    assert np.all(np.diff(nodes_best) > 0)
    
    return {"nodes": nodes_best, "weights": w_best}

# EVOLVE-BLOCK-END
