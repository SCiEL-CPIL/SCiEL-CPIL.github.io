"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.integrate import quad

def construct():
    """
    Construct an 8-node positive quadrature rule using a two-phase approach:
    Phase 1: LP-based minimax weight optimization with fixed Gauss-Legendre nodes
    Phase 2: Joint node-weight optimization to refine
    
    Strategy:
    1. Fix nodes to Gauss-Legendre on [0,1]
    2. Solve a linear program to find weights minimizing worst-case error
    3. Use LP solution as starting point for joint optimization
    4. Return best result found within time budget
    """
    n = 8
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s limit
    
    def E_z(z, Omega_m, w0, wa):
        """H(z)/H0 for flat CPL model"""
        Omega_de = 1.0 - Omega_m
        z = np.asarray(z, dtype=float)
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_de * (1.0 + z)**(3.0*(1.0 + w0 + wa*z))
        return np.sqrt(term_m + term_de)
    
    def exact_integral(z, Omega_m, w0, wa):
        """Exact value of integral_0^1 1/E(1/(1+z*t)) dt"""
        def f(t):
            a = 1.0 / (1.0 + z * t)
            return 1.0 / E_z(z * t, Omega_m, w0, wa)
        result, _ = quad(f, 0, 1, limit=200)
        return result
    
    def quadrature_value(nodes, weights, z, Omega_m, w0, wa):
        """Quadrature approximation of the integral"""
        z_nodes = z * nodes
        E_vals = E_z(z_nodes, Omega_m, w0, wa)
        return np.sum(weights / E_vals)
    
    # Define test cosmologies - diverse set covering parameter space
    test_cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.5),    # CPL positive wa
        (0.3, -1.0, -0.5),   # CPL negative wa
        (0.2, -1.0, 0.0),    # Low Omega_m
        (0.4, -1.0, 0.0),    # High Omega_m
        (0.3, -0.8, 0.0),    # w0 > -1
        (0.3, -1.2, 0.0),    # w0 < -1
        (0.3, -1.0, 1.0),    # Large positive wa
        (0.3, -1.0, -1.0),   # Large negative wa
        (0.25, -0.9, 0.3),   # Mixed
        (0.35, -1.1, -0.2),  # Mixed
        (0.3, -0.5, 0.5),    # Quintessence-like
        (0.3, -1.5, -0.3),   # Phantom-like
        (0.2, -0.9, 0.2),    # Low Omega_m, varying w
        (0.4, -1.1, -0.3),   # High Omega_m, varying w
    ]
    
    # Test redshifts - log-spaced
    z_test_values = np.logspace(-2, 1.2, 20)  # z from 0.01 to ~16
    
    # Precompute exact integrals for all test cases
    print("Precomputing exact integrals...")
    exact_values = {}
    for i, (Om, w0, wa) in enumerate(test_cosmologies):
        for z in z_test_values:
            key = (i, z)
            exact_values[key] = exact_integral(z, Om, w0, wa)
    
    # Get Gauss-Legendre nodes on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    nodes_gl = (x_gl + 1.0) / 2.0
    weights_gl = w_gl / 2.0
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Phase 1: LP-based minimax weight optimization with fixed GL nodes
    print("Phase 1: LP-based weight optimization...")
    
    # Build constraint matrix for LP
    # Variables: w_1, ..., w_8, t (8+1 = 9 variables)
    # Minimize: t
    # Constraints: |w @ f_j - I_j| <= t for each test case j
    #              sum(w_i) = 1
    #              w_i >= eps
    
    eps = 1e-12
    num_test_cases = len(test_cosmologies) * len(z_test_values)
    
    # Precompute integrand values at GL nodes for all test cases
    F = np.zeros((num_test_cases, n))
    I = np.zeros(num_test_cases)
    
    idx = 0
    for i, (Om, w0, wa) in enumerate(test_cosmologies):
        for z in z_test_values:
            key = (i, z)
            I[idx] = exact_values[key]
            z_nodes = z * nodes_gl
            E_vals = E_z(z_nodes, Om, w0, wa)
            F[idx, :] = 1.0 / E_vals
            idx += 1
    
    # Set up LP: minimize t
    # s.t. F @ w - t <= I  and  -F @ w - t <= -I
    #      sum(w) = 1
    #      w >= eps, t >= 0
    
    c = np.zeros(n + 1)
    c[n] = 1.0  # minimize t
    
    A_ub = np.zeros((2 * num_test_cases, n + 1))
    b_ub = np.zeros(2 * num_test_cases)
    
    for j in range(num_test_cases):
        A_ub[2*j, :n] = F[j, :]
        A_ub[2*j, n] = -1.0
        b_ub[2*j] = I[j]
        
        A_ub[2*j+1, :n] = -F[j, :]
        A_ub[2*j+1, n] = -1.0
        b_ub[2*j+1] = -I[j]
    
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(eps, None)] * n + [(0.0, None)]
    
    print(f"Solving LP with {num_test_cases} constraints, {n+1} variables...")
    weights_lp = None
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
        if result.success:
            weights_lp = result.x[:n]
            t_opt = result.x[n]
            # Normalize weights to sum exactly to 1
            weights_lp = weights_lp / np.sum(weights_lp)
            print(f"LP solved. Optimal t = {t_opt:.8f}")
        else:
            print(f"LP failed: {result.message}")
    except Exception as e:
        print(f"LP exception: {e}")
    
    if weights_lp is None:
        weights_lp = weights_gl.copy()
    
    # Compute error for LP solution
    def worst_case_error(nodes, weights):
        """Compute worst-case relative error in percent"""
        max_err = 0.0
        for i, (Om, w0, wa) in enumerate(test_cosmologies):
            for z in z_test_values:
                key = (i, z)
                exact = exact_values[key]
                if abs(exact) < 1e-15:
                    continue
                quad_val = quadrature_value(nodes, weights, z, Om, w0, wa)
                rel_err = abs(quad_val - exact) / abs(exact)
                if rel_err > max_err:
                    max_err = rel_err
        return max_err * 100.0
    
    err_lp = worst_case_error(nodes_gl, weights_lp)
    print(f"LP solution error: {err_lp:.6f}%")
    
    # Phase 2: Joint node-weight optimization
    print("Phase 2: Joint node-weight optimization...")
    
    def worst_case_error_params(params):
        """Compute worst-case relative error from parameter vector"""
        if time.time() > deadline:
            return np.inf
        
        node_logits = params[:n]
        weight_logs = params[n:2*n]
        
        nodes = 1.0 / (1.0 + np.exp(-node_logits))
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weight_logs = weight_logs[sort_idx]
        
        if np.any(np.diff(nodes) <= 1e-8):
            return np.inf
        
        if np.any(nodes <= 1e-6) or np.any(nodes >= 1 - 1e-6):
            return np.inf
        
        weights = np.exp(weight_logs)
        weights = weights / np.sum(weights)
        
        return worst_case_error(nodes, weights)
    
    # Initial guess: use GL nodes and LP weights
    init_node_logits = np.log(nodes_gl / (1.0 - nodes_gl))
    init_weight_logs = np.log(weights_lp)
    init_params = np.concatenate([init_node_logits, init_weight_logs])
    
    best_params = init_params.copy()
    best_error = err_lp
    
    # Try Nelder-Mead optimization
    if time.time() < deadline - 10:
        try:
            print("Running Nelder-Mead...")
            result = minimize(
                worst_case_error_params,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
                print(f"  Nelder-Mead improved to: {best_error:.6f}%")
            else:
                print(f"  Nelder-Mead did not improve: {result.fun:.6f}%")
                
        except Exception as e:
            print(f"  Nelder-Mead failed: {e}")
    
    # Try Powell optimization
    if time.time() < deadline - 10:
        try:
            print("Running Powell...")
            result = minimize(
                worst_case_error_params,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_params = result.x
                print(f"  Powell improved to: {best_error:.6f}%")
            else:
                print(f"  Powell did not improve: {result.fun:.6f}%")
                
        except Exception as e:
            print(f"  Powell failed: {e}")
    
    # Extract final nodes and weights
    node_logits = best_params[:n]
    weight_logs = best_params[n:2*n]
    
    nodes = 1.0 / (1.0 + np.exp(-node_logits))
    sort_idx = np.argsort(nodes)
    nodes = nodes[sort_idx]
    weight_logs = weight_logs[sort_idx]
    
    weights = np.exp(weight_logs)
    weights = weights / np.sum(weights)
    
    # Final validation and fallback
    valid = True
    if not (np.all(nodes > 1e-6) and np.all(nodes < 1 - 1e-6)):
        valid = False
    if not (np.all(np.diff(nodes) > 1e-8)):
        valid = False
    if not np.all(weights > 0):
        valid = False
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / np.sum(weights)
    
    if not valid:
        print("Falling back to GL with LP weights")
        nodes = nodes_gl.copy()
        weights = weights_lp.copy()
        weights = weights / np.sum(weights)
    
    # Ensure weights sum to 1 within tolerance
    weights = weights / np.sum(weights)
    
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s, final error: {best_error:.6f}%")
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
