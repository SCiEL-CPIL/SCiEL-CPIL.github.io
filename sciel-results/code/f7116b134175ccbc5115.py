"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_legendre, roots_chebyshev


def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Start with Chebyshev-Gauss nodes (optimal for spectral approximation on [0,1]).
    2. Solve for optimal weights via linear least-squares (globally optimal for fixed nodes).
    3. Refine both nodes and weights jointly using L-BFGS-B with cumulative-difference
       parameterization to handle the ordering constraint.
    4. Use a broad set of representative CPL cosmologies and redshifts.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before 90s kill
    
    # Representative CPL cosmologies: (Om_m, Om_de, w0, wa)
    models = [
        (0.3, 0.7, -1.0, 0.0),    # LCDM
        (0.3, 0.7, -1.0, 0.5),    # CPL with positive wa
        (0.3, 0.7, -0.8, -0.5),   # CPL with negative wa
        (0.3, 0.7, -0.5, 0.0),    # wCDM
        (0.3, 0.7, -1.2, 0.3),    # CPL with w0 < -1
        (0.3, 0.7, -0.9, 0.2),    # Another CPL
        (0.25, 0.75, -1.0, 0.0),  # Different matter content
        (0.35, 0.65, -1.0, 0.0),  # Different matter content
        (0.3, 0.7, -1.0, -0.3),   # CPL with negative wa
        (0.3, 0.7, -1.5, 0.5),    # Extreme w0
    ]
    
    # Test z values covering the relevant range
    z_vals = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    
    def E_of_z(z, Om, OL, w0, wa):
        """Compute E(z) = H(z)/H0 for CPL model."""
        if abs(wa) < 1e-12 and abs(w0 + 1.0) < 1e-12:
            return np.sqrt(Om * (1 + z)**3 + OL)
        else:
            term1 = Om * (1 + z)**3
            term2 = OL * (1 + z)**(3*(1+w0)) * np.exp(3*wa*z/(1+z))
            return np.sqrt(term1 + term2)
    
    def true_distance(z, Om, OL, w0, wa, n_ref=10001):
        """High-accuracy reference integral using Simpson's rule."""
        if z < 1e-10:
            return 0.0
        u = np.linspace(0, 1, n_ref)
        z_vals_ref = z * u
        E_vals = E_of_z(z_vals_ref, Om, OL, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        integral = h * (integrand[0] + integrand[-1] + 
                        4*np.sum(integrand[1:-1:2]) + 
                        2*np.sum(integrand[2:-1:2]))
        return (1 + z) * z * integral
    
    def quadrature_distance(z, nodes, weights, Om, OL, w0, wa):
        """Compute quadrature approximation of luminosity distance."""
        if z < 1e-10:
            return 0.0
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, Om, OL, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1 + z) * z * integral_approx
    
    def compute_worst_error(nodes, weights):
        """Compute worst-case relative error across all test cases."""
        max_err = 0.0
        for z in z_vals:
            for (Om, OL, w0, wa) in models:
                d_true = true_distance(z, Om, OL, w0, wa)
                if d_true < 1e-12:
                    continue
                d_quad = quadrature_distance(z, nodes, weights, Om, OL, w0, wa)
                rel_err = abs(d_quad - d_true) / d_true
                if rel_err > max_err:
                    max_err = rel_err
                    if max_err > 1.0:  # Early exit if error is very large
                        return max_err
        return max_err
    
    # Step 1: Start with Chebyshev-Gauss nodes mapped to [0,1]
    # Chebyshev-Gauss nodes are roots of T_8(x) in [-1,1]
    cheb_nodes, cheb_weights = roots_chebyshev(8)
    nodes = 0.5 * (cheb_nodes + 1.0)  # Map to [0,1]
    
    # Step 2: Solve for optimal weights via linear least-squares
    # Build reference data
    ref_vals = []
    node_integrands = []  # shape: (n_cases, 8)
    
    for z in z_vals:
        for (Om, OL, w0, wa) in models:
            if time.time() > deadline - 30:
                break
            try:
                rv = true_distance(z, Om, OL, w0, wa)
                if rv <= 1e-12:
                    continue
                
                # Node integrand values: 1/E(z*nodes[i])
                z_nodes = z * nodes
                E_vals = E_of_z(z_nodes, Om, OL, w0, wa)
                node_vals = 1.0 / E_vals
                
                ref_vals.append(rv)
                node_integrands.append(node_vals)
            except Exception:
                continue
        if time.time() > deadline - 30:
            break
    
    if not ref_vals:
        # Fallback to Chebyshev weights
        weights = 0.5 * cheb_weights
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}
    
    ref_vals = np.array(ref_vals)
    node_integrands = np.array(node_integrands)
    
    # Solve linear least-squares: A @ w = b
    try:
        w_sol, residuals, rank, sv = np.linalg.lstsq(node_integrands, ref_vals, rcond=None)
    except Exception:
        w_sol = 0.5 * cheb_weights
    
    # Check if solution is valid (all positive)
    if np.all(w_sol > 1e-10):
        weights = w_sol / np.sum(w_sol)
    else:
        # Fall back to Chebyshev weights
        weights = 0.5 * cheb_weights
        weights = weights / np.sum(weights)
    
    # Step 3: Joint refinement using L-BFGS-B
    # Parameterization: [u1, d1, d2, ..., d7, w1, ..., w8]
    # where u1 > 0, di > 0, u_{i+1} = u_i + d_i, u8 < 1
    # weights are normalized internally
    
    def params_to_nodes_weights(params):
        u1 = params[0]
        ds = params[1:8]
        ws = params[8:16]
        
        nodes_arr = np.zeros(8)
        nodes_arr[0] = u1
        for i in range(1, 8):
            nodes_arr[i] = nodes_arr[i-1] + ds[i-1]
        
        # Normalize weights
        w_sum = np.sum(ws)
        if w_sum <= 1e-10:
            return None, None
        ws_norm = ws / w_sum
        
        # Check validity
        if nodes_arr[0] <= 0 or nodes_arr[-1] >= 1.0 or not np.all(np.diff(nodes_arr) > 0):
            return None, None
        if not np.all(ws_norm > 0):
            return None, None
        
        return nodes_arr, ws_norm
    
    def objective(params):
        if time.time() > deadline - 5:
            return 1e10
        nodes_arr, ws_norm = params_to_nodes_weights(params)
        if nodes_arr is None:
            return 1e10
        return compute_worst_error(nodes_arr, ws_norm)
    
    # Initial parameters from current nodes and weights
    u1 = nodes[0]
    ds = np.diff(nodes)
    ws = weights
    
    init_params = np.concatenate(([u1], ds, ws))
    
    # Bounds
    bounds = [(1e-6, 1.0)]  # u1
    bounds.extend([(1e-6, 1.0)] * 7)  # ds
    bounds.extend([(1e-6, 1.0)] * 8)  # ws
    
    # Optimize
    best_params = init_params.copy()
    best_score = objective(init_params)
    
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', 
                      bounds=bounds, 
                      options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Try a second optimization from a slightly perturbed starting point
    if time.time() < deadline - 20:
        # Perturb initial parameters slightly
        np.random.seed(42)
        perturbed = init_params + np.random.normal(0, 1e-4, size=init_params.shape)
        perturbed[0] = max(1e-6, min(perturbed[0], 1.0))
        for i in range(1, 8):
            perturbed[i] = max(1e-6, min(perturbed[i], 1.0))
        for i in range(8, 16):
            perturbed[i] = max(1e-6, min(perturbed[i], 1.0))
        
        try:
            res2 = minimize(objective, perturbed, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-12})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Extract final nodes and weights
    nodes_final, weights_final = params_to_nodes_weights(best_params)
    
    if nodes_final is None:
        # Fallback to initial Chebyshev solution
        nodes_final = nodes
        weights_final = weights
    
    # Final sanity checks
    if not np.all(np.diff(nodes_final) > 0):
        nodes_final = 0.5 * (roots_chebyshev(8)[0] + 1.0)
        weights_final = 0.5 * roots_chebyshev(8)[1]
    
    if not np.all(weights_final > 0):
        weights_final = 0.5 * roots_chebyshev(8)[1]
    
    weights_final = weights_final / np.sum(weights_final)
    
    # Verify sum is within tolerance
    if abs(np.sum(weights_final) - 1.0) > 1e-12:
        weights_final = 0.5 * roots_chebyshev(8)[1]
        weights_final = weights_final / np.sum(weights_final)
    
    return {"nodes": nodes_final, "weights": weights_final}
# EVOLVE-BLOCK-END
