"""
Strategy: Jointly optimize nodes and weights using a two-stage approach with a richer
design space. Stage 1: use a diverse set of candidate node configurations (including
Gauss-Legendre, Gauss-Lobatto, Chebyshev, and custom spacings) and for each, solve
a minimax LP for weights. Stage 2: refine the best node configuration using Nelder-Mead
on the minimax error (re-solving LP at each iteration). This explores the joint
(node, weight) space more thoroughly than fixing nodes a priori.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    a_safe = np.maximum(a, 1e-10)
    z = 1.0 / a_safe - 1.0
    h2 = om_m * (1 + z)**3 + om_r * (1 + z)**4 + om_k + \
         (1 - om_m - om_r - om_k) * (1 + z)**(3 * (1 + w0 + wa * z / (1 + z)))
    return np.sqrt(np.maximum(h2, 1e-30))

def _true_dL_integral(z, w0, wa, om_m, om_r, om_k, n=4001):
    """Compute the true integral value using high-resolution Simpson's rule."""
    x = np.linspace(0, 1, n)
    a_vals = 1.0 / (1.0 + z * x)
    E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
    y = 1.0 / E_vals
    
    h = 1.0 / (n - 1)
    integral = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return integral

def _solve_minimax_lp(A, true_vals):
    """Solve minimax LP: minimize max_i |A_i . w - true_vals_i| s.t. sum(w)=1, w>=0."""
    n_cases, n_nodes = A.shape
    
    # Variables: w_0 ... w_{n_nodes-1}, t
    c = np.zeros(n_nodes + 1)
    c[n_nodes] = 1.0  # Minimize t
    
    A_ub = np.zeros((2 * n_cases, n_nodes + 1))
    b_ub = np.zeros(2 * n_cases)
    
    for i in range(n_cases):
        A_ub[2*i, :n_nodes] = A[i, :]
        A_ub[2*i, n_nodes] = -1.0
        b_ub[2*i] = true_vals[i]
        
        A_ub[2*i+1, :n_nodes] = -A[i, :]
        A_ub[2*i+1, n_nodes] = -1.0
        b_ub[2*i+1] = -true_vals[i]
    
    A_eq = np.zeros((1, n_nodes + 1))
    A_eq[0, :n_nodes] = 1.0
    b_eq = np.array([1.0])
    
    bounds = [(0, None)] * n_nodes + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            weights = res.x[:n_nodes]
            if np.any(weights <= 0):
                weights = weights + 1e-8
                weights = weights / np.sum(weights)
            else:
                weights = weights / np.sum(weights)
            return weights
        else:
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Define test set: diverse CPL cosmologies and redshifts
    models = [
        (0.0, 0.0, 0.3, 0.05, 0.65),   # Lambda CDM
        (-0.5, 0.5, 0.3, 0.05, 0.65),  # Quintessence
        (-1.5, 0.5, 0.3, 0.05, 0.65),  # Phantom
        (-1.0, 1.0, 0.2, 0.05, 0.75),  # Varying DE
        (0.5, -0.5, 0.4, 0.05, 0.55),  # Decelerating early
        (-0.9, 0.3, 0.3, 0.0, 0.70),   # No radiation
        (-1.1, 0.0, 0.25, 0.05, 0.70), # More DE
        (0.0, 1.0, 0.3, 0.05, 0.65),   # Strong wa
        (-1.0, -0.5, 0.3, 0.05, 0.65), # Negative wa
        (0.2, 0.3, 0.35, 0.05, 0.60),  # Matter-dominated-ish
    ]
    
    # Test redshifts - broader range
    z_vals = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
    
    # Generate test cases
    test_cases = []
    for m in models:
        for z in z_vals:
            test_cases.append((m, z))
            
    n_cases = len(test_cases)
    
    # Pre-compute true values
    true_vals = np.zeros(n_cases)
    for i, (m, z) in enumerate(test_cases):
        w0, wa, om_m, om_r, om_k = m
        true_vals[i] = _true_dL_integral(z, w0, wa, om_m, om_r, om_k)
    
    # Generate candidate node configurations
    def _get_candidate_nodes():
        candidates = []
        
        # 1. Gauss-Legendre
        x_gl, _ = np.polynomial.legendre.leggauss(8)
        nodes_gl = (x_gl + 1.0) / 2.0
        candidates.append(nodes_gl)
        
        # 2. Gauss-Lobatto (includes endpoints, but we need interior points)
        # Use Radau-like or just shifted Legendre
        # Actually, let's use different polynomial bases
        
        # 3. Chebyshev nodes (first kind)
        k = np.arange(8)
        nodes_cheb = (1 + np.cos(np.pi * (2*k + 1) / 16)) / 2.0
        nodes_cheb = np.sort(nodes_cheb)
        candidates.append(nodes_cheb)
        
        # 4. Chebyshev nodes (second kind) - excluding endpoints
        k = np.arange(8)
        nodes_cheb2 = (1 + np.cos(np.pi * k / 7)) / 2.0
        nodes_cheb2 = np.sort(nodes_cheb2)
        # Remove duplicates if any, ensure strictly in (0,1)
        nodes_cheb2 = nodes_cheb2[(nodes_cheb2 > 0.001) & (nodes_cheb2 < 0.999)]
        if len(nodes_cheb2) == 8:
            candidates.append(nodes_cheb2)
        else:
            # Fallback: evenly spaced
            candidates.append(np.linspace(0.05, 0.95, 8))
        
        # 5. Evenly spaced
        nodes_even = np.linspace(1/17, 16/17, 8)
        candidates.append(nodes_even)
        
        # 6. Log-spaced (emphasize low z)
        nodes_log = np.logspace(np.log10(0.02), np.log10(0.98), 8)
        candidates.append(nodes_log)
        
        # 7. Power-law spaced (emphasize mid-range)
        nodes_pow = (np.arange(1, 9) ** 0.8) / (8 ** 0.8)
        nodes_pow = nodes_pow * 0.98 + 0.01
        candidates.append(nodes_pow)
        
        # 8. Custom: cluster around z~1 (a~0.5)
        nodes_custom = np.array([0.05, 0.15, 0.3, 0.45, 0.55, 0.7, 0.85, 0.95])
        candidates.append(nodes_custom)
        
        # 9. Another custom: more weight on high z
        nodes_high_z = np.array([0.03, 0.08, 0.15, 0.25, 0.4, 0.6, 0.8, 0.97])
        candidates.append(nodes_high_z)
        
        # 10. Another custom: more weight on low z
        nodes_low_z = np.array([0.05, 0.12, 0.2, 0.3, 0.45, 0.65, 0.82, 0.95])
        candidates.append(nodes_low_z)
        
        return candidates
    
    candidates = _get_candidate_nodes()
    
    # Evaluate each candidate node configuration
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    for nodes in candidates:
        if time.time() > deadline - 10:
            break
        
        # Ensure nodes are strictly increasing in (0,1)
        if not np.all(np.diff(nodes) > 0) or not (nodes[0] > 0 and nodes[-1] < 1):
            continue
        
        # Pre-compute A matrix
        A = np.zeros((n_cases, 8))
        for i, (m, z) in enumerate(test_cases):
            w0, wa, om_m, om_r, om_k = m
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
            A[i, :] = 1.0 / E_vals
        
        # Solve minimax LP
        weights = _solve_minimax_lp(A, true_vals)
        if weights is None:
            continue
        
        # Compute error
        quad_vals = A @ weights
        rel_errors = np.abs(quad_vals - true_vals) / true_vals
        max_err = np.max(rel_errors)
        
        if max_err < best_err:
            best_err = max_err
            best_nodes = nodes.copy()
            best_weights = weights.copy()
    
    # If we have time, refine the best node configuration using Nelder-Mead
    if best_nodes is not None and time.time() < deadline - 15:
        def _node_loss(node_vec):
            # Ensure valid nodes
            if not np.all(np.diff(node_vec) > 0) or not (node_vec[0] > 0.001 and node_vec[-1] < 0.999):
                return 1.0
            
            # Pre-compute A matrix
            A = np.zeros((n_cases, 8))
            for i, (m, z) in enumerate(test_cases):
                w0, wa, om_m, om_r, om_k = m
                a_vals = 1.0 / (1.0 + z * node_vec)
                E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
                A[i, :] = 1.0 / E_vals
            
            # Solve minimax LP
            weights = _solve_minimax_lp(A, true_vals)
            if weights is None:
                return 1.0
            
            # Compute error
            quad_vals = A @ weights
            rel_errors = np.abs(quad_vals - true_vals) / true_vals
            return np.max(rel_errors)
        
        # Initial guess: best nodes found
        x0 = best_nodes.copy()
        
        # Use Nelder-Mead with bounds handled by penalty
        try:
            res = minimize(_node_loss, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-8, 'disp': False})
            
            if res.success and res.fun < best_err:
                # Verify the refined nodes
                refined_nodes = res.x
                if np.all(np.diff(refined_nodes) > 0) and (refined_nodes[0] > 0.001 and refined_nodes[-1] < 0.999):
                    # Recompute weights for the refined nodes
                    A = np.zeros((n_cases, 8))
                    for i, (m, z) in enumerate(test_cases):
                        w0, wa, om_m, om_r, om_k = m
                        a_vals = 1.0 / (1.0 + z * refined_nodes)
                        E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
                        A[i, :] = 1.0 / E_vals
                    
                    weights = _solve_minimax_lp(A, true_vals)
                    if weights is not None:
                        quad_vals = A @ weights
                        rel_errors = np.abs(quad_vals - true_vals) / true_vals
                        max_err = np.max(rel_errors)
                        if max_err < best_err:
                            best_err = max_err
                            best_nodes = refined_nodes.copy()
                            best_weights = weights.copy()
        except Exception:
            pass
    
    # Fallback if nothing worked
    if best_nodes is None:
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        best_nodes = (x_gl + 1.0) / 2.0
        best_weights = w_gl / 2.0
    
    # Final sanity check
    if not np.all(best_nodes > 0) or not np.all(best_nodes < 1):
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        best_nodes = (x_gl + 1.0) / 2.0
        best_weights = w_gl / 2.0
        
    if not np.all(np.diff(best_nodes) > 0):
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        best_nodes = (x_gl + 1.0) / 2.0
        best_weights = w_gl / 2.0
        
    # Normalize weights
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}

# EVOLVE-BLOCK-END
