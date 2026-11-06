"""Improved quadrature rule discovery for luminosity distance."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_T0 = time.time()
_DEADLINE = _T0 + 85.0

_OM = 0.3
_ODE = 1.0 - _OM

def _E_of_a(a, w0, wa):
    """E(a) = H(a)/H0 for flat universe with matter + CPL dark energy."""
    term_m = _OM / (a ** 3)
    exponent = 3.0 * wa * (a - 1.0) + 3.0 * (w0 + wa) * (1.0 - a)
    term_de = _ODE * np.exp(exponent)
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))

def _compute_all_integrals(nodes, z_vals, models):
    """Compute true integrals and quadrature matrices for all test cases.
    
    Returns:
        true_vals: array of shape (n_cases,) - true (1+z)*z*integral
        quad_matrix: array of shape (n_cases, 8) - for each case, weights[i]/E(a_i)
    """
    n_cases = len(z_vals) * len(models)
    true_vals = np.zeros(n_cases)
    quad_matrix = np.zeros((n_cases, 8))
    
    case_idx = 0
    for w0, wa in models:
        for z in z_vals:
            # True value via high-order Gauss-Legendre
            x, w_gl = np.polynomial.legendre.leggauss(64)
            a_pts = 1.0 - z * 0.5 * (x + 1.0)
            E_pts = _E_of_a(a_pts, w0, wa)
            integral = np.sum(w_gl * (z / 2.0) / E_pts)
            true_vals[case_idx] = (1.0 + z) * integral
            
            # Quadrature contributions
            a_i = 1.0 / (1.0 + z * nodes)
            E_i = _E_of_a(a_i, w0, wa)
            quad_matrix[case_idx, :] = (1.0 + z) * z / E_i
            
            case_idx += 1
    
    return true_vals, quad_matrix

def _solve_optimal_weights(true_vals, quad_matrix):
    """Solve LP to minimize worst-case relative error given fixed nodes.
    
    Minimize t subject to:
        |quad_matrix @ w * t_case - true_vals| / true_vals <= t  for all cases
        sum(w) = 1
        w >= 0
    
    This is: minimize t s.t.
        quad_matrix @ w <= true_vals * (1 + t)
        quad_matrix @ w >= true_vals * (1 - t)
        sum(w) = 1
        w >= 0
    """
    n_cases = len(true_vals)
    n_weights = quad_matrix.shape[1]
    
    # Variables: [w_0, ..., w_7, t]
    n_vars = n_weights + 1
    
    # c = [0, 0, ..., 0, 1]
    c = np.zeros(n_vars)
    c[-1] = 1.0
    
    # Inequality constraints: A_ub @ x <= b_ub
    # Row 1: quad_matrix[i] @ w - true_vals[i]*t <= 0  ->  quad_matrix[i] - true_vals[i]*e_t
    # Row 2: -quad_matrix[i] @ w + true_vals[i]*t <= 0 -> -quad_matrix[i] + true_vals[i]*e_t
    
    A_ub = np.zeros((2 * n_cases, n_vars))
    b_ub = np.zeros(2 * n_cases)
    
    for i in range(n_cases):
        A_ub[2*i, :n_weights] = quad_matrix[i]
        A_ub[2*i, -1] = -true_vals[i]
        b_ub[2*i] = 0.0
        
        A_ub[2*i+1, :n_weights] = -quad_matrix[i]
        A_ub[2*i+1, -1] = true_vals[i]
        b_ub[2*i+1] = 0.0
    
    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_weights] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 0, t >= 0
    bounds = [(0, None)] * n_weights + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        if res.success:
            w_opt = res.x[:n_weights]
            t_opt = res.x[-1]
            w_opt = np.maximum(w_opt, 1e-15)
            w_opt = w_opt / np.sum(w_opt)
            return w_opt, t_opt
    except Exception:
        pass
    
    # Fallback: uniform weights
    return np.ones(n_weights) / n_weights, np.inf

def _worst_err_direct(nodes, weights, z_vals, models):
    """Compute worst relative error directly."""
    true_vals, quad_matrix = _compute_all_integrals(nodes, z_vals, models)
    quad_vals = quad_matrix @ weights
    rel_errs = np.abs(quad_vals - true_vals) / np.maximum(np.abs(true_vals), 1e-30)
    return np.max(rel_errs)

def _node_refine(nodes, weights, z_vals, models, step=0.01):
    """Try small perturbations to nodes and re-solve for weights."""
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    true_vals, quad_matrix = _compute_all_integrals(nodes, z_vals, models)
    current_err = np.max(np.abs(quad_matrix @ weights - true_vals) / 
                         np.maximum(np.abs(true_vals), 1e-30))
    
    for _ in range(3):
        improved = False
        for i in range(8):
            for delta in [step, -step]:
                trial_nodes = nodes.copy()
                trial_nodes[i] += delta
                # Ensure strictly increasing in (0,1)
                if trial_nodes[0] <= 0 or trial_nodes[-1] >= 1:
                    continue
                if np.any(np.diff(trial_nodes) <= 0):
                    continue
                
                trial_tv, trial_qm = _compute_all_integrals(trial_nodes, z_vals, models)
                tw, tt = _solve_optimal_weights(trial_tv, trial_qm)
                trial_err = tt
                if trial_err < current_err - 1e-15:
                    current_err = trial_err
                    nodes = trial_nodes.copy()
                    weights = tw.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5
            if step < 1e-5:
                break
        best_nodes = nodes.copy()
        best_weights = weights.copy()
    
    return best_nodes, best_weights

def construct():
    t_start = time.time()
    
    # Test set: diverse z and models
    z_test = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    models_test = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5),
                   (0.0, -0.5), (0.5, 0.0), (-1.0, 0.5), (-0.5, -0.5)]
    
    # Starting node configurations
    starting_points = []
    
    # 1. Gauss-Legendre mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    starting_points.append(0.5 * (gl_nodes + 1.0))
    
    # 2. Chebyshev nodes
    k = np.arange(8)
    starting_points.append(0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8)))
    
    # 3. Equally spaced
    starting_points.append((np.arange(8) + 0.5) / 8.0)
    
    # 4. Biased toward 0 (smaller z region)
    starting_points.append(np.array([0.03, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]))
    
    # 5. Biased toward 1 (larger z region)
    starting_points.append(np.array([0.05, 0.15, 0.3, 0.45, 0.6, 0.75, 0.88, 0.97]))
    
    # 6. Log-spaced
    starting_points.append(np.geomspace(0.02, 0.98, 8))
    
    # 7. Another mix
    starting_points.append(np.array([0.04, 0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 0.96]))
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    for start_nodes in starting_points:
        if time.time() > _DEADLINE - 15.0:
            break
        
        start_nodes = np.sort(np.clip(start_nodes, 1e-6, 1.0 - 1e-6))
        if np.any(np.diff(start_nodes) <= 0):
            continue
        
        # Solve for optimal weights
        tv, qm = _compute_all_integrals(start_nodes, z_test, models_test)
        w_opt, t_opt = _solve_optimal_weights(tv, qm)
        
        if t_opt < best_err:
            best_err = t_opt
            best_nodes = start_nodes.copy()
            best_weights = w_opt.copy()
    
    # Node refinement from best
    if time.time() < _DEADLINE - 10.0 and best_nodes is not None:
        refined_nodes, refined_weights = _node_refine(
            best_nodes.copy(), best_weights.copy(), z_test, models_test, step=0.02)
        
        tv, qm = _compute_all_integrals(refined_nodes, z_test, models_test)
        w_opt, t_opt = _solve_optimal_weights(tv, qm)
        if t_opt < best_err:
            best_err = t_opt
            best_nodes = refined_nodes.copy()
            best_weights = w_opt.copy()
    
    # Second refinement round with smaller steps
    if time.time() < _DEADLINE - 5.0 and best_nodes is not None:
        refined_nodes, refined_weights = _node_refine(
            best_nodes.copy(), best_weights.copy(), z_test, models_test, step=0.005)
        
        tv, qm = _compute_all_integrals(refined_nodes, z_test, models_test)
        w_opt, t_opt = _solve_optimal_weights(tv, qm)
        if t_opt < best_err:
            best_err = t_opt
            best_nodes = refined_nodes.copy()
            best_weights = w_opt.copy()
    
    # Final validation
    if time.time() < _DEADLINE - 2.0 and best_nodes is not None:
        z_final = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
        models_final = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (-0.5, 0.5),
                       (0.0, -0.5), (0.5, 0.0), (-1.0, 0.5), (-0.5, -0.5),
                       (1.0, 0.0), (-1.5, 0.0)]
        
        final_err = _worst_err_direct(best_nodes, best_weights, z_final, models_final)
        
        if final_err > best_err * 2.0:
            # Fallback: re-solve with expanded set
            tv, qm = _compute_all_integrals(best_nodes, z_final, models_final)
            w_opt, t_opt = _solve_optimal_weights(tv, qm)
            if t_opt < final_err:
                best_weights = w_opt.copy()
    
    # Ensure final constraints
    if best_nodes is None:
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
        best_nodes = 0.5 * (gl_nodes + 1.0)
        best_weights = 0.5 * gl_weights
    
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure strictly increasing and in (0,1)
    for i in range(7):
        if best_nodes[i+1] <= best_nodes[i]:
            best_nodes[i+1] = best_nodes[i] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
