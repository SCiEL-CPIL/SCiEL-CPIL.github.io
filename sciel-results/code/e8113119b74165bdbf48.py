"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E_of_z(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def _integrand(z, Om0, Ol0, w0, w1):
    """f(z) = z/(1+z) * 1/E(z)"""
    if z < 1e-15:
        return 0.0
    E = _E_of_z(z, Om0, Ol0, w0, w1)
    return z / (1.0 + z) / E

def _true_integral(z_max, Om0, Ol0, w0, w1):
    """High-accuracy adaptive integration for reference."""
    result, _ = quad(lambda z: _integrand(z, Om0, Ol0, w0, w1), 0, z_max, limit=200)
    return result

def _quadrature_at_nodes(nodes, weights, z_max, Om0, Ol0, w0, w1):
    """Compute quadrature approximation for given nodes/weights."""
    z_eval = z_max * nodes
    vals = np.array([_integrand(z_eval[i], Om0, Ol0, w0, w1) for i in range(len(nodes))])
    return z_max * np.dot(weights, vals)

def _solve_weights_for_nodes(nodes, test_data):
    """
    Given fixed nodes, find optimal positive weights minimizing worst-case relative error.
    Uses convex optimization: minimize t s.t. |approx_i - true_i|/true_i <= t, sum(w)=1, w>0
    """
    n = len(nodes)
    m = len(test_data)
    
    # Precompute integrand values at nodes for each test case
    A = np.zeros((m, n))
    true_vals = np.zeros(m)
    for i, (Om0, Ol0, w0, w1, z_max, tv) in enumerate(test_data):
        z_eval = z_max * nodes
        for j in range(n):
            A[i, j] = z_max * _integrand(z_eval[j], Om0, Ol0, w0, w1)
        true_vals[i] = tv
    
    # Filter out zero true values
    mask = np.abs(true_vals) > 1e-15
    A = A[mask]
    true_vals = true_vals[mask]
    m = len(A)
    
    if m == 0:
        return np.ones(n) / n
    
    # Minimize t s.t. -t*true_i <= A_i @ w - true_i <= t*true_i, sum(w)=1, w >= 0
    # Variables: w (n) + t (1) = n+1
    def obj(v):
        return v[-1]  # minimize t
    
    # Constraints: sum(w) = 1, w >= 0, |A@w - true| <= t*true
    from scipy.optimize import LinearConstraint
    
    # Linear constraints:
    # sum(w) = 1: [1,1,...,1,0] @ v = 1
    # w_j >= 0: handled by bounds
    # A_i @ w - t*true_i <= true_i  =>  [A_i, -true_i] @ v <= true_i
    # -(A_i @ w) + t*true_i <= true_i =>  [-A_i, true_i] @ v <= true_i
    
    c_eq = np.zeros((1, n + 1))
    c_eq[0, :n] = 1.0
    
    c_ineq = np.zeros((2 * m, n + 1))
    b_ineq = np.zeros(2 * m)
    for i in range(m):
        c_ineq[2*i, :n] = A[i]
        c_ineq[2*i, n] = -true_vals[i]
        b_ineq[2*i] = true_vals[i]
        
        c_ineq[2*i+1, :n] = -A[i]
        c_ineq[2*i+1, n] = true_vals[i]
        b_ineq[2*i+1] = true_vals[i]
    
    bounds = [(0, None)] * n + [(0, None)]
    
    # Initial guess: equal weights
    v0 = np.zeros(n + 1)
    v0[:n] = 1.0 / n
    v0[n] = 0.1
    
    try:
        result = minimize(obj, v0, method='SLSQP',
                         constraints=[
                             {'type': 'eq', 'fun': lambda v: np.sum(v[:n]) - 1.0,
                              'jac': lambda v: np.concatenate([np.ones(n), [0]])},
                             {'type': 'ineq', 'fun': lambda v: true_vals - (A @ v[:n] - v[n]*true_vals),
                              'jac': lambda v: np.hstack([A, -np.diag(true_vals)]).T},
                             {'type': 'ineq', 'fun': lambda v: true_vals + (A @ v[:n] - v[n]*true_vals),
                              'jac': lambda v: np.hstack([-A, np.diag(true_vals)]).T},
                         ],
                         bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-12})
        if result.success:
            w = np.maximum(result.x[:n], 1e-15)
            w = w / np.sum(w)
            return w
    except Exception:
        pass
    
    # Fallback: simple iterative proportional fitting
    w = np.ones(n) / n
    for _ in range(50):
        errs = np.abs(A @ w - true_vals) / np.abs(true_vals)
        max_err_idx = np.argmax(errs)
        # Adjust weights toward reducing worst error
        grad = np.sign(A[max_err_idx, :] - true_vals[max_err_idx]) * A[max_err_idx] / true_vals[max_err_idx]
        w = w - 0.01 * grad
        w = np.maximum(w, 1e-15)
        w = w / np.sum(w)
    return w

def _eval_worst_error(nodes, weights, test_data):
    """Evaluate worst-case relative error."""
    max_err = 0.0
    for (Om0, Ol0, w0, w1, z_max, tv) in test_data:
        if abs(tv) < 1e-15:
            continue
        approx = _quadrature_at_nodes(nodes, weights, z_max, Om0, Ol0, w0, w1)
        rel_err = abs(approx - tv) / abs(tv) * 100.0
        if rel_err > max_err:
            max_err = rel_err
    return max_err

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s
    
    # Design cosmologies
    design_cosmos = [
        (0.3, 0.7, -1.0, 0.0),
        (0.27, 0.73, -1.0, 0.0),
        (0.3, 0.7, -1.0, 1.0),
        (0.3, 0.7, -1.0, -1.0),
        (0.2, 0.8, -1.2, 0.5),
        (0.35, 0.65, -0.8, 0.3),
        (0.25, 0.75, -1.5, 1.0),
        (0.3, 0.7, -0.5, -0.5),
        (0.3, 0.7, -1.0, 0.5),
        (0.3, 0.7, -1.5, -0.5),
        (0.3, 0.7, -1.0, 0.8),
        (0.3, 0.7, -1.2, -0.8),
        (0.25, 0.75, -0.9, 0.6),
        (0.32, 0.68, -1.1, -0.6),
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals
    test_data = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            tv = _true_integral(z_max, Om0, Ol0, w0, w1)
            test_data.append((Om0, Ol0, w0, w1, z_max, tv))
    
    # Initialize with Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    best_nodes = 0.5 * (gl_nodes + 1.0)
    best_weights = gl_weights.copy()
    best_score = _eval_worst_error(best_nodes, best_weights, test_data)
    
    # Alternating optimization: fix nodes, optimize weights; then update nodes
    for outer_iter in range(15):
        if time.time() > deadline:
            break
        
        # Step 1: Optimize weights for current nodes
        current_weights = _solve_weights_for_nodes(best_nodes, test_data)
        current_score = _eval_worst_error(best_nodes, current_weights, test_data)
        
        if current_score < best_score:
            best_weights = current_weights
            best_score = current_score
        
        # Step 2: Update nodes via subgradient descent on worst-case error
        # Find which test case has worst error
        worst_idx = -1
        worst_err = 0.0
        for i, (Om0, Ol0, w0, w1, z_max, tv) in enumerate(test_data):
            if abs(tv) < 1e-15:
                continue
            approx = _quadrature_at_nodes(best_nodes, best_weights, z_max, Om0, Ol0, w0, w1)
            rel_err = abs(approx - tv) / abs(tv)
            if rel_err > worst_err:
                worst_err = rel_err
                worst_idx = i
        
        if worst_idx < 0:
            break
        
        Om0, Ol0, w0, w1, z_max, tv = test_data[worst_idx]
        sign = np.sign(_quadrature_at_nodes(best_nodes, best_weights, z_max, Om0, Ol0, w0, w1) - tv)
        
        # Subgradient of error w.r.t. nodes
        # d/dx_i [z_max * sum_j w_j f(z_max * x_j)] = z_max^2 * w_i * f'(z_max * x_i)
        # Approximate f' numerically
        h = 1e-6
        grad_nodes = np.zeros(8)
        for i in range(8):
            x_i = best_nodes[i]
            z_plus = z_max * (x_i + h)
            z_minus = z_max * (x_i - h)
            f_plus = _integrand(z_plus, Om0, Ol0, w0, w1) if z_plus > 0 else 0.0
            f_minus = _integrand(z_minus, Om0, Ol0, w0, w1) if z_minus > 0 else 0.0
            df_dz = (f_plus - f_minus) / (2 * h)
            grad_nodes[i] = sign * z_max * z_max * best_weights[i] * df_dz / abs(tv)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_nodes)
        if grad_norm > 1e-15:
            grad_nodes = grad_nodes / grad_norm
        
        # Learning rate schedule
        lr = 0.05 / (1.0 + 0.3 * outer_iter)
        
        # Update nodes
        new_nodes = best_nodes - lr * grad_nodes
        
        # Enforce strictly increasing in (0,1)
        new_nodes = np.clip(new_nodes, 1e-8, 1.0 - 1e-8)
        # Ensure strictly increasing
        for i in range(1, 8):
            if new_nodes[i] <= new_nodes[i-1] + 1e-10:
                new_nodes[i] = new_nodes[i-1] + 1e-10
        if new_nodes[-1] > 1.0 - 1e-8:
            # Rescale
            new_nodes = 1e-8 + (1.0 - 2e-8) * (new_nodes - new_nodes[0]) / (new_nodes[-1] - new_nodes[0])
        
        # Re-optimize weights for new nodes
        new_weights = _solve_weights_for_nodes(new_nodes, test_data)
        new_score = _eval_worst_error(new_nodes, new_weights, test_data)
        
        if new_score < best_score:
            best_nodes = new_nodes
            best_weights = new_weights
            best_score = new_score
    
    # Final refinement with Nelder-Mead on both nodes and weights
    if time.time() < deadline - 5:
        def _joint_obj(params):
            n = 8
            node_params = params[:n]
            log_w = params[n:]
            
            nodes = 0.5 + 0.5 * np.tanh(node_params)
            if np.any(np.diff(nodes) <= 1e-12):
                return 1e6
            
            weights = np.exp(log_w)
            weights = weights / np.sum(weights)
            
            return _eval_worst_error(nodes, weights, test_data)
        
        init_node_params = np.arctanh(2.0 * best_nodes - 1.0)
        init_log_w = np.log(best_weights)
        x0 = np.concatenate([init_node_params, init_log_w])
        
        try:
            result = minimize(_joint_obj, x0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_score:
                node_params = result.x[:8]
                log_w = result.x[8:]
                best_nodes = 0.5 + 0.5 * np.tanh(node_params)
                best_weights = np.exp(log_w)
                best_weights = best_weights / np.sum(best_weights)
                best_score = result.fun
        except Exception:
            pass
    
    # Ensure validity
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    order = np.argsort(best_nodes)
    best_nodes = best_nodes[order]
    best_weights = best_weights[order]
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
