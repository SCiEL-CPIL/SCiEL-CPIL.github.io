"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.integrate import quad

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    return abs(approx - exact) / exact * 100.0

def _gauss_legendre_nodes(n):
    """Generate Gauss-Legendre nodes and weights on [0,1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]
    w = w / np.sum(w)
    return x, w

def _log_param_to_nodes(log_nodes):
    """Convert log-parameterized nodes to actual nodes in (0,1)."""
    nodes = 0.5 * (1.0 + np.tanh(log_nodes))
    nodes = np.clip(nodes, 1e-8, 1.0 - 1e-8)
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1] + 1e-12:
            nodes[i] = nodes[i-1] + 1e-10
    return nodes

def _log_param_to_weights(log_weights):
    """Convert log-parameterized weights to positive weights summing to 1."""
    log_w = log_weights - np.max(log_weights)
    w = np.exp(log_w)
    w = w / np.sum(w)
    return w

def _objective(params, z_samples, models):
    """Objective: worst-case relative error."""
    n = len(params) // 2
    log_nodes = params[:n]
    log_weights = params[n:]
    
    nodes = _log_param_to_nodes(log_nodes)
    weights = _log_param_to_weights(log_weights)
    
    max_err = 0.0
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, nodes, weights, wa, wb, w0)
            if err > max_err:
                max_err = err
    return max_err

def _find_optimal_weights_lp(nodes, z_samples, models, max_iter=20):
    """
    Find weights that minimize worst-case relative error via iterative LP.
    For fixed nodes, the problem is:
      min t
      s.t. -t <= (sum_i w_i / E_i(z,m)) * (1+z)*z - exact(z,m) <= t
           sum_i w_i = 1
           w_i >= 0
    """
    n = len(nodes)
    n_cosmo = len(models)
    n_z = len(z_samples)
    
    # Precompute exact values and 1/E values
    exact_vals = np.zeros((n_cosmo, n_z))
    inv_E = np.zeros((n_cosmo, n_z, n))
    
    for ci, (wa, wb, w0) in enumerate(models):
        for zi, z in enumerate(z_samples):
            exact_vals[ci, zi] = _exact_dL(z, wa, wb, w0)
            a = 1.0 / (1.0 + z * nodes)
            E_vals = _E_of_a(a, wa, wb, w0)
            inv_E[ci, zi, :] = 1.0 / E_vals
    
    # Coefficient matrix for LP: A_ub @ vars <= b_ub
    # vars = [w_0, ..., w_{n-1}, t]
    # Constraints:
    #   For each (ci, zi): 
    #     (1+z)*z * sum_i w_i * inv_E[ci,zi,i] - exact[ci,zi] <= t
    #     -(1+z)*z * sum_i w_i * inv_E[ci,zi,i] + exact[ci,zi] <= t
    #   sum_i w_i = 1 (handled as equality)
    #   w_i >= 0 (handled by bounds)
    #   t >= 0 (handled by bounds)
    
    n_constraints = 2 * n_cosmo * n_z
    A_ub = np.zeros((n_constraints, n + 1))
    b_ub = np.zeros(n_constraints)
    
    for ci in range(n_cosmo):
        for zi in range(n_z):
            z = z_samples[zi]
            coeff = (1.0 + z) * z
            row_idx = ci * n_z + zi
            
            # (1+z)*z * sum_i w_i * inv_E - exact <= t
            # => (1+z)*z * sum_i w_i * inv_E - t <= exact
            A_ub[row_idx, :n] = coeff * inv_E[ci, zi, :]
            A_ub[row_idx, n] = -1.0
            b_ub[row_idx] = exact_vals[ci, zi]
            
            # -((1+z)*z * sum_i w_i * inv_E) - exact <= t  ... wait, let me re-derive
            # -t <= (1+z)*z * sum_i w_i * inv_E - exact
            # => t >= (1+z)*z * sum_i w_i * inv_E - exact
            # => -(1+z)*z * sum_i w_i * inv_E + t <= -exact
            # => -(1+z)*z * sum_i w_i * inv_E + t <= -exact
            # Actually: -t <= X - exact  =>  exact - X <= t  =>  -X - t <= -exact
            # Let me be more careful.
            # Constraint 1: X - exact <= t  =>  X - t <= exact
            # Constraint 2: -(X - exact) <= t  =>  -X + t <= -exact
            # where X = (1+z)*z * sum_i w_i * inv_E
            
            row_idx2 = n_cosmo * n_z + ci * n_z + zi
            A_ub[row_idx2, :n] = -coeff * inv_E[ci, zi, :]
            A_ub[row_idx2, n] = 1.0
            b_ub[row_idx2] = -exact_vals[ci, zi]
    
    # Equality constraint: sum_i w_i = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds: w_i >= 0, t >= 0
    bounds = [(0, None)] * n + [(0, None)]
    
    # Initial guess
    x0 = np.zeros(n + 1)
    x0[:n] = 1.0 / n
    x0[n] = 1.0
    
    best_w = np.ones(n) / n
    best_t = np.inf
    
    for iteration in range(max_iter):
        try:
            result = linprog(
                c=np.concatenate([np.zeros(n), [1.0]]),
                A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            if result.success:
                w = result.x[:n]
                t = result.x[n]
                if t < best_t:
                    best_t = t
                    best_w = w.copy()
                # The LP is linear in w and t for fixed nodes, so one solve should suffice
                # But due to numerical issues, we might want to iterate
                # Actually, for fixed nodes, this IS the optimal solution in one LP
                break
            else:
                break
        except Exception:
            break
    
    return best_w, best_t

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline (5s margin)
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
        (0.0, -0.2, -1.1),      # Another variant
        (0.15, 0.1, -0.95),     # Mild evolution
        (0.0, 0.5, -0.85),      # Moderate w_a
        (0.05, -0.4, -1.05),    # Slight phantom
    ]
    
    # Z samples for evaluation
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    
    n = 8
    
    # Start with Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = _gauss_legendre_nodes(n)
    
    # Convert to log-parameterization
    log_nodes_init = np.arctanh(2.0 * gl_nodes - 1.0)
    log_weights_init = np.log(gl_weights)
    
    params_init = np.concatenate([log_nodes_init, log_weights_init])
    
    # Evaluate initial score
    best_params = params_init.copy()
    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # Phase 1: Optimize nodes with LP-optimized weights for each node configuration
    # This is the structural change: instead of jointly optimizing nodes+weights,
    # we optimize nodes where the weights are always the LP-optimal ones for those nodes.
    
    if time.time() < deadline - 20:
        def node_objective(log_nodes):
            """Objective for nodes only: LP-optimal weights, then worst-case error."""
            nodes = _log_param_to_nodes(log_nodes)
            try:
                weights, t = _find_optimal_weights_lp(nodes, z_samples, models, max_iter=1)
                # Re-evaluate actual worst-case error with these weights
                max_err = 0.0
                for z in z_samples:
                    for wa, wb, w0 in models:
                        err = _relative_error(z, nodes, weights, wa, wb, w0)
                        if err > max_err:
                            max_err = err
                return max_err
            except Exception:
                return 100.0
        
        # Try a few starting points for node optimization
        node_starts = [
            log_nodes_init,  # GL
            np.arctanh(2.0 * np.linspace(0.05, 0.95, n) - 1.0),  # Uniform
            np.arctanh(2.0 * np.sqrt(np.linspace(0.01, 0.99, n)) - 1.0),  # Sqrt-spaced
            np.arctanh(2.0 * (1.0 - np.sqrt(np.linspace(0.01, 0.99, n))) - 1.0),  # Inverse sqrt
        ]
        
        for ns in node_starts:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(
                    node_objective, ns,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-10}
                )
                if result.fun < best_score:
                    nodes = _log_param_to_nodes(result.x)
                    weights, _ = _find_optimal_weights_lp(nodes, z_samples, models, max_iter=1)
                    # Verify
                    max_err = 0.0
                    for z in z_samples:
                        for wa, wb, w0 in models:
                            err = _relative_error(z, nodes, weights, wa, wb, w0)
                            if err > max_err:
                                max_err = err
                    if max_err < best_score:
                        best_score = max_err
                        best_nodes = nodes
                        best_weights = weights
                        best_params = np.concatenate([result.x, np.log(weights)])
            except Exception:
                pass
    
    # Phase 2: Joint refinement with L-BFGS-B on both nodes and weights
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _objective, best_params,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception:
            pass
    
    # Phase 3: Differential evolution with multiple restarts
    remaining = deadline - time.time()
    if remaining > 25:
        bounds = [(-5, 5)] * n + [(-5, 5)] * n
        
        def de_objective(params):
            return _objective(params, z_samples, models)
        
        n_restarts = max(1, int(remaining / 12))
        time_per_restart = remaining / n_restarts - 1
        
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                result = differential_evolution(
                    de_objective, bounds,
                    maxiter=40, popsize=12,
                    tol=1e-6, seed=42 + r,
                    polish=True,
                    workers=1,
                    time_limit=max(5, time_per_restart)
                )
                
                if result.fun < best_score:
                    best_params = result.x
                    best_nodes = _log_param_to_nodes(result.x[:n])
                    best_weights = _log_param_to_weights(result.x[n:])
                    best_score = result.fun
            except Exception:
                pass
    
    # Phase 4: Final refinement
    if time.time() < deadline - 3:
        try:
            result = minimize(
                _objective, best_params,
                args=(z_samples, models),
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-14}
            )
            if result.fun < best_score:
                best_params = result.x
                best_nodes = _log_param_to_nodes(result.x[:n])
                best_weights = _log_param_to_weights(result.x[n:])
                best_score = result.fun
        except Exception:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
