"""Cosmology: universal luminosity-distance quadrature (8-node positive rule)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

_START = time.time()
_DEADLINE = _START + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_of_z_vec(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Vectorized."""
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-300))

def construct():
    # Design cosmologies: structured grid in parameter space
    Om0_vals = [0.20, 0.25, 0.30, 0.35]
    Ol0_vals = [0.65, 0.70, 0.75, 0.80]
    w0_vals = [-1.5, -1.0, -0.5, 0.0]
    w1_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Create a representative subset to keep computation manageable
    design_cosmos = []
    for Om0 in Om0_vals:
        Ol0 = 1.0 - Om0
        for w0 in w0_vals:
            for w1 in w1_vals:
                design_cosmos.append((Om0, Ol0, w0, w1))
    
    # Also add some specific edge cases
    design_cosmos.append((0.30, 0.70, -1.00, 0.00))  # LambdaCDM
    design_cosmos.append((0.27, 0.73, -1.00, 0.00))  # Planck-like
    design_cosmos.append((0.30, 0.70, -1.00, 1.00))  # wCDM+
    design_cosmos.append((0.30, 0.70, -1.00, -1.00))  # wCDM-
    
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
    
    # Precompute true integrals using quad
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            def integrand_u(u, z_max=z_max, Om0=Om0, Ol0=Ol0, w0=w0, w1=w1):
                z_val = z_max * u
                if z_val < 1e-15:
                    return 1.0
                return 1.0 / _E_of_z_vec(z_val, Om0, Ol0, w0, w1)
            result, _ = quad(integrand_u, 0, 1, limit=200)
            true_val = z_max * result
            true_vals.append((Om0, Ol0, w0, w1, z_max, true_val))
    
    true_vals_arr = np.array([tv[5] for tv in true_vals])
    cosmos_params = np.array([tv[:4] for tv in true_vals])
    z_max_arr = np.array([tv[4] for tv in true_vals])
    n_design = len(true_vals_arr)
    
    def _compute_inv_E(nodes):
        """Compute 1/E for all design points and given nodes. Returns (n_design, 8)."""
        z_eval = z_max_arr[:, None] * nodes[None, :]
        a = 1.0 / (1.0 + z_eval)
        a3 = a ** 3
        exponent = -3.0 * cosmos_params[:, 3:4] * (1.0 - a)
        de_factor = a ** (-3.0 * (1.0 + cosmos_params[:, 2:3])) * np.exp(exponent)
        E2 = cosmos_params[:, 0:1] / a3 + cosmos_params[:, 1:2] * de_factor
        E = np.sqrt(np.maximum(E2, 1e-300))
        return 1.0 / E
    
    def _worst_rel_error(nodes, weights):
        """Compute worst relative error in percent."""
        inv_E = _compute_inv_E(nodes)
        approx = z_max_arr * (inv_E @ weights)
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        return np.max(rel_err)
    
    def _solve_weights_lp(nodes):
        """Solve for optimal weights minimizing worst-case absolute residual via LP.
        Minimize t s.t. |A @ w - b|_inf <= t, sum(w) = 1, w >= 0.
        """
        inv_E = _compute_inv_E(nodes)
        A = z_max_arr[:, None] * inv_E  # (n_design, 8)
        b = true_vals_arr  # (n_design,)
        
        # Variables: w[0..7], t
        # Minimize t
        c = np.zeros(9)
        c[8] = 1.0
        
        # Inequality constraints: A@w - b <= t and -(A@w - b) <= t
        # => A@w - t <= b and -A@w - t <= -b
        A_ub = np.zeros((2 * n_design, 9))
        b_ub = np.zeros(2 * n_design)
        
        A_ub[:n_design, :8] = A
        A_ub[:n_design, 8] = -1.0
        b_ub[:n_design] = b
        
        A_ub[n_design:, :8] = -A
        A_ub[n_design:, 8] = -1.0
        b_ub[n_design:] = -b
        
        # Equality: sum(w) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w >= 0, t >= 0
        bounds = [(0, None)] * 8 + [(0, None)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
        
        if res.success:
            weights = res.x[:8]
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            return weights
        else:
            # Fallback: equal weights
            return np.ones(8) / 8.0
    
    def _nodes_objective(raw_nodes):
        """Objective for node optimization: convert raw to nodes, solve LP for weights,
        return worst relative error."""
        if _time_left() < 10:
            return 1e10
        nodes = 0.5 + 0.5 * np.tanh(raw_nodes)
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        order = np.argsort(nodes)
        nodes = nodes[order]
        
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e10
        
        weights = _solve_weights_lp(nodes)
        return _worst_rel_error(nodes, weights)
    
    # Initial node guesses - multiple diverse starting points
    init_nodes_list = []
    
    # Chebyshev nodes
    k = np.arange(8)
    cheb_nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8))
    cheb_nodes = np.sort(cheb_nodes)
    init_nodes_list.append(cheb_nodes)
    
    # Gauss-Legendre nodes
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    gl_01 = 0.5 * (gl_nodes + 1.0)
    init_nodes_list.append(gl_01)
    
    # Uniform nodes
    unif_nodes = np.linspace(0.05, 0.95, 8)
    init_nodes_list.append(unif_nodes)
    
    # Log-spaced nodes (better for low-z behavior)
    log_nodes = np.logspace(np.log10(0.02), np.log10(0.98), 8)
    init_nodes_list.append(log_nodes)
    
    # Power-law nodes
    power_nodes = np.linspace(0.02, 0.98, 8) ** 0.7
    init_nodes_list.append(power_nodes)
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand_nodes = np.sort(rng.uniform(0.01, 0.99, 8))
        init_nodes_list.append(rand_nodes)
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Phase 1: For each initial node set, solve LP for weights
    for init_nodes in init_nodes_list:
        if _time_left() < 15:
            break
        weights = _solve_weights_lp(init_nodes)
        err = _worst_rel_error(init_nodes, weights)
        if err < best_err:
            best_err = err
            best_nodes = init_nodes.copy()
            best_weights = weights.copy()
    
    # Phase 2: Nelder-Mead refinement of nodes with LP weight re-solution
    if _time_left() > 20:
        # Convert best nodes to raw parameter space
        best_raw = np.arctanh(2.0 * best_nodes - 1.0)
        
        try:
            res_nm = minimize(_nodes_objective, best_raw, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14,
                                      'adaptive': True})
            
            nodes = 0.5 + 0.5 * np.tanh(res_nm.x)
            nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
            order = np.argsort(nodes)
            nodes = nodes[order]
            weights = _solve_weights_lp(nodes)
            err = _worst_rel_error(nodes, weights)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
        except Exception:
            pass
    
    # Phase 3: Differential Evolution for global search
    if _time_left() > 30:
        try:
            from scipy.optimize import differential_evolution
            
            # Bounds for raw parameters
            bounds_raw = [(-8, 8)] * 8
            
            def _de_objective(raw_nodes):
                if _time_left() < 10:
                    return 1e10
                nodes = 0.5 + 0.5 * np.tanh(raw_nodes)
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                order = np.argsort(nodes)
                nodes = nodes[order]
                
                if np.any(np.diff(nodes) <= 1e-12):
                    return 1e10
                
                weights = _solve_weights_lp(nodes)
                return _worst_rel_error(nodes, weights)
            
            # Run DE with limited population
            res_de = differential_evolution(_de_objective, bounds_raw,
                                           maxiter=50, popsize=15,
                                           seed=42, tol=1e-8)
            
            nodes = 0.5 + 0.5 * np.tanh(res_de.x)
            nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
            order = np.argsort(nodes)
            nodes = nodes[order]
            weights = _solve_weights_lp(nodes)
            err = _worst_rel_error(nodes, weights)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
        except Exception:
            pass
    
    # Phase 4: Additional random restarts with Nelder-Mead if time permits
    if _time_left() > 25:
        for seed in [123, 456, 789, 101, 202]:
            if _time_left() < 15:
                break
            rng2 = np.random.default_rng(seed)
            rand_n = np.sort(rng2.uniform(0.01, 0.99, 8))
            raw_n = np.arctanh(2.0 * rand_n - 1.0)
            
            try:
                res = minimize(_nodes_objective, raw_n, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-14,
                                      'adaptive': True})
                nodes = 0.5 + 0.5 * np.tanh(res.x)
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                order = np.argsort(nodes)
                nodes = nodes[order]
                weights = _solve_weights_lp(nodes)
                err = _worst_rel_error(nodes, weights)
                
                if err < best_err:
                    best_err = err
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
            except Exception:
                pass
    
    # Final cleanup
    nodes = best_nodes
    weights = best_weights
    
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
