"""Cosmology: universal luminosity-distance quadrature (8-node positive rule)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
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
    # Design cosmologies
    design_cosmos = [
        (0.30, 0.70, -1.00,  0.00),
        (0.27, 0.73, -1.00,  0.00),
        (0.30, 0.70, -1.00,  1.00),
        (0.30, 0.70, -1.00, -1.00),
        (0.20, 0.80, -1.20,  0.50),
        (0.35, 0.65, -0.80,  0.30),
        (0.25, 0.75, -1.50,  1.00),
        (0.30, 0.70, -0.50, -0.50),
        (0.30, 0.70, -1.00,  0.50),
        (0.30, 0.70, -1.50, -0.50),
        (0.30, 0.70, -0.80, -1.00),
        (0.25, 0.75, -1.00,  0.80),
        (0.28, 0.72, -0.90,  0.20),
        (0.32, 0.68, -1.10, -0.30),
        (0.30, 0.70, -1.00, -0.50),
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Precompute true integrals
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

    def _smooth_obj(params):
        """
        Objective for joint optimization.
        params: [8 node_raws, 8 weight_raws]
        Nodes: 0.5 + 0.5 * tanh(node_raw)
        Weights: softmax(weight_raw)
        Returns smooth max relative error (log-sum-exp approximation).
        """
        node_raws = params[:8]
        weight_raws = params[8:16]
        
        nodes = 0.5 + 0.5 * np.tanh(node_raws)
        # Sort nodes to ensure strict ordering for quadrature rule definition?
        # Actually, the formula dL = (1+z)z sum w_i / E(a_i) doesn't require sorted nodes
        # as long as weights correspond to the nodes. But the problem statement says
        # "Nodes must be strictly increasing". The evaluator likely checks this.
        # However, during optimization, we can treat them as unordered and sort at the end.
        # But if we sort nodes, we must sort weights accordingly.
        # Let's enforce sorting in the objective to satisfy the constraint implicitly?
        # No, better to let them be free and sort at the end.
        # But if we don't sort, the "node" identity is fixed to the index.
        # Let's assume the optimizer finds a set of 8 points. We will sort them at the end.
        
        # Softmax for weights
        w_max = np.max(weight_raws)
        exp_w = np.exp(weight_raws - w_max)
        weights = exp_w / np.sum(exp_w)
        
        # Check for duplicate nodes
        sorted_nodes = np.sort(nodes)
        if np.any(np.diff(sorted_nodes) < 1e-6):
            return 1e10

        inv_E = _compute_inv_E(nodes)
        approx = z_max_arr * (inv_E @ weights)
        rel_err = np.abs(approx - true_vals_arr) / np.maximum(np.abs(true_vals_arr), 1e-300) * 100.0
        
        # Smooth max: log-sum-exp with beta
        beta = 50.0
        max_err = np.max(rel_err)
        # Avoid overflow
        if max_err > 1e6:
            return max_err
        smooth_max = (1.0/beta) * np.log(np.sum(np.exp(beta * rel_err)))
        return smooth_max

    # Initial guesses
    # 1. Gauss-Legendre
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    gl_01 = 0.5 * (gl_nodes + 1.0)
    init_gl = np.concatenate([np.arctanh(2*gl_01 - 1), np.zeros(8)])
    
    # 2. Chebyshev
    k = np.arange(8)
    cheb_nodes = 0.5 * (1.0 - np.cos(np.pi * (k + 0.5) / 8))
    init_cheb = np.concatenate([np.arctanh(2*cheb_nodes - 1), np.zeros(8)])
    
    # 3. Random
    rng = np.random.default_rng(42)
    rand_nodes = np.sort(rng.uniform(0.05, 0.95, 8))
    init_rand = np.concatenate([np.arctanh(2*rand_nodes - 1), np.zeros(8)])

    best_params = None
    best_score = np.inf

    # Phase 1: Differential Evolution for global search on nodes only?
    # DE is expensive. Let's try a few restarts of L-BFGS-B from different inits.
    
    inits = [init_gl, init_cheb, init_rand]
    
    for init in inits:
        if _time_left() < 20:
            break
        try:
            # L-BFGS-B with finite difference gradients
            res = minimize(_smooth_obj, init, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
        except Exception:
            pass

    # Phase 2: Local refinement with Nelder-Mead on the best params found
    if best_params is not None and _time_left() > 10:
        try:
            res_nm = minimize(_smooth_obj, best_params, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # Extract final nodes and weights
    node_raws = best_params[:8]
    weight_raws = best_params[8:16]
    
    nodes = 0.5 + 0.5 * np.tanh(node_raws)
    w_max = np.max(weight_raws)
    exp_w = np.exp(weight_raws - w_max)
    weights = exp_w / np.sum(exp_w)
    
    # Sort nodes and reorder weights
    order = np.argsort(nodes)
    nodes = nodes[order]
    weights = weights[order]
    
    # Cleanup
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    weights = np.maximum(weights, 1e-15)
    weights = weights / np.sum(weights)
    
    # Ensure strict increase
    # If duplicates exist, perturb slightly
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-10

    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"

    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
