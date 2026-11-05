"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.integrate import quad


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s

    # 8-point Gauss-Legendre nodes and weights on [-1, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes_init = (gl_nodes + 1.0) / 2.0
    weights_init = gl_weights / 2.0
    nodes_init = np.sort(nodes_init)
    order = np.argsort(gl_nodes)
    weights_init = gl_weights[order] / 2.0
    weights_init = weights_init / weights_init.sum()

    # Define cosmological E(a) functions
    def E_lcdm(a):
        return np.sqrt(0.3 / a**3 + 0.7)

    def E_cpl(a, w0, w3):
        om = 0.3
        ol = 1.0 - om
        w = w0 + w3 * (1.0 - a)
        return np.sqrt(om / a**3 + ol * a**(-3.0 * (1.0 + w)))

    def E_power(a, alpha):
        om = 0.3
        ol = 1.0 - om
        w = -1.0 + alpha / 3.0
        return np.sqrt(om / a**3 + ol * a**(-3.0 * (1.0 + w)))

    # Training models: diverse CPL and power-law models
    models = [
        ("lcdm", E_lcdm),
        ("cpl_-0.5_0.5", lambda a: E_cpl(a, -0.5, 0.5)),
        ("cpl_-1.0_0.0", lambda a: E_cpl(a, -1.0, 0.0)),
        ("cpl_-0.8_0.3", lambda a: E_cpl(a, -0.8, 0.3)),
        ("cpl_-1.2_0.8", lambda a: E_cpl(a, -1.2, 0.8)),
        ("cpl_-0.3_0.2", lambda a: E_cpl(a, -0.3, 0.2)),
        ("cpl_-1.5_1.0", lambda a: E_cpl(a, -1.5, 1.0)),
        ("cpl_-0.7_-0.3", lambda a: E_cpl(a, -0.7, -0.3)),
        ("pow_0.2", lambda a: E_power(a, 0.2)),
        ("pow_0.5", lambda a: E_power(a, 0.5)),
        ("pow_1.0", lambda a: E_power(a, 1.0)),
        ("pow_1.5", lambda a: E_power(a, 1.5)),
    ]

    # Redshifts to test
    test_zs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Pre-compute reference integrals
    ref_integrals = {}
    for name, E_func in models:
        for z in test_zs:
            def integrand(x, E=E_func, zz=z):
                a = 1.0 / (1.0 + zz * x)
                return 1.0 / E(a)
            result, _ = quad(integrand, 0.0, 1.0, limit=300)
            ref_integrals[(name, z)] = result

    def compute_worst_error(nodes_arr, weights_arr):
        worst = 0.0
        for name, E_func in models:
            for z in test_zs:
                a_vals = 1.0 / (1.0 + z * nodes_arr)
                quad_val = np.sum(weights_arr / E_func(a_vals))
                ref_val = ref_integrals[(name, z)]
                if ref_val != 0:
                    err = abs(quad_val - ref_val) / abs(ref_val)
                    if err > worst:
                        worst = err
        return worst

    def soft_max_error(nodes_arr, weights_arr, beta=50.0):
        errors = []
        for name, E_func in models:
            for z in test_zs:
                a_vals = 1.0 / (1.0 + z * nodes_arr)
                quad_val = np.sum(weights_arr / E_func(a_vals))
                ref_val = ref_integrals[(name, z)]
                if ref_val != 0:
                    err = abs(quad_val - ref_val) / abs(ref_val)
                    errors.append(err)
        errors = np.array(errors)
        max_err = np.max(errors)
        if max_err < 1e-15:
            return 0.0
        shifted = errors / max_err - 1.0
        return max_err * np.log(np.sum(np.exp(beta * shifted))) / beta

    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_score = compute_worst_error(best_nodes, best_weights)

    # Parameterization: nodes via cumulative differences, weights via log
    def param_to_nodes_weights(params):
        # First 8 params: cumulative node positions (with sigmoid to stay in (0,1))
        # Use a transformation: nodes[i] = sigmoid(params[i] * scale)
        # But we need strictly increasing, so use cumulative sum of positive values
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: use cumulative sum of softmax-normalized positive values
        node_pos = np.exp(node_params)  # ensure positive
        node_pos = node_pos / node_pos.sum()
        # Scale to [0.01, 0.99]
        nodes_out = 0.01 + 0.98 * node_pos
        # Ensure strictly increasing
        if not np.all(np.diff(nodes_out) > 1e-6):
            return None, None
        
        # Weights: softmax
        w_pos = np.exp(weight_params - np.max(weight_params))
        weights_out = w_pos / w_pos.sum()
        
        return nodes_out, weights_out

    def objective(params):
        nodes_out, weights_out = param_to_nodes_weights(params)
        if nodes_out is None:
            return 1e10
        return soft_max_error(nodes_out, weights_out, beta=50.0)

    # Initial parameters
    # Invert the parameterization
    node_pos_init = (best_nodes - 0.01) / 0.98
    node_pos_init = node_pos_init / node_pos_init.sum()
    node_params_init = np.log(node_pos_init)
    weight_params_init = np.log(best_weights + 1e-10)
    x0 = np.concatenate([node_params_init, weight_params_init])

    # Differential Evolution for global search
    if time.time() < deadline - 20.0:
        try:
            de_result = optimize.differential_evolution(
                objective,
                bounds=[(-5, 5)] * 16,
                maxiter=100,
                tol=1e-8,
                popsize=15,
                seed=42,
                polish=True
            )
            if de_result.fun < best_score:
                nodes_out, weights_out = param_to_nodes_weights(de_result.x)
                if nodes_out is not None:
                    best_nodes = nodes_out
                    best_weights = weights_out
                    best_score = compute_worst_error(best_nodes, best_weights)
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            nm_result = optimize.minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if nm_result.fun < best_score:
                nodes_out, weights_out = param_to_nodes_weights(nm_result.x)
                if nodes_out is not None:
                    best_nodes = nodes_out
                    best_weights = weights_out
                    best_score = compute_worst_error(best_nodes, best_weights)
        except Exception:
            pass

    # Final polish with tighter tolerance
    if time.time() < deadline - 2.0:
        try:
            nm_result2 = optimize.minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if nm_result2.fun < best_score:
                nodes_out, weights_out = param_to_nodes_weights(nm_result2.x)
                if nodes_out is not None:
                    best_nodes = nodes_out
                    best_weights = weights_out
                    best_score = compute_worst_error(best_nodes, best_weights)
        except Exception:
            pass

    # Ensure final validity
    best_nodes = np.sort(best_nodes)
    best_weights = best_weights / best_weights.sum()
    
    # Final validity check
    if not (np.all(best_nodes > 0) and np.all(best_nodes < 1) and np.all(np.diff(best_nodes) > 1e-6)):
        # Fallback to Gauss-Legendre
        best_nodes = nodes_init.copy()
        best_weights = weights_init.copy()
    
    if not np.all(best_weights > 0):
        best_weights = weights_init.copy()
    
    best_weights = best_weights / best_weights.sum()

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
