"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
import scipy.integrate as si

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    Omega_m = 0.3
    Omega_L = 0.7

    # Basis set: representative cosmologies
    w0_vals = [-1.5, -1.0, -0.5, 0.0]
    wa_vals = [-1.0, 0.0, 1.0]
    z_vals = [0.1, 0.5, 1.0, 2.0, 5.0]

    basis_cases = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                basis_cases.append((z, w0, wa))

    def _compute_exact(z, w0, wa):
        def integrand(t):
            a = 1.0 / (1.0 + z * t)
            if a <= 0:
                return 0.0
            log_term = 3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (a - 1.0)
            if log_term < -700:
                term_L = 0.0
            else:
                term_L = Omega_L * np.exp(log_term)
            term_m = Omega_m * (a ** -3)
            E2 = term_m + term_L
            if E2 <= 0:
                return 0.0
            E = np.sqrt(E2)
            return 1.0 / E

        res, err = si.quad(integrand, 0.0, 1.0, limit=100, epsabs=1e-10, epsrel=1e-10)
        return res

    # Precompute exact values for basis cases
    exact_vals = []
    for z, w0, wa in basis_cases:
        if time.time() > deadline:
            break
        try:
            val = _compute_exact(z, w0, wa)
            exact_vals.append(val)
        except Exception:
            exact_vals.append(1.0)

    if not exact_vals:
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return {"nodes": nodes, "weights": weights}

    def _quadrature_vec(nodes, weights, z, w0, wa):
        a_vals = 1.0 / (1.0 + z * nodes)
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = term_m + term_L
        E2 = np.maximum(E2, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        return np.dot(weights, inv_E)

    def _rel_errors(nodes, weights):
        errs = np.zeros(len(basis_cases))
        for i, (z, w0, wa) in enumerate(basis_cases):
            exact_val = exact_vals[i]
            if exact_val == 0:
                continue
            quad_val = _quadrature_vec(nodes, weights, z, w0, wa)
            errs[i] = (quad_val - exact_val) / exact_val
        return errs

    # Parameterization:
    # nodes: use log-spaced parameterization to ensure strict increase in (0,1)
    # Let s_i be positive, then u_i = 1 - exp(-c * cumsum(s)) / (1 - exp(-c))
    # Actually simpler: u_i = exp(-exp(-x_i)) mapped to (0,1) via u = 1/(1+exp(-x))
    # But we need strictly increasing. Use: u_i = sigmoid(x_i) where x_1 < x_2 < ... < x_8
    # Parameterize x_i as x_1 + sum_{j=1}^{i-1} exp(y_j) where y_j > 0
    # Then x is strictly increasing, and u = sigmoid(x) is strictly increasing in (0,1)

    def _param_to_nodes_weights(params):
        # params: [x1, y1, y2, ..., y7, logw1, logw2, ..., logw8]
        # x1: first node logit
        # y1..y7: log of gaps between consecutive logits
        # logw1..logw8: log weights
        x1 = params[0]
        y = params[1:8]
        logw = params[8:16]

        # Compute logits
        x = np.zeros(8)
        x[0] = x1
        for i in range(1, 8):
            x[i] = x[i-1] + np.exp(y[i-1])

        # Convert to nodes in (0,1)
        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)

        # Convert to weights
        w = np.exp(logw)
        w = w / np.sum(w)

        return nodes, w

    def _residuals(params):
        nodes, weights = _param_to_nodes_weights(params)
        errs = _rel_errors(nodes, weights)
        return errs

    # Initial guess: Gauss-Legendre
    nodes_init, weights_init = np.polynomial.legendre.leggauss(8)
    nodes_init = (nodes_init + 1.0) / 2.0
    weights_init = weights_init / 2.0

    # Convert to parameter space
    x_init = np.log(nodes_init / (1.0 - nodes_init))
    params_init = np.zeros(16)
    params_init[0] = x_init[0]
    for i in range(1, 8):
        gap = x_init[i] - x_init[i-1]
        if gap <= 0:
            gap = 1e-6
        params_init[1+i-1] = np.log(gap)
    params_init[8:16] = np.log(weights_init)

    # Run least_squares
    best_params = params_init
    best_cost = np.sum(np.square(_rel_errors(*_param_to_nodes_weights(params_init))))

    try:
        res = so.least_squares(_residuals, params_init, method='trf',
                               max_nfev=3000, xtol=1e-14, ftol=1e-14)
        if res.cost < best_cost:
            best_params = res.x
            best_cost = res.cost
    except Exception:
        pass

    # Min-max refinement: minimize max |relative error|
    def _minmax_objective(params):
        nodes, weights = _param_to_nodes_weights(params)
        errs = _rel_errors(nodes, weights)
        return np.max(np.abs(errs))

    if time.time() < deadline - 10.0:
        try:
            res_mm = so.minimize(_minmax_objective, best_params, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res_mm.fun < _minmax_objective(best_params):
                best_params = res_mm.x
        except Exception:
            pass

    # Second least-squares pass from min-max result
    if time.time() < deadline - 5.0:
        try:
            res2 = so.least_squares(_residuals, best_params, method='trf',
                                    max_nfev=2000, xtol=1e-14, ftol=1e-14)
            if res2.cost < best_cost:
                best_params = res2.x
                best_cost = res2.cost
        except Exception:
            pass

    final_nodes, final_weights = _param_to_nodes_weights(best_params)

    # Final cleanup
    final_nodes = np.clip(final_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-12
    final_weights = np.clip(final_weights, 1e-12, None)
    final_weights = final_weights / np.sum(final_weights)

    return {"nodes": final_nodes, "weights": final_weights}

# EVOLVE-BLOCK-END
