"""Minimax quadrature design via differential evolution with vectorized evaluation."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # High-accuracy reference using 128-point Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(128)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    # Diverse test set: broader range of z, w0, wa
    z_vals = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    w0_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    wa_vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    n_tests = len(test_cases)

    # Precompute reference values vectorized
    ref_vals = np.zeros(n_tests)
    for i, (z, w0, wa) in enumerate(test_cases):
        if time.time() > deadline - 60.0:
            break
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        ref_vals[i] = np.dot(gl_weights, inv_E)

    # Precompute test case arrays for fast vectorized evaluation
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    def _param_to_nodes_weights(params):
        """
        Parameterization:
        - First 7 values: log-spacings between consecutive nodes (guarantees increasing)
        - 8th value: log of first node position (sigmoid-scaled to (0,1))
        - Next 8 values: log-weights (normalized to sum to 1)
        Total: 16 parameters
        """
        log_gaps = params[:7]
        log_first = params[7]
        log_weights = params[8:16]

        # Build nodes from log-spacings in logit space
        # x_i = logit(node_i), so node_i = sigmoid(x_i)
        # gaps in logit space: x_{i+1} - x_i = exp(log_gaps[i])
        gap_sizes = np.exp(log_gaps)
        x0 = log_first
        x = np.zeros(8)
        x[0] = x0
        for i in range(7):
            x[i+1] = x[i] + gap_sizes[i]

        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)

        # Ensure strictly increasing (should be by construction but clip for safety)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-10

        w = np.exp(log_weights)
        w = w / np.sum(w)
        return nodes, w

    def _nodes_weights_to_param(nodes, weights):
        """Inverse of _param_to_nodes_weights."""
        x = np.log(np.clip(nodes, 1e-10, 1-1e-10) / (1.0 - np.clip(nodes, 1e-10, 1-1e-10)))
        params = np.zeros(16)
        params[7] = x[0]
        for i in range(7):
            gap = x[i+1] - x[i]
            if gap < 1e-12:
                gap = 1e-12
            params[i] = np.log(gap)
        params[8:16] = np.log(np.maximum(weights, 1e-12))
        return params

    def _worst_rel_error_vec(params):
        """Vectorized worst-case relative error in percent."""
        nodes, weights = _param_to_nodes_weights(params)

        # Vectorized evaluation: for each test case, compute quadrature
        # a_vals[i,j] = 1/(1 + z_i * nodes_j)
        a_vals = 1.0 / (1.0 + z_arr[:, None] * nodes[None, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)

        log_terms = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) \
                   - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_terms = np.clip(log_terms, -700, 700)

        term_L = Omega_L * np.exp(log_terms)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E

        # Quadrature values: dot(weights, inv_E) for each test case
        qvals = inv_E @ weights

        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        return np.max(rel_errs)

    # Generate diverse initial candidates for multi-start
    def _make_gauss_legendre():
        nodes, weights = np.polynomial.legendre.leggauss(8)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        return _nodes_weights_to_param(nodes, weights)

    def _make_jacobi(alpha, beta):
        from scipy.special import roots_jacobi
        nodes, weights = roots_jacobi(8, alpha, beta)
        nodes = (nodes + 1.0) / 2.0
        weights = weights / 2.0
        if np.all(np.diff(nodes) > 0) and np.all(weights > 0):
            return _nodes_weights_to_param(nodes, weights)
        return _make_gauss_legendre()

    def _make_beta_transformed(beta):
        nodes_g, weights_g = np.polynomial.legendre.leggauss(8)
        nodes_g = (nodes_g + 1.0) / 2.0
        nodes_w = nodes_g ** beta
        weights_w = weights_g * beta * (nodes_g ** (beta - 1))
        weights_w = np.abs(weights_w)
        weights_w = weights_w / np.sum(weights_w)
        if np.all(np.diff(nodes_w) > 0):
            return _nodes_weights_to_param(nodes_w, weights_w)
        return _make_gauss_legendre()

    def _make_random(seed):
        rng = np.random.RandomState(seed)
        gaps = rng.exponential(1.0, size=8)
        nodes = np.cumsum(gaps)
        nodes = nodes / nodes[-1]
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        # Simpson-like weights
        weights = np.ones(8)
        weights[0] = (nodes[1] - nodes[0]) / 2
        weights[-1] = (nodes[-1] - nodes[-2]) / 2
        for i in range(1, 7):
            weights[i] = (nodes[i+1] - nodes[i-1]) / 2
        weights = np.maximum(weights, 1e-12)
        weights = weights / np.sum(weights)
        return _nodes_weights_to_param(nodes, weights)

    # Build initial population
    initial_params = []
    initial_params.append(_make_gauss_legendre())
    for alpha, beta in [(0.5, 0.5), (1, 0), (0, 1), (2, 1), (1, 2), (0.5, 2), (2, 0.5), (1.5, 0.5)]:
        initial_params.append(_make_jacobi(alpha, beta))
    for beta in [0.5, 0.7, 1.3, 1.5, 2.0, 0.3]:
        initial_params.append(_make_beta_transformed(beta))
    for seed in range(10):
        initial_params.append(_make_random(seed))

    # Evaluate initial population
    best_err = float('inf')
    best_params = initial_params[0].copy()

    for p in initial_params:
        if time.time() > deadline - 40.0:
            break
        try:
            err = _worst_rel_error_vec(p)
            if err < best_err:
                best_err = err
                best_params = p.copy()
        except Exception:
            pass

    # Phase 1: Differential Evolution (global search)
    if time.time() < deadline - 25.0:
        try:
            bounds = [(-5.0, 5.0)] * 16  # reasonable bounds for log-spacings, log-first, log-weights
            # Tighten bounds: log_gaps should be moderate, log_first in (-10, 10)
            bounds = [(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 8

            res = so.differential_evolution(
                _worst_rel_error_vec,
                bounds=bounds,
                seed=42,
                maxiter=500,
                popsize=15,
                tol=1e-10,
                polish=True,
                init='sobol',
                updating='immediate',
                mutation=(0.3, 0.9),
                recombination=0.8,
            )
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with L-BFGS-B from best DE solution
    if time.time() < deadline - 10.0:
        try:
            def _objective_local(params):
                return _worst_rel_error_vec(params)

            res = so.minimize(
                _objective_local,
                best_params,
                method='L-BFGS-B',
                bounds=[(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 8,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12},
            )
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-start local refinement from several diverse starting points
    if time.time() < deadline - 5.0:
        start_points = [best_params.copy()]
        # Add a few diverse starting points
        for sp in initial_params[:5]:
            start_points.append(sp.copy())

        for sp in start_points:
            if time.time() > deadline - 2.0:
                break
            try:
                res = so.minimize(
                    _worst_rel_error_vec,
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12},
                )
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Extract final nodes and weights
    best_nodes, best_weights = _param_to_nodes_weights(best_params)

    # Final cleanup
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
