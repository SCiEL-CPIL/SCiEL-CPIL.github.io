"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy (structural change):
    1. Fast ground-truth via fixed 64-point Gauss-Legendre (replaces slow quad).
    2. Alternating optimization: fix nodes -> solve weights via LP (minimax); 
       fix weights -> optimize nodes via Nelder-Mead.
    3. Multiple restarts from diverse initial node configurations.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))

    # Diverse set of CPL models
    test_models = [
        (-1.0, 0.0), (-1.0, 0.3), (-1.0, 0.6), (-1.0, -0.3),
        (-0.5, 0.0), (-0.5, 0.4), (-0.5, -0.2),
        (-1.5, 0.0), (-1.5, 0.3), (-1.5, -0.3),
        (-0.8, 0.3), (-0.8, -0.3),
        (-1.2, 0.4), (-1.2, -0.2),
        (-0.3, 0.2), (-0.3, -0.1),
        (-2.0, 0.2), (-2.0, -0.1),
        (0.0, 0.0), (0.0, 0.3),
        (-1.8, 0.2), (-1.8, -0.2),
        (-0.7, 0.5), (-0.7, -0.4),
        (-1.3, 0.3), (-1.3, -0.3),
        (-0.2, 0.3), (-0.2, -0.2),
        (-1.6, 0.1), (-1.6, -0.1),
        (-1.0, 0.8), (-1.0, -0.5),
    ]

    z_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

    n_models = len(test_models)
    n_z = len(z_values)
    n_cases = n_models * n_z

    # Fast ground truth: 64-point Gauss-Legendre per integral
    gl64_nodes, gl64_weights = np.polynomial.legendre.leggauss(64)
    true_vals = np.zeros(n_cases)
    for i, (w0, wa) in enumerate(test_models):
        for j, z in enumerate(z_values):
            idx = i * n_z + j
            # Map [-1,1] to [0, z]
            zp = 0.5 * z * (gl64_nodes + 1.0)
            E_vals = E_cpl(zp, w0, wa)
            integral = 0.5 * z * np.dot(gl64_weights, 1.0 / E_vals)
            true_vals[idx] = (1.0 + z) * integral

    def compute_max_error(nodes, weights):
        """Vectorized worst-case relative error."""
        max_err = 0.0
        for i, (w0, wa) in enumerate(test_models):
            for j, z in enumerate(z_values):
                idx = i * n_z + j
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                approx = (1.0 + z) * z * np.dot(weights, 1.0 / E_vals)
                tv = true_vals[idx]
                if tv > 1e-10:
                    err = abs(approx - tv) / tv
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_weights_lp(nodes):
        """
        For fixed nodes, find weights minimizing max relative error.
        Minimize t s.t. for each case k:
            -t <= (sum_i w_i * f_ki - t_k) / t_k <= t
            where f_ki = 1/E(a_i) at the node, t_k = true value.
        Actually: approx_k = (1+z_k)*z_k * sum_i w_i / E(a_i; model_k)
        relative error: |approx_k - true_k| / true_k <= t
        => -t*true_k <= (1+z_k)*z_k * sum_i w_i * (1/E_ki) - true_k <= t*true_k
        => sum_i w_i * (1/E_ki) >= (true_k - t*true_k) / ((1+z_k)*z_k)
        => sum_i w_i * (1/E_ki) <= (true_k + t*true_k) / ((1+z_k)*z_k)
        
        Variables: w_1..w_8, t
        Minimize t
        s.t. A_ub @ [w, t] <= b_ub
              A_eq @ [w, t] = b_eq
              w_i >= 0
        """
        # Precompute f_ki for all cases
        f = np.zeros((n_cases, 8))
        for i, (w0, wa) in enumerate(test_models):
            for j, z in enumerate(z_values):
                idx = i * n_z + j
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                f[idx, :] = 1.0 / E_vals

        # For each case k: approx_k = (1+z_k)*z_k * sum_i w_i * f_ki
        # Let c_k = (1+z_k)*z_k
        c = np.zeros(n_cases)
        for i, (w0, wa) in enumerate(test_models):
            for j, z in enumerate(z_values):
                idx = i * n_z + j
                c[idx] = (1.0 + z) * z

        # true_vals already computed
        # Constraints: |c_k * f_k @ w - true_k| <= t * true_k
        # => c_k * f_k @ w - true_k <= t * true_k  =>  c_k*f_k @ w - t*true_k <= true_k
        # => -(c_k * f_k @ w - true_k) <= t * true_k  =>  -c_k*f_k @ w - t*true_k <= -true_k

        A_ub = np.zeros((2 * n_cases, 9))
        b_ub = np.zeros(2 * n_cases)
        for k in range(n_cases):
            tv = true_vals[k]
            if tv < 1e-10:
                continue
            A_ub[k, :8] = c[k] * f[k, :]
            A_ub[k, 8] = -tv
            b_ub[k] = tv
            A_ub[k + n_cases, :8] = -c[k] * f[k, :]
            A_ub[k + n_cases, 8] = -tv
            b_ub[k + n_cases] = -tv

        # Equality: sum(w) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])

        # Bounds: w_i >= 0 (use small epsilon), t >= 0
        bounds = [(1e-15, None)] * 8 + [(0, None)]

        try:
            res = linprog(c=np.zeros(9) + np.array([0]*8 + [1.0]),
                          A_ub=A_ub, b_ub=b_ub,
                          A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.success:
                w = res.x[:8]
                w = w / w.sum()
                return w
        except Exception:
            pass

        # Fallback: equal weights
        return np.full(8, 1.0 / 8.0)

    def node_objective(nodes_log, weights):
        """Objective for node optimization: nodes parameterized as log-spaced to ensure ordering."""
        if time.time() > deadline:
            return 1e10
        # Convert log-spaced parameters to actual nodes in (0,1)
        # Use cumulative softmax-like mapping
        raw = np.exp(nodes_log)
        nodes = np.cumsum(raw) / np.sum(raw)
        # Ensure strictly in (0,1)
        nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
        return compute_max_error(nodes, weights)

    # Initial configurations
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0

    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = compute_max_error(best_nodes, best_weights)

    # Multiple restarts
    np.random.seed(42)
    n_restarts = 6

    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break

        if r == 0:
            cur_nodes = gl8_nodes.copy()
        elif r == 1:
            cur_nodes = (np.arange(8) + 0.5) / 8.0
        else:
            cur_nodes = np.sort(np.random.uniform(0.05, 0.95, 8))

        # Iterate alternating optimization
        for it in range(15):
            if time.time() > deadline - 10.0:
                break

            # Step 1: Fix nodes, optimize weights via LP
            cur_weights = solve_weights_lp(cur_nodes)

            # Step 2: Fix weights, optimize nodes
            cur_err = compute_max_error(cur_nodes, cur_weights)

            # Parameterize nodes as log-spaced for smooth optimization
            # nodes = cumsum(exp(x)) / sum(exp(x))
            # Invert: x_i = log(nodes_i / nodes_{i-1} - 1) roughly, but simpler:
            # Just use the nodes directly with Nelder-Mead on constrained space
            def node_obj(x):
                if time.time() > deadline:
                    return 1e10
                n = np.sort(x)
                n = np.clip(n, 1e-8, 1.0 - 1e-8)
                # Ensure strict increase
                for k in range(1, 8):
                    if n[k] <= n[k-1] + 1e-12:
                        n[k] = n[k-1] + 1e-10
                return compute_max_error(n, cur_weights)

            try:
                res = minimize(node_obj, cur_nodes, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < cur_err:
                    cur_nodes = np.sort(res.x)
                    cur_nodes = np.clip(cur_nodes, 1e-8, 1.0 - 1e-8)
                    for k in range(1, 8):
                        if cur_nodes[k] <= cur_nodes[k-1] + 1e-12:
                            cur_nodes[k] = cur_nodes[k-1] + 1e-10
                    cur_err = compute_max_error(cur_nodes, cur_weights)
            except Exception:
                pass

        # Update best
        final_err = compute_max_error(cur_nodes, cur_weights)
        if final_err < best_err:
            best_err = final_err
            best_nodes = cur_nodes.copy()
            best_weights = cur_weights.copy()

    # Final LP polish on best nodes
    best_weights = solve_weights_lp(best_nodes)
    final_err = compute_max_error(best_nodes, best_weights)

    # Ensure validity
    if not np.all(np.diff(best_nodes) > 0) or best_nodes[0] <= 0 or best_nodes[7] >= 1:
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 1.0 / 8.0)
    if not np.all(best_weights > 0):
        best_weights = np.full(8, 1.0 / 8.0)

    best_weights = best_weights / best_weights.sum()

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
