"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin before 90s limit

    # Step 1: Gauss-Legendre initialization
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes_init = (gl_nodes + 1.0) / 2.0
    weights_init = gl_weights / 2.0
    nodes_init = np.sort(nodes_init)

    # Step 2: Define CPL E(a) function
    def make_E(Om, w0, w1):
        def E(a):
            a = np.asarray(a, dtype=float)
            a_safe = np.maximum(a, 1e-10)
            term_m = Om / a_safe**3
            term_de = (1 - Om) / a_safe**(3 * (1 + w0)) * np.exp(-3 * w1 * (1 - a_safe))
            return np.sqrt(term_m + term_de)
        return E

    # Step 3: Test cosmologies and redshifts - expanded for better coverage
    test_cosmos = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -1.0, 0.0),   # Cosmological constant
        (0.3, -0.5, -0.5),  # CPL
        (0.3, -1.5, 0.5),   # CPL
        (0.25, -0.8, 0.3),  # CPL
        (0.35, -1.2, 0.2),  # CPL
        (0.28, -0.9, -0.2), # CPL
        (0.32, -1.1, 0.1),  # CPL
        (0.27, -1.3, -0.3), # Additional CPL
        (0.33, -0.7, 0.4),  # Additional CPL
        (0.3, -2.0, 1.0),   # Extreme CPL
        (0.3, -0.3, -1.0),  # Extreme CPL
    ]

    z_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])

    # Pre-compute reference integrals for speed
    reference_data = []
    for (Om, w0, w1) in test_cosmos:
        E = make_E(Om, w0, w1)
        for z in z_vals:
            def integrand(u):
                a_val = 1.0 / (1.0 + z * u)
                return 1.0 / E(a_val)
            true_integral, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
            true_value = (1 + z) * z * true_integral
            reference_data.append((Om, w0, w1, z, true_value, E))

    n_ref = len(reference_data)

    # Step 4: Objective function with smooth max (log-sum-exp) for better optimization
    def objective(params):
        t = time.time()
        if t > deadline:
            return None

        n = 8
        nodes_p = params[:n]
        weights_p = params[n:]

        # Map nodes to (0,1)
        nodes_mapped = (np.tanh(nodes_p) + 1.0) / 2.0
        nodes_mapped = np.sort(nodes_mapped)

        # Map weights to positive, sum to 1
        weights_mapped = np.exp(weights_p - np.max(weights_p))
        weights_mapped = weights_mapped / np.sum(weights_mapped)

        # Compute relative errors for all reference cases
        rel_errors = np.zeros(n_ref)
        for idx, (Om, w0, w1, z, true_value, E) in enumerate(reference_data):
            a_i = 1.0 / (1.0 + z * nodes_mapped)
            E_vals = E(a_i)
            quad_value = (1 + z) * z * np.sum(weights_mapped / E_vals)

            if abs(true_value) > 1e-15:
                rel_errors[idx] = abs(quad_value - true_value) / abs(true_value)
            else:
                rel_errors[idx] = 0.0

        # Use smooth max via log-sum-exp with large beta for near-hard max
        beta = 50.0
        max_val = np.max(rel_errors)
        if max_val < 1e-15:
            return 0.0
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        scaled = beta * rel_errors
        scaled_max = np.max(scaled)
        log_sum_exp = scaled_max + np.log(np.sum(np.exp(scaled - scaled_max)))
        smooth_max = log_sum_exp / beta

        return smooth_max * 100.0

    # Step 5: Optimization with multiple restarts and methods
    nodes_params_init = np.arctanh(np.clip(2 * nodes_init - 1, -0.999, 0.999))
    weights_params_init = np.log(weights_init)
    params_init = np.concatenate([nodes_params_init, weights_params_init])

    best_params = params_init.copy()
    best_err = objective(best_params)

    if best_err is None:
        return {"nodes": nodes_init, "weights": weights_init}

    # Try multiple optimization methods with restarts
    methods_to_try = ['Nelder-Mead', 'Powell', 'COBYLA', 'L-BFGS-B']

    for method in methods_to_try:
        if time.time() > deadline - 15.0:
            break

        # Multiple restarts for each method
        n_restarts = 3 if method in ['Nelder-Mead', 'Powell'] else 2

        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break

            try:
                if restart == 0:
                    start_params = best_params.copy()
                else:
                    # Perturb from best found
                    noise_scale = 0.1 * (restart - 0.5)
                    start_params = best_params + np.random.randn(16) * noise_scale
                    # Keep nodes in reasonable range
                    start_params[:8] = np.clip(start_params[:8], -5, 5)

                options = {}
                if method == 'Nelder-Mead':
                    options = {'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                elif method == 'Powell':
                    options = {'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-12}
                elif method == 'COBYLA':
                    options = {'maxiter': 300, 'rhobeg': 1e-4}
                elif method == 'L-BFGS-B':
                    options = {'maxiter': 200, 'gtol': 1e-10}

                result = minimize(
                    objective,
                    start_params,
                    method=method,
                    options=options
                )

                if result.fun is not None and result.fun < best_err:
                    best_err = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass

    # Step 6: Final high-precision polish with tight tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun is not None and result.fun < best_err:
                best_err = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    # Step 7: Extract final nodes and weights
    nodes_final_raw = np.tanh(best_params[:8])
    nodes_final = np.sort((nodes_final_raw + 1.0) / 2.0)

    weights_final_raw = best_params[8:]
    weights_final = np.exp(weights_final_raw - np.max(weights_final_raw))
    weights_final = weights_final / np.sum(weights_final)

    # Ensure nodes are strictly in (0,1) and strictly increasing
    nodes_final = np.clip(nodes_final, 1e-8, 1 - 1e-8)
    nodes_final = np.sort(nodes_final)
    for i in range(1, len(nodes_final)):
        if nodes_final[i] <= nodes_final[i-1]:
            nodes_final[i] = nodes_final[i-1] + 1e-10

    # Ensure weights sum to 1 within 1e-12
    weights_final = weights_final / np.sum(weights_final)

    if time.time() > deadline:
        return {"nodes": nodes_init, "weights": weights_init}

    return {"nodes": nodes_final, "weights": weights_final}

# EVOLVE-BLOCK-END
