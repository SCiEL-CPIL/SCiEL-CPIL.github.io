"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    
    Strategy:
    1. Precompute high-accuracy reference integrals for a diverse CPL model set.
    2. Formulate the quadrature design as a linear least-squares problem for weights
       given fixed nodes, and an alternating optimization over nodes and weights.
    3. Use differential evolution for global node search and Nelder-Mead for local polish.
    4. Ensure positivity and strict ordering of nodes.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))

    # Diverse set of CPL models (w0, wa)
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

    # Test redshifts
    z_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Precompute true integrals
    n_models = len(test_models)
    n_z = len(z_values)
    true_vals = np.zeros((n_models, n_z))
    for i, (w0, wa) in enumerate(test_models):
        for j, z in enumerate(z_values):
            def integrand(zp):
                return 1.0 / E_cpl(zp, w0, wa)
            result, _ = quad(integrand, 0, z, limit=200)
            true_vals[i, j] = (1.0 + z) * result

    def compute_max_error(nodes, weights):
        """Compute worst-case relative error given nodes and weights."""
        max_err = 0.0
        for i in range(n_models):
            w0, wa = test_models[i]
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                approx = (1.0 + z) * z * np.dot(weights, 1.0 / E_vals)
                tv = true_vals[i, j]
                if tv > 1e-10:
                    err = abs(approx - tv) / tv
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_weights_for_nodes(nodes):
        """
        Given fixed nodes, find weights that minimize the worst-case relative error
        using a linear least-squares approach with a sum-to-1 constraint.
        """
        # Build the design matrix A and target vector b
        # For each (model, z) pair: approx = (1+z) * z * sum_i w_i / E(z*nodes_i)
        # We want approx ≈ true_val
        # So: sum_i w_i * [(1+z)*z/E(z*nodes_i)] ≈ true_val
        # This is a linear system in w_i.
        
        n_samples = n_models * n_z
        A = np.zeros((n_samples, 8))
        b = np.zeros(n_samples)
        idx = 0
        for i in range(n_models):
            w0, wa = test_models[i]
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                row_vals = (1.0 + z) * z / E_vals
                A[idx, :] = row_vals
                b[idx] = true_vals[i, j]
                idx += 1
        
        # Solve least squares with constraint sum(w) = 1
        # Use Lagrange multiplier or simply solve and normalize
        # First, try unconstrained LS
        try:
            w_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            w_ls = np.full(8, 1.0/8.0)
        
        # If any weights are negative, we need a constrained solution
        # For now, normalize and clip (simple approach)
        w_ls = np.clip(w_ls, 1e-10, None)
        w_ls = w_ls / w_ls.sum()
        
        return w_ls

    def objective_bounded(x):
        """x = [n1,...,n8, log_w1,...,log_w8] with 0<n_i<1, w_i>0, sum(w)=1."""
        if time.time() > deadline:
            return 1e10
        nodes = x[:8]
        # Ensure strictly increasing
        for k in range(1, 8):
            if nodes[k] <= nodes[k-1]:
                return 1e10
        if nodes[0] <= 0 or nodes[7] >= 1:
            return 1e10
        
        # Use the log-weights from x, but also try the LS-optimized weights
        log_w = x[8:16]
        w_param = np.exp(log_w)
        w_param = w_param / w_param.sum()
        
        # Compute error with parametric weights
        err_param = compute_max_error(nodes, w_param)
        
        # Also compute error with LS-optimized weights for these nodes
        w_ls = solve_weights_for_nodes(nodes)
        err_ls = compute_max_error(nodes, w_ls)
        
        # Return the minimum of the two (we can choose either representation)
        return min(err_param, err_ls)

    # Bounds for differential evolution
    node_bounds = [(1e-6, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5),
                   (0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0 - 1e-6)]
    w_bounds = [(-10, 10)] * 8
    bounds = node_bounds + w_bounds

    # Initial population: Gauss-Legendre based
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    init_x = np.concatenate([gl_nodes, np.log(gl_weights)])

    # Run differential evolution for global search
    np.random.seed(42)
    best_x = init_x.copy()
    best_obj = objective_bounded(init_x)
    
    # Also track the best nodes and their LS-optimal weights separately
    best_nodes = init_x[:8].copy()
    best_weights = solve_weights_for_nodes(best_nodes)
    best_error = compute_max_error(best_nodes, best_weights)

    try:
        result = differential_evolution(
            objective_bounded,
            bounds,
            seed=42,
            maxiter=80,
            popsize=12,
            tol=1e-10,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol',
            x0=init_x,
            updating='immediate',
            workers=1,
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
            # Update best nodes and LS weights
            cand_nodes = result.x[:8]
            cand_weights = solve_weights_for_nodes(cand_nodes)
            cand_err = compute_max_error(cand_nodes, cand_weights)
            if cand_err < best_error:
                best_error = cand_err
                best_nodes = cand_nodes.copy()
                best_weights = cand_weights.copy()
    except Exception:
        pass

    # Local refinement with Nelder-Mead from multiple starting points
    if time.time() < deadline - 10.0:
        # Starting points: GL, DE result, uniform, random
        start_points = [best_x.copy()]
        
        # Uniform nodes
        uniform_nodes = (np.arange(8) + 0.5) / 8.0
        uniform_w = np.full(8, 1.0 / 8.0)
        start_points.append(np.concatenate([uniform_nodes, np.log(uniform_w)]))
        
        # Random points
        np.random.seed(123)
        for _ in range(3):
            rand_nodes = np.sort(np.random.uniform(0.01, 0.99, 8))
            rand_w = np.random.uniform(0.5, 1.5, 8)
            start_points.append(np.concatenate([rand_nodes, np.log(rand_w)]))
        
        for init_p in start_points:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    objective_bounded, init_p,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_x = res.x
                    # Update best nodes and LS weights
                    cand_nodes = res.x[:8]
                    cand_weights = solve_weights_for_nodes(cand_nodes)
                    cand_err = compute_max_error(cand_nodes, cand_weights)
                    if cand_err < best_error:
                        best_error = cand_err
                        best_nodes = cand_nodes.copy()
                        best_weights = cand_weights.copy()
            except Exception:
                pass

    # Final: use the best nodes found, and compute their LS-optimal weights
    final_weights = solve_weights_for_nodes(best_nodes)
    
    # Ensure validity
    if not np.all(np.diff(best_nodes) > 0) or best_nodes[0] <= 0 or best_nodes[7] >= 1:
        best_nodes = (np.arange(8) + 0.5) / 8.0
        final_weights = np.full(8, 1.0 / 8.0)
    if not np.all(final_weights > 0):
        final_weights = np.full(8, 1.0 / 8.0)

    final_weights = final_weights / final_weights.sum()

    return {"nodes": best_nodes, "weights": final_weights}
# EVOLVE-BLOCK-END
