"""Chebyshev spectral collocation quadrature design for luminosity distance."""
import numpy as np
import time
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # High-accuracy reference using 256-point Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(256)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    # Diverse test set: broader range of z, w0, wa
    z_vals = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    w0_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    wa_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    n_tests = len(test_cases)

    # Precompute reference values
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

    # Precompute test case arrays
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    def _compute_inv_E_matrix(nodes):
        """Compute 1/E(a) matrix for all test cases at given nodes."""
        a_vals = 1.0 / (1.0 + z_arr[:, None] * nodes[None, :])
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        log_terms = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) \
                   - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_terms = np.clip(log_terms, -700, 700)
        term_L = Omega_L * np.exp(log_terms)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        return 1.0 / E

    def _worst_rel_error(nodes, weights):
        """Worst-case relative error in percent."""
        inv_E = _compute_inv_E_matrix(nodes)
        qvals = inv_E @ weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        return np.max(rel_errs)

    def _soft_max_rel_error(nodes, weights, beta=50.0):
        """Smooth approximation of worst-case relative error using log-sum-exp."""
        inv_E = _compute_inv_E_matrix(nodes)
        qvals = inv_E @ weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        max_err = np.max(rel_errs)
        if max_err < 1e-15:
            return 0.0
        scaled = beta * (rel_errs - max_err)
        log_sum_exp = max_err + (1.0 / beta) * np.log(np.sum(np.exp(scaled)))
        return log_sum_exp

    # Start with Gauss-Legendre as initial guess
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0

    best_nodes = gl8_nodes.copy()
    best_weights = gl8_weights.copy()
    best_err = _worst_rel_error(best_nodes, best_weights)

    # === CHEBYSHEV SPECTRAL COLLOCATION APPROACH ===
    # Use Chebyshev nodes which minimize Runge phenomenon and have
    # excellent spectral convergence properties. The key insight is that
    # for smooth integrands, Chebyshev collocation points with appropriate
    # weights provide near-optimal quadrature.

    # Chebyshev-Gauss-Lobatto nodes (including endpoints)
    def _chebyshev_gauss_lobatto(n):
        """Chebyshev-Gauss-Lobatto nodes in [0,1] with weights."""
        k = np.arange(n)
        x = 0.5 * (1.0 + np.cos(np.pi * k / (n - 1)))
        x = x[::-1]  # increasing order
        # Weights for CGL quadrature
        w = np.full(n, np.pi / (n - 1))
        w[0] = w[0] / 2
        w[-1] = w[-1] / 2
        return x, w

    # Chebyshev roots (Gauss-Chebyshev)
    def _chebyshev_gauss(n):
        """Chebyshev-Gauss nodes in [0,1] with weights."""
        k = np.arange(n)
        x = 0.5 * (1.0 + np.cos(np.pi * (2 * k + 1) / (2 * n)))
        x = x[::-1]  # increasing order
        w = np.full(n, np.pi / n)
        return x, w

    # Generate diverse node sets for initial exploration
    node_sets = []
    
    # Gauss-Legendre
    node_sets.append(gl8_nodes.copy())
    
    # Chebyshev-Gauss
    cg_nodes, cg_weights = _chebyshev_gauss(8)
    node_sets.append(cg_nodes.copy())
    
    # Chebyshev-Gauss-Lobatto (interior points only, since we need 8 in (0,1))
    cgl_nodes, cgl_weights = _chebyshev_gauss_lobatto(10)
    # Take interior points (excluding endpoints)
    interior = cgl_nodes[1:-1]
    if len(interior) == 8:
        node_sets.append(interior.copy())
    
    # Various parameterized Chebyshev-like distributions
    for alpha in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        k = np.arange(8)
        # Generalized Chebyshev: x_k = 0.5*(1+cos(pi*(2k+1)/(2n) * alpha))
        x = 0.5 * (1.0 + np.cos(np.pi * (2 * k + 1) * alpha / (2 * 8)))
        x = x[::-1]
        x = np.clip(x, 1e-6, 1.0 - 1e-6)
        if np.all(np.diff(x) > 1e-8):
            node_sets.append(x.copy())
    
    # Beta-distributed nodes
    for a, b in [(1.5, 1.5), (2, 2), (2, 3), (3, 2), (1.5, 2.5), (2.5, 1.5), (3, 3), (1.2, 1.2)]:
        try:
            from scipy.stats import beta as beta_dist
            quantiles = np.linspace(0.02, 0.98, 8)
            nodes_beta = beta_dist.ppf(quantiles, a, b)
            if np.all(np.diff(nodes_beta) > 1e-8) and np.all(nodes_beta > 1e-4) and np.all(nodes_beta < 1 - 1e-4):
                node_sets.append(nodes_beta.copy())
        except:
            pass

    # Power-law spaced
    for p in [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        t = np.linspace(0.03, 0.97, 8)
        nodes_pow = t ** p
        if np.all(np.diff(nodes_pow) > 1e-8):
            node_sets.append(nodes_pow.copy())

    # Evaluate each node set with optimal weights
    for nodes in node_sets:
        if time.time() > deadline - 30.0:
            break
        try:
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            for i in range(1, 8):
                if nodes[i] <= nodes[i-1]:
                    nodes[i] = nodes[i-1] + 1e-6
            nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
            
            inv_E = _compute_inv_E_matrix(nodes)
            w, _, _, _ = np.linalg.lstsq(inv_E, ref_vals, rcond=None)
            
            if np.all(w > 0):
                w = w / np.sum(w)
                err = _worst_rel_error(nodes, w)
                if err < best_err:
                    best_err = err
                    best_nodes = nodes.copy()
                    best_weights = w.copy()
        except Exception:
            pass

    # Phase 2: Local refinement with smooth soft-max objective using Chebyshev-inspired parameterization
    if time.time() < deadline - 15.0:
        try:
            def _param_to_nodes_weights(params):
                """Convert parameter vector to nodes and weights using Chebyshev-inspired mapping."""
                # Use Chebyshev-like parameterization: nodes = 0.5*(1+cos(theta))
                # where theta is a smooth function of parameters
                log_gaps = params[:7]
                log_first = params[7]
                log_weights = params[8:16]
                
                gap_sizes = np.exp(log_gaps)
                x = np.zeros(8)
                x[0] = log_first
                for i in range(7):
                    x[i+1] = x[i] + gap_sizes[i]
                
                # Map to [0,1] using sigmoid, then apply Chebyshev warping
                nodes = 1.0 / (1.0 + np.exp(-x))
                nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
                for i in range(1, 8):
                    if nodes[i] <= nodes[i-1]:
                        nodes[i] = nodes[i-1] + 1e-10
                
                w = np.exp(log_weights)
                w = w / np.sum(w)
                
                return nodes, w

            def _objective_smooth(params):
                """Smooth objective using soft-max."""
                nodes, weights = _param_to_nodes_weights(params)
                return _soft_max_rel_error(nodes, weights, beta=50.0)

            # Convert best to parameter space
            x_best = np.log(best_nodes / (1.0 - best_nodes))
            params_init = np.zeros(16)
            params_init[7] = x_best[0]
            for i in range(7):
                gap = x_best[i+1] - x_best[i]
                if gap < 1e-12:
                    gap = 1e-12
                params_init[i] = np.log(gap)
            params_init[8:16] = np.log(np.maximum(best_weights, 1e-12))
            
            bounds = [(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 8
            
            # Multiple restarts from different initial points
            for restart in range(3):
                if time.time() > deadline - 15.0:
                    break
                try:
                    if restart == 0:
                        p_init = params_init.copy()
                    elif restart == 1:
                        # Perturb from Chebyshev-Gauss
                        p_init = params_init.copy()
                        p_init[:7] += np.random.normal(0, 0.1, 7)
                        p_init[8:16] += np.random.normal(0, 0.1, 8)
                    else:
                        # Perturb from Gauss-Legendre
                        x_gl = np.log(gl8_nodes / (1.0 - gl8_nodes))
                        p_init = np.zeros(16)
                        p_init[7] = x_gl[0]
                        for i in range(7):
                            gap = x_gl[i+1] - x_gl[i]
                            if gap < 1e-12:
                                gap = 1e-12
                            p_init[i] = np.log(gap)
                        p_init[8:16] = np.log(gl8_weights)
                    
                    res = so.minimize(
                        _objective_smooth,
                        p_init,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11},
                    )
                    
                    if res.fun < best_err * 1.01:
                        nodes_smooth, weights_smooth = _param_to_nodes_weights(res.x)
                        err_smooth = _worst_rel_error(nodes_smooth, weights_smooth)
                        if err_smooth < best_err:
                            best_err = err_smooth
                            best_nodes = nodes_smooth.copy()
                            best_weights = weights_smooth.copy()
                except Exception:
                    pass
        except Exception:
            pass

    # Phase 3: Differential evolution for global search
    if time.time() < deadline - 10.0:
        try:
            def _de_objective(params):
                nodes, weights = _param_to_nodes_weights(params)
                return _worst_rel_error(nodes, weights)

            bounds_de = [(-4.0, 4.0)] * 7 + [(-10.0, 10.0)] + [(-10.0, 10.0)] * 8

            # Use differential evolution with moderate population
            res_de = so.differential_evolution(
                _de_objective,
                bounds=bounds_de,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42,
            )
            
            if res_de.fun < best_err:
                best_nodes, best_weights = _param_to_nodes_weights(res_de.x)
                best_err = res_de.fun
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 2.0:
        try:
            inv_E = _compute_inv_E_matrix(best_nodes)
            w_final, _, _, _ = np.linalg.lstsq(inv_E, ref_vals, rcond=None)
            if np.all(w_final > 0):
                w_final = w_final / np.sum(w_final)
                err_final = _worst_rel_error(best_nodes, w_final)
                if err_final < best_err:
                    best_err = err_final
                    best_weights = w_final.copy()
        except Exception:
            pass

    # Final cleanup
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
