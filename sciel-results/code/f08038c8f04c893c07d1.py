import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    t_start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # CPL model: w(z) = w0 + wa*z/(1+z)
    # E(z) = H(z)/H0 for flat universe
    # E(z)^2 = (1+z)^3 + (1+z)^(3+3*w0) * exp(3*wa*(1 - 1/(1+z)))
    #        = (1+z)^3 + (1+z)^(3(1+w0)) * exp(3*wa*z/(1+z))
    # 1/E(z) = 1/sqrt(E_sq)

    def inv_E(zp, w0, wa):
        """Compute 1/E(z) = 1/H(z)/H0 for flat CPL cosmology, vectorized."""
        if np.isscalar(zp):
            zp = np.array([zp])
        t = zp
        # Handle t=0 case
        out = np.ones_like(t, dtype=float)
        mask = t > 1e-15
        if np.any(mask):
            tm = t[mask]
            log1pt = np.log1p(tm)
            # log_term1 = 3 * log(1+t)
            log_term1 = 3.0 * log1pt
            # log_term2 = (3 + 3*w0)*log(1+t) + 3*wa*t/(1+t)
            log_term2 = (3.0 + 3.0 * w0) * log1pt + 3.0 * wa * (tm / (1.0 + tm))
            m = np.maximum(log_term1, log_term2)
            sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
            log_E_sq = m + np.log(sum_exp)
            out[mask] = np.exp(-0.5 * log_E_sq)
        if np.ndim(out) == 0:
            return out.item()
        return out

    def reference_integral(z, w0, wa):
        """Compute dL/(c/H0) = (1+z) * integral_0^z dz'/E(z') via adaptive quad"""
        def integrand(zp):
            return inv_E(zp, w0, wa)
        val, _ = quad(integrand, 0, z, limit=200, epsabs=1e-13, epsrel=1e-13)
        return (1.0 + z) * val

    def quadrature_value(z, nodes, weights, w0, wa):
        """Compute (1+z)*z*sum_i w_i/E(a_i) where a_i=1/(1+z*nodes[i])"""
        zp_vals = z * nodes
        E_inv_vals = inv_E(pzp_vals, w0, wa)
        return (1.0 + z) * z * np.sum(weights * E_inv_vals)

    # Generate representative cosmologies - broader range
    w0_vals = np.linspace(-1.5, 0.5, 8)
    wa_vals = np.linspace(-1.5, 1.5, 8)
    cosmologies = [(w0, wa) for w0 in w0_vals for wa in wa_vals]

    # z values to test - logarithmic + linear spread
    z_test = np.unique(np.concatenate([
        np.logspace(np.log10(0.05), np.log10(1.0), 12),
        np.linspace(1.0, 5.0, 12)
    ]))

    # Precompute reference values
    ref_data = []  # (z, w0, wa, ref_val)
    for (w0, wa) in cosmologies:
        for z in z_test:
            rv = reference_integral(z, w0, wa)
            ref_data.append((z, w0, wa, rv))

    n_ref = len(ref_data)
    ref_z = np.array([d[0] for d in ref_data])
    ref_w0 = np.array([d[1] for d in ref_data])
    ref_wa = np.array([d[2] for d in ref_data])
    ref_val = np.array([d[3] for d in ref_data])

    def compute_errors(nodes, weights):
        """Compute vector of relative errors for all reference cases."""
        # nodes shape (8,), weights shape (8,)
        # zp_vals shape (n_ref, 8)
        zp_vals = ref_z[:, None] * nodes[None, :]
        # inv_E vectorized: shape (n_ref, 8)
        t = zp_vals
        out = np.ones_like(t, dtype=float)
        mask = t > 1e-15
        if np.any(mask):
            tm = t[mask]
            log1pt = np.log1p(tm)
            w0_col = ref_w0[:, None]
            wa_col = ref_wa[:, None]
            # Need to handle this carefully for 2D
            # Flatten, compute, reshape
            flat_t = tm.ravel()
            flat_w0 = np.repeat(ref_w0, 8)[mask.ravel()]
            flat_wa = np.repeat(ref_wa, 8)[mask.ravel()]
            log_term1 = 3.0 * np.log1p(flat_t)
            log_term2 = (3.0 + 3.0 * flat_w0) * np.log1p(flat_t) + 3.0 * flat_wa * (flat_t / (1.0 + flat_t))
            m = np.maximum(log_term1, log_term2)
            sum_exp = np.exp(log_term1 - m) + np.exp(log_term2 - m)
            log_E_sq = m + np.log(sum_exp)
            out[mask] = np.exp(-0.5 * log_E_sq).reshape(-1, 8)[mask.reshape(-1, 8).any(axis=1)] if False else np.exp(-0.5 * log_E_sq)
        # Simpler: compute directly
        out = np.ones((n_ref, 8), dtype=float)
        for i in range(n_ref):
            zp_i = ref_z[i] * nodes
            out[i] = inv_E(zp_i, ref_w0[i], ref_wa[i])
        
        quad_sums = np.sum(out * weights[None, :], axis=1)
        quad_vals = (1.0 + ref_z) * ref_z * quad_sums
        rel_errors = np.abs(quad_vals - ref_val) / np.maximum(ref_val, 1e-30)
        return rel_errors

    def objective_minmax(params):
        """Minimize max relative error."""
        nodes = params[:8]
        log_weights = params[8:]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        
        rel_errors = compute_errors(nodes, weights)
        return np.max(rel_errors)

    def objective_sum_squares(params):
        """Minimize sum of squared relative errors."""
        nodes = params[:8]
        log_weights = params[8:]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        
        rel_errors = compute_errors(nodes, weights)
        return np.sum(rel_errors**2)

    def objective_smooth_max(params, lam=50.0):
        """Smooth approximation of max error."""
        nodes = params[:8]
        log_weights = params[8:]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        
        rel_errors = compute_errors(nodes, weights)
        # Smooth max: (1/lam)*log(sum(exp(lam*errors)))
        max_e = np.max(rel_errors)
        if max_e > 5.0:
            return max_e
        return (1.0/lam) * np.log(np.sum(np.exp(lam * rel_errors)))

    # Initial guess: Gauss-Legendre nodes and weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    init_nodes = (gl_nodes + 1.0) / 2.0
    init_weights = gl_weights / 2.0
    init_log_w = np.log(init_weights)
    init_params = np.concatenate([init_nodes, init_log_w])

    best_params = init_params.copy()
    best_max_err = objective_minmax(init_params)
    best_sum_sq = objective_sum_squares(init_params)

    t_check = time.time()
    if t_check - t_start < DEADLINE - 20:
        # Phase 1: Smooth max optimization with L-BFGS-B
        try:
            res = minimize(objective_smooth_max, init_params, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
            if res.fun < 1e5:
                cand_params = res.x
                cand_max = objective_minmax(cand_params)
                if cand_max < best_max_err:
                    best_max_err = cand_max
                    best_params = cand_params.copy()
        except Exception:
            pass

    t_check = time.time()
    if t_check - t_start < DEADLINE - 15:
        # Phase 2: Sum of squares refinement
        try:
            res2 = minimize(objective_sum_squares, best_params, method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1500})
            if res2.fun < 1e5:
                cand_params = res2.x
                cand_max = objective_minmax(cand_params)
                if cand_max < best_max_err:
                    best_max_err = cand_max
                    best_params = cand_params.copy()
        except Exception:
            pass

    t_check = time.time()
    if t_check - t_start < DEADLINE - 10:
        # Phase 3: Powell for direct max-error minimization
        try:
            res3 = minimize(objective_minmax, best_params, method='Powell',
                           options={'maxiter': 50, 'ftol': 1e-14, 'xtol': 1e-12})
            if res3.fun < best_max_err:
                best_max_err = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass

    t_check = time.time()
    if t_check - t_start < DEADLINE - 5:
        # Phase 4: Nelder-Mead local refinement
        try:
            res4 = minimize(objective_minmax, best_params, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            if res4.fun < best_max_err:
                best_max_err = res4.fun
                best_params = res4.x.copy()
        except Exception:
            pass

    t_check = time.time()
    if t_check - t_start < DEADLINE - 3:
        # Phase 5: Random restarts with smooth max
        np.random.seed(123)
        for _ in range(5):
            t_check = time.time()
            if t_check - t_start > DEADLINE - 3:
                break
            try:
                noise = np.random.randn(16) * 0.03
                pert_params = best_params + noise
                pert_params[:8] = np.sort(pert_params[:8])
                pert_params[:8] = np.clip(pert_params[:8], 0.005, 0.995)
                # Enforce strict increase
                for i in range(1, 8):
                    if pert_params[i] <= pert_params[i-1]:
                        pert_params[i] = pert_params[i-1] + 0.005
                pert_params[:8] = np.clip(pert_params[:8], 0.005, 0.995)
                
                obj_pert = objective_smooth_max(pert_params)
                if obj_pert < best_max_err + 0.01:
                    res5 = minimize(objective_minmax, pert_params, method='Powell',
                                   options={'maxiter': 30, 'ftol': 1e-14, 'xtol': 1e-12})
                    if res5.fun < best_max_err:
                        best_max_err = res5.fun
                        best_params = res5.x.copy()
            except Exception:
                pass

    # Extract final result
    final_nodes = best_params[:8].copy()
    final_log_w = best_params[8:].copy()
    final_weights = np.exp(final_log_w)
    final_weights = final_weights / np.sum(final_weights)

    # Ensure nodes are strictly increasing in (0,1)
    final_nodes = np.sort(final_nodes)
    final_nodes = np.clip(final_nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if final_nodes[i] <= final_nodes[i-1]:
            final_nodes[i] = final_nodes[i-1] + 1e-6
    final_nodes = np.clip(final_nodes, 1e-6, 1.0 - 1e-6)

    # Verify weights sum to 1
    final_weights = final_weights / np.sum(final_weights)

    return {"nodes": final_nodes, "weights": final_weights}
