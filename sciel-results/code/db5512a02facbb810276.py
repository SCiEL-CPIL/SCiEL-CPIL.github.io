import numpy as np
import scipy.integrate
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 85.0

    # Representative cosmology grid for design
    w0_vals = np.linspace(-1.2, 0.5, 12)
    w1_vals = np.linspace(-1.5, 1.5, 12)
    z_vals = np.concatenate([
        np.linspace(0.01, 0.1, 4),
        np.linspace(0.2, 1.0, 8),
        np.linspace(1.5, 3.0, 8),
        np.linspace(3.5, 6.0, 4)
    ])

    exact_ints = []
    model_params = []

    for w0 in w0_vals:
        for w1 in w1_vals:
            for z in z_vals:
                def integrand(u, w0=w0, w1=w1, z=z):
                    t = z * u
                    log_E = 3.0 * (1.0 + w0) * np.log1p(t) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + t))
                    return np.exp(-log_E)
                try:
                    val, _ = scipy.integrate.quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))
                except Exception:
                    us = np.linspace(0, 1, 10001)
                    ts = z * us
                    log_E = 3.0 * (1.0 + w0) * np.log1p(ts) + 3.0 * w1 * (1.0 - 1.0 / (1.0 + ts))
                    vals = np.exp(-log_E)
                    val = np.trapz(vals, us)
                    exact_ints.append(val)
                    model_params.append((w0, w1, z))

    exact_ints = np.array(exact_ints)
    w0_arr = np.array([p[0] for p in model_params])
    w1_arr = np.array([p[1] for p in model_params])
    z_arr = np.array([p[2] for p in model_params])

    def params_to_nodes_weights(params):
        theta = params[:8]
        alpha = params[8:16]
        u = 1.0 / (1.0 + np.exp(-theta))
        alpha_max = np.max(alpha)
        exp_alpha = np.exp(alpha - alpha_max)
        w = exp_alpha / np.sum(exp_alpha)
        return u, w

    def objective(params):
        u, w = params_to_nodes_weights(params)
        t = z_arr[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.max(rel_errors)

    def objective_smooth(params, temp=0.01):
        u, w = params_to_nodes_weights(params)
        t = z_arr[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        max_err = np.max(rel_errors)
        if max_err > 0:
            scaled = (rel_errors - max_err) / temp
            sum_exp = np.sum(np.exp(scaled))
            if sum_exp > 0:
                return max_err + temp * np.log(sum_exp)
        return max_err

    def objective_l2(params):
        u, w = params_to_nodes_weights(params)
        t = z_arr[:, None] * u[None, :]
        log_E = 3.0 * (1.0 + w0_arr)[:, None] * np.log1p(t) + 3.0 * w1_arr[:, None] * (1.0 - 1.0 / (1.0 + t))
        f_vals = np.exp(-log_E)
        quad_sums = np.sum(f_vals * w[None, :], axis=1)
        rel_errors = np.abs(quad_sums - exact_ints) / exact_ints
        return np.mean(rel_errors**2)

    # Initial guesses: multiple starting points
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    u0 = 0.5 + 0.5 * gl_nodes
    w0_init = 0.5 * gl_weights

    u0_safe = np.clip(u0, 1e-9, 1 - 1e-9)
    theta0 = np.log(u0_safe / (1 - u0_safe))
    w0_init_safe = np.clip(w0_init, 1e-9, 1.0)
    alpha0 = np.log(w0_init_safe)
    p0 = np.concatenate([theta0, alpha0])

    # Additional starting points with perturbation
    np.random.seed(42)
    start_points = [p0.copy()]
    for _ in range(3):
        pert = p0 + np.random.randn(16) * 0.1
        start_points.append(pert)

    best_params = p0
    best_score = objective(p0)

    for sp in start_points:
        if time.time() - start_time > time_limit - 15:
            break
        try:
            # L2 minimization first (smooth, differentiable)
            res_l2 = scipy.optimize.minimize(
                objective_l2, sp, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if objective(res_l2.x) < best_score:
                best_params = res_l2.x
                best_score = objective(res_l2.x)

            # Soft-max refinement
            res_sm = scipy.optimize.minimize(
                lambda p: objective_smooth(p, temp=0.005), best_params,
                method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if objective(res_sm.x) < best_score:
                best_params = res_sm.x
                best_score = objective(res_sm.x)

            # Powell on raw max
            res_pw = scipy.optimize.minimize(
                objective, best_params, method='Powell',
                options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-14}
            )
            if res_pw.fun < best_score:
                best_params = res_pw.x
                best_score = res_pw.fun

        except Exception:
            pass

    if time.time() - start_time > time_limit - 2:
        u_final, w_final = params_to_nodes_weights(best_params)
        sort_idx = np.argsort(u_final)
        u_sorted = u_final[sort_idx]
        w_sorted = w_final[sort_idx]
        w_norm = w_sorted / np.sum(w_sorted)
        return {"nodes": u_sorted, "weights": w_norm}

    u_final, w_final = params_to_nodes_weights(best_params)
    sort_idx = np.argsort(u_final)
    u_sorted = u_final[sort_idx]
    w_sorted = w_final[sort_idx]
    w_norm = w_sorted / np.sum(w_sorted)
    return {"nodes": u_sorted, "weights": w_norm}
