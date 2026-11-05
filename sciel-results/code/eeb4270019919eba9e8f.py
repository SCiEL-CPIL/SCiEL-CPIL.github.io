"""
Dual-space Chebyshev projection for compact dark-energy growth approximation.

This program re-parameterizes the six-coefficient gamma ansatz by fitting on a
Chebyshev-normalized (q,u) grid, then solves a weighted least-squares problem
with explicit derivative (smoothness) regularization.  The ODE reference is
computed with looser tolerances to free compute budget for the fit.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 82

    def om_a(a, Om0, w0, wa):
        a_safe = np.maximum(np.asarray(a, dtype=float), 1e-10)
        E2 = Om0 * a_safe**(-3) + (1.0 - Om0) * a_safe**(-3*(1+w0)) * np.exp(3*wa*(a_safe - 1))
        return Om0 * a_safe**(-3) / E2

    def gamma_of(c, a, Om0, w0, wa):
        a = np.asarray(a, dtype=float)
        om = om_a(a, Om0, w0, wa)
        q = 1.0 - om
        u = 1.0 + w0 + wa * (1.0 - a)
        return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

    def f_of(c, a, Om0, w0, wa):
        a = np.asarray(a, dtype=float)
        om = om_a(a, Om0, w0, wa)
        g = gamma_of(c, a, Om0, w0, wa)
        return np.power(np.maximum(om, 1e-12), g)

    def d_approx(c, a, Om0, w0, wa):
        """Normalized growth factor via log-space trapezoid integration of f."""
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return np.array([])
        a_safe = np.maximum(a, 1e-6)
        x0 = np.log(a_safe[0])
        n_pts = max(400, a.size * 20)
        x_grid = np.linspace(x0, 0.0, n_pts)
        a_grid = np.exp(x_grid)
        f_grid = f_of(c, a_grid, Om0, w0, wa)
        # cumulative integral from x to 0
        integral = np.zeros(n_pts)
        integral[-1] = 0.0
        dx = np.diff(x_grid)
        for i in range(n_pts - 2, -1, -1):
            integral[i] = integral[i+1] + 0.5 * (f_grid[i] + f_grid[i+1]) * dx[i]
        # interpolate back to original a values
        x_orig = np.log(a_safe)
        D = np.exp(-np.interp(x_orig, x_grid, integral))
        return D

    def solve_ode(a_vals, Om0, w0, wa):
        a_vals = np.asarray(a_vals, dtype=float)
        if a_vals.size == 0:
            return np.array([]), np.array([])
        a_min = max(a_vals.min(), 1e-6)
        x_start = np.log(a_min)
        om_1 = om_a(1.0, Om0, w0, wa)
        y0 = [om_1**0.55, 1.0]

        def ode(x, y):
            ax = np.exp(x)
            omx = om_a(ax, Om0, w0, wa)
            fx = y[0]
            dfdx = -fx*fx - 1.5*(1.0 - omx)*fx + 1.5
            dDdx = fx * y[1]
            return [dfdx, dDdx]

        sol = solve_ivp(ode, [x_start, 0.0], y0, rtol=1e-6, atol=1e-8,
                        t_eval=np.sort(np.log(a_vals)))
        if not sol.success:
            return None, None
        f_vals = sol.y[0]
        D_vals = sol.y[1] / sol.y[1][-1]
        # map back to original order
        orig_order = np.argsort(np.log(a_vals))
        sort_idx = np.argsort(orig_order)
        return f_vals[sort_idx], D_vals[sort_idx]

    # --- Build reference data on a dense grid ---
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -0.5, -0.5),
        (0.25, -1.0, 0.3),
        (0.35, -1.2, 0.4),
        (0.30, -0.8, -0.3),
    ]
    a_dense = np.linspace(0.05, 0.95, 60)

    ref_f = []
    ref_D = []
    ref_q = []
    ref_u = []
    ref_a = []
    ref_wa = []

    for Om0, w0, wa in models:
        f_ref, D_ref = solve_ode(a_dense, Om0, w0, wa)
        if f_ref is None:
            continue
        om = om_a(a_dense, Om0, w0, wa)
        q = 1.0 - om
        u = 1.0 + w0 + wa * (1.0 - a_dense)
        ref_f.append(f_ref)
        ref_D.append(D_ref)
        ref_q.append(q)
        ref_u.append(u)
        ref_a.append(a_dense)
        ref_wa.append(np.full_like(a_dense, wa))

    if not ref_f:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    ref_f = np.concatenate(ref_f)
    ref_D = np.concatenate(ref_D)
    ref_q = np.concatenate(ref_q)
    ref_u = np.concatenate(ref_u)
    ref_a = np.concatenate(ref_a)
    ref_wa = np.concatenate(ref_wa)

    n_pts = ref_f.size

    # --- Chebyshev-normalized grid mapping ---
    # Map q and u to [-1, 1] for Chebyshev node placement
    q_lo, q_hi = ref_q.min(), ref_q.max()
    u_lo, u_hi = ref_u.min(), ref_u.max()
    if q_hi - q_lo < 1e-12:
        q_hi = q_lo + 1e-6
    if u_hi - u_lo < 1e-12:
        u_hi = u_lo + 1e-6

    # Build design matrix A such that gamma = c . row
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    A = np.column_stack([
        np.ones(n_pts),
        ref_q,
        ref_q**2,
        ref_u,
        ref_wa * ref_a,
        ref_u * ref_q,
    ])

    # --- Chebyshev weight (B-splines / equioscillation approximation) ---
    # Weight points near the edges of the (q,u) domain more heavily
    q_norm = 2.0 * (ref_q - q_lo) / (q_hi - q_lo) - 1.0
    u_norm = 2.0 * (ref_u - u_lo) / (u_hi - u_lo) - 1.0
    w_cheb = 1.0 / np.sqrt(np.maximum(1.0 - q_norm**2, 1e-8))
    w_cheb *= 1.0 / np.sqrt(np.maximum(1.0 - u_norm**2, 1e-8))
    w_cheb /= w_cheb.mean()

    # --- Weighted least-squares with derivative regularization ---
    # Target: gamma_target = ln(f_ref) / ln(om)  (only where om < 1 and > 0)
    om_ref = om_a(ref_a, *[m for m in [0]]*0)  # placeholder; recompute properly
    # Recompute om for all points
    om_all = np.empty(n_pts)
    idx = 0
    for i, (Om0, w0, wa) in enumerate(models):
        cnt = a_dense.size
        om_all[idx:idx+cnt] = om_a(a_dense, Om0, w0, wa)
        idx += cnt

    valid = (om_all > 0.01) & (om_all < 0.9999)
    if valid.sum() < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    gamma_target = np.where(valid, np.log(ref_f[valid]) / np.log(np.maximum(om_all[valid], 1e-12)), 0.55)
    gamma_target = np.clip(gamma_target, 0.1, 1.2)

    # Build weighted system: minimize || W*(A c - gamma_target) ||^2 + lambda * ||deriv penalty||^2
    # Derivative penalty: penalize changes in gamma along a (smoothness)
    # Approximate dgamma/da via finite differences on the dense grid
    dgamma_da = np.zeros(n_pts)
    idx2 = 0
    for i, (Om0, w0, wa) in enumerate(models):
        cnt = a_dense.size
        g_block = gamma_target[idx2:idx2+cnt]
        if cnt > 1:
            dgamma_da[idx2:idx2+cnt-1] = np.diff(g_block) / np.diff(a_dense[:cnt-1])
        idx2 += cnt

    # Regularization matrix R: penalize large dgamma/da
    R = np.diag(dgamma_da)
    lam = 0.5  # smoothness strength

    W = np.diag(w_cheb)
    WA = W @ A
    Wg = W @ gamma_target

    # Solve (WA^T WA + lam R^T R) c = WA^T Wg
    A_t_A = WA.T @ WA + lam * (R.T @ R)
    rhs = WA.T @ Wg

    try:
        c_ls = np.linalg.solve(A_t_A, rhs)
    except np.linalg.LinAlgError:
        c_ls = np.linalg.lstsq(A_t_A, rhs, rcond=None)[0]

    c_ls = np.clip(c_ls, -10, 10)

    # --- Refine with Nelder-Mead on the true max-relative-error objective ---
    def max_rel_err(c):
        if time.time() > deadline - 5:
            return 1e10
        max_e = 0.0
        for Om0, w0, wa in models:
            f_ref_m, D_ref_m = solve_ode(a_dense, Om0, w0, wa)
            if f_ref_m is None:
                continue
            f_app = f_of(c, a_dense, Om0, w0, wa)
            D_app = d_approx(c, a_dense, Om0, w0, wa)
            err_f = np.abs(f_app - f_ref_m) / np.maximum(np.abs(f_ref_m), 1e-10)
            err_D = np.abs(D_app - D_ref_m) / np.maximum(np.abs(D_ref_m), 1e-10)
            m = max(err_f.max(), err_D.max())
            if m > max_e:
                max_e = m
            # gamma bounds check
            g = gamma_of(c, a_dense, Om0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                return 1e10
        return 100.0 * max_e

    best_c = c_ls.copy()
    best_score = max_rel_err(best_c)

    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break
        if s == 0:
            c_init = best_c.copy()
        else:
            c_init = best_c + rng.normal(0, 0.05, size=6)
            c_init = np.clip(c_init, -10, 10)
        try:
            res = minimize(max_rel_err, c_init, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if time.time() > deadline - 10:
                break
            if res.fun < best_score:
                best_score = res.fun
                best_c = np.clip(res.x, -10, 10)
        except Exception:
            pass

    return list(np.clip(best_c, -10, 10))

# EVOLVE-BLOCK-END
