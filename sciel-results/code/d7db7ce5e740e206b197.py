"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # ---- Representative validation cosmologies (dense grid) ----
    om_m_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    w0_values = [-1.2, -1.0, -0.8]
    wa_values = [-0.3, 0.0, 0.3]
    test_models = [(om, w0, wa) for om in om_m_values for w0 in w0_values for wa in wa_values]

    # Common evaluation grid in a
    a_eval = np.linspace(0.1, 1.0, 100)
    ln_a_eval = np.log(a_eval)
    a_rev = a_eval[::-1]
    ln_a_rev = ln_a_eval[::-1]

    # ---- Precompute reference data (GR growth ODE) ----
    ref_data = []
    for om_m, w0, wa in test_models:
        if time.time() > deadline - 5.0:
            break
        exp_wa = 3.0 * (1.0 + w0 + wa)

        def ode_rhs(ln_a, f, om_m=om_m, w0=w0, wa=wa, exp_wa=exp_wa):
            a = np.exp(ln_a)
            de_term = (1.0 - om_m) * (a ** exp_wa)
            mat_term = om_m * (a ** 3)
            e2 = mat_term + de_term
            om_a = mat_term / e2 if e2 > 0 else 0.0
            coeff = 1.5 * (1.0 + om_a) + (2.0 / 3.0) * w0 + (2.0 / 3.0) * wa * (1.0 - a)
            return -f ** 2 - f * coeff + 0.5 * om_a

        a_init = 0.1
        ln_a_init = np.log(a_init)
        de_init = (1.0 - om_m) * (a_init ** exp_wa)
        mat_init = om_m * (a_init ** 3)
        e2_init = mat_init + de_init
        om_a_init = mat_init / e2_init if e2_init > 0 else 0.0
        f_init = om_a_init ** 0.55

        try:
            sol = solve_ivp(ode_rhs, (ln_a_init, 0.0), [f_init],
                            t_eval=ln_a_eval, rtol=1e-9, atol=1e-11)
            if not sol.success:
                continue
            f_ref = sol.y[0]
            # D(a)/D(1) = exp(int_0^{ln a} f dx); integrate backwards from x=0
            int_vals = np.concatenate(([0.0], np.cumtrapz(f_ref[::-1], ln_a_rev)))
            int_vals = int_vals[::-1]
            D_ref = np.exp(int_vals)
        except Exception:
            continue

        # Precompute model inputs on the grid
        de_term = (1.0 - om_m) * (a_eval ** exp_wa)
        mat_term = om_m * (a_eval ** 3)
        e2 = mat_term + de_term
        om_a = mat_term / e2
        q = 1.0 - om_a
        u = 1.0 + w0 + wa * (1.0 - a_eval)
        ref_data.append({
            'f_ref': f_ref, 'D_ref': D_ref,
            'om_a': om_a, 'q': q, 'u': u, 'wa_a': wa * a_eval,
            'uq': u * q, 'q2': q * q,
        })

    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---- Fast vectorized error (max relative error over f and D, x100) ----
    def calculate_error_fast(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        for data in ref_data:
            om_a = data['om_a']
            q = data['q']
            u = data['u']
            gamma = c0 + c1 * q + c2 * data['q2'] + c3 * u + c4 * data['wa_a'] + c5 * data['uq']
            # smooth gamma-bound penalty (quadratic)
            lo = np.maximum(0.1 - gamma, 0.0)
            hi = np.maximum(gamma - 1.2, 0.0)
            pen = np.max(lo * lo + hi * hi)
            f_approx = om_a ** gamma
            err_f = np.max(np.abs(f_approx - data['f_ref']) / np.maximum(np.abs(data['f_ref']), 1e-8))
            # D_approx via trapezoidal integration of f_approx over x
            int_vals = np.concatenate(([0.0], np.cumtrapz(f_approx[::-1], ln_a_rev)))
            int_vals = int_vals[::-1]
            D_approx = np.exp(int_vals)
            err_D = np.max(np.abs(D_approx - data['D_ref']) / np.maximum(np.abs(data['D_ref']), 1e-8))
            max_err = max(max_err, err_f, err_D)
        return 100.0 * max_err + 1000.0 * pen

    def objective(x):
        if time.time() > deadline:
            return 1e10
        return calculate_error_fast(x)

    # ---- Multi-start Nelder-Mead ----
    x0_list = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.05, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.05, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
    ]

    best_x = x0_list[0]
    best_score = calculate_error_fast(best_x)

    for s in x0_list:
        if time.time() > deadline - 8.0:
            break
        try:
            res = minimize(objective, np.array(s, dtype=float), method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass

    best_x = np.clip(best_x, -10.0, 10.0)

    # ---- Final high-precision polish pass ----
    if time.time() < deadline - 6.0:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
        best_x = np.clip(best_x, -10.0, 10.0)

    # ---- Safety: if gamma constraint badly violated, fall back to constant ----
    try:
        g_min = 1.2
        g_max = 0.1
        for data in ref_data:
            gamma = (best_x[0] + best_x[1] * data['q'] + best_x[2] * data['q2']
                     + best_x[3] * data['u'] + best_x[4] * data['wa_a'] + best_x[5] * data['uq'])
            g_min = min(g_min, float(np.min(gamma)))
            g_max = max(g_max, float(np.max(gamma)))
        if g_min < 0.05 or g_max > 1.25:
            best_x = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        best_x = np.clip(best_x, -10.0, 10.0)

    return list(best_x)

# EVOLVE-BLOCK-END
