"""Direct constrained least-squares regression for 6-coefficient growth-index ansatz."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a for CPL dark energy."""
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
        """Compute reference f and D/D(1) using GR ODE."""
        n = len(a_grid)
        x_grid = np.log(a_grid)

        def ode(x, y):
            a = np.exp(x)
            E2 = Om0/a**3 + (1-Om0)*a**(-3*(1+w0+wa))*np.exp(-3*wa*(1-a))
            E = np.sqrt(E2)
            Om_m = Om0/a**3 / E2
            dD = y[1]
            d2D = -2.0*E*dD + 1.5*Om_m*y[0]
            return [dD, d2D]

        x_min = x_grid[0]
        x_max = 0.0

        x_fine = np.linspace(x_min, x_max, 2000)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-13, atol=1e-15)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        f_ref = dD_ref / D_ref

        return f_ref, D_ref

    # Dense validation cosmologies
    cosmologies = [
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -0.9, -0.2),
        (0.25, -0.8, 0.1),
        (0.35, -1.0, 0.0),
        (0.30, -0.7, 0.3),
        (0.28, -1.1, -0.1),
        (0.32, -1.0, 0.1),
        (0.27, -0.9, 0.15),
        (0.33, -1.0, -0.15),
        (0.29, -0.8, -0.25),
    ]

    # Dense a-grid for reference
    a_grid = np.linspace(0.05, 1.0, 150)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute arrays for fast vectorized evaluation
    valid_cosmos = [(Om0, w0, wa) for (Om0, w0, wa) in cosmologies if (Om0, w0, wa) in refs]
    n_cosmo = len(valid_cosmos)
    n_a = len(a_grid)

    ref_f_flat = np.array([refs[(Om0, w0, wa)][0] for (Om0, w0, wa) in valid_cosmos]).ravel()
    ref_D_flat = np.array([refs[(Om0, w0, wa)][1] for (Om0, w0, wa) in valid_cosmos]).ravel()

    # Precompute q, u, wa*a for each cosmology
    precomp = []
    for (Om0, w0, wa) in valid_cosmos:
        q = 1.0 - Om_of_a(a_grid, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a_grid)
        wa_a = wa * a_grid
        Om_m = Om_of_a(a_grid, Om0, w0, wa)
        precomp.append((q, u, wa_a, Om_m))

    x_grid = np.log(a_grid)
    dx = np.diff(x_grid)

    def hard_objective_vec(c):
        """Vectorized hard max-error objective."""
        max_err = 0.0
        for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            f_mod = Om_m ** gamma_clipped

            # D via cumulative trapezoid
            trap = 0.5*(f_mod[:-1] + f_mod[1:]) * dx
            cum = np.zeros(n_a)
            cum[-2] = trap[-1]
            for i in range(n_a-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
            D_mod = np.zeros(n_a)
            D_mod[-1] = 1.0
            D_mod[:-1] = np.exp(-cum)

            f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
            D_ref_slice = ref_D_flat[idx*n_a:(idx+1)*n_a]

            err_f = np.abs((f_mod - f_ref_slice) / f_ref_slice)
            err_D = np.abs((D_mod - D_ref_slice) / D_ref_slice)

            local_max = max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max

        return max_err

    def residual_vector(c):
        """Return the full residual vector for least_squares."""
        residuals = []
        for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            f_mod = Om_m ** gamma_clipped

            trap = 0.5*(f_mod[:-1] + f_mod[1:]) * dx
            cum = np.zeros(n_a)
            cum[-2] = trap[-1]
            for i in range(n_a-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
            D_mod = np.zeros(n_a)
            D_mod[-1] = 1.0
            D_mod[:-1] = np.exp(-cum)

            f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
            D_ref_slice = ref_D_flat[idx*n_a:(idx+1)*n_a]

            residuals.append((f_mod - f_ref_slice) / f_ref_slice)
            residuals.append((D_mod - D_ref_slice) / D_ref_slice)

        return np.concatenate(residuals)

    # Multi-start local refinement
    bounds = [(-10.0, 10.0)] * 6
    initial_guesses = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.1, 0.05, 0.0, 0.0, 0.0],
        [0.55, 0.05, -0.05, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, -0.05, 0.0, 0.0],
        [0.5, 0.05, 0.05, 0.0, 0.05, 0.0],
        [0.6, -0.05, 0.0, 0.05, -0.05, 0.0],
    ]

    best_c = None
    best_err = np.inf

    for guess in initial_guesses:
        if time.time() > DEADLINE - 5:
            break
        try:
            res = least_squares(
                residual_vector,
                np.array(guess),
                bounds=(-10.0, 10.0),
                method='trf',
                max_nfev=500,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            hard_val = hard_objective_vec(res.x)
            if hard_val < best_err:
                best_err = hard_val
                best_c = res.x.copy()
        except Exception:
            pass

    if best_c is None:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Final polish with L-BFGS-B on the hard objective
    if time.time() < DEADLINE - 2:
        from scipy.optimize import minimize
        res2 = minimize(
            hard_objective_vec,
            best_c,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        hard_val2 = hard_objective_vec(res2.x)
        if hard_val2 < best_err:
            best_c = res2.x
            best_err = hard_val2

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
