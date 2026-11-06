"""Global differential-evolution search over 6-coefficient growth-index ansatz."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

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

        # Use a fine grid for integration
        x_fine = np.linspace(x_min, x_max, 1000)
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

    def model_gamma(c, a, Om0, w0, wa):
        """Compute gamma from the 6-coefficient model."""
        q = 1.0 - Om_of_a(a, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def model_f(c, a, Om0, w0, wa):
        """Compute f from the model."""
        gamma = model_gamma(c, a, Om0, w0, wa)
        Om_m = Om_of_a(a, Om0, w0, wa)
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
        """Compute normalized D from the model via quadrature."""
        n = len(a)
        f_vals = model_f(c, a, Om0, w0, wa)
        x = np.log(a)
        dx = np.diff(x)
        trap = 0.5*(f_vals[:-1] + f_vals[1:]) * dx
        cum = np.zeros(n)
        cum[-2] = trap[-1]
        for i in range(n-3, -1, -1):
            cum[i] = cum[i+1] + trap[i]
        D_norm = np.zeros(n)
        D_norm[-1] = 1.0
        D_norm[:-1] = np.exp(-cum)
        return D_norm

    # Define validation cosmologies
    cosmologies = [
        (0.30, -0.5, 0.0),
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.2),
        (0.30, -0.9, -0.2),
        (0.25, -0.8, 0.1),
        (0.35, -1.0, 0.0),
        (0.30, -0.7, 0.3),
        (0.28, -1.1, -0.1),
    ]

    # Use a moderately dense a-grid for reference (80 points for speed)
    a_grid = np.linspace(0.05, 1.0, 80)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Precompute arrays for fast vectorized evaluation
    n_cosmo = len(cosmologies)
    n_a = len(a_grid)
    # Store reference arrays in flat form for fast max
    ref_f_flat = np.array([refs[(Om0,w0,wa)][0] for (Om0,w0,wa) in cosmologies if (Om0,w0,wa) in refs]).ravel()
    ref_D_flat = np.array([refs[(Om0,w0,wa)][1] for (Om0,w0,wa) in cosmologies if (Om0,w0,wa) in refs]).ravel()

    # Precompute q, u, wa*a for each cosmology to speed up model evaluation
    precomp = []
    for (Om0, w0, wa) in cosmologies:
        if (Om0, w0, wa) not in refs:
            continue
        q = 1.0 - Om_of_a(a_grid, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a_grid)
        wa_a = wa * a_grid
        Om_m = Om_of_a(a_grid, Om0, w0, wa)
        precomp.append((q, u, wa_a, Om_m))

    def hard_objective_vec(c):
        """Vectorized hard max-error objective."""
        max_err = 0.0
        for idx, (q, u, wa_a, Om_m) in enumerate(precomp):
            gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa_a + c[5]*u*q
            gamma_clipped = np.clip(gamma, 0.1, 1.2)
            f_mod = Om_m ** gamma_clipped

            # D via cumulative trapezoid
            x = np.log(a_grid)
            dx = np.diff(x)
            trap = 0.5*(f_mod[:-1] + f_mod[1:]) * dx
            cum = np.zeros(n_a)
            cum[-2] = trap[-1]
            for i in range(n_a-3, -1, -1):
                cum[i] = cum[i+1] + trap[i]
            D_mod = np.zeros(n_a)
            D_mod[-1] = 1.0
            D_mod[:-1] = np.exp(-cum)

            # Relative errors
            f_ref_slice = ref_f_flat[idx*n_a:(idx+1)*n_a]
            D_ref_slice = ref_D_flat[idx*n_a:(idx+1)*n_a]

            err_f = np.abs((f_mod - f_ref_slice) / f_ref_slice)
            err_D = np.abs((D_mod - D_ref_slice) / D_ref_slice)

            local_max = max(np.max(err_f), np.max(err_D))
            if local_max > max_err:
                max_err = local_max

        return max_err

    # Global search using differential evolution
    bounds = [(-10.0, 10.0)] * 6
    np.random.seed(123)

    # Run differential evolution with a reasonable number of iterations
    # popsize=15, maxiter=300, tol=1e-8
    def de_objective(c):
        """Wrapper for differential_evolution."""
        return hard_objective_vec(c)

    # Check time before starting DE
    if time.time() > DEADLINE - 20:
        # Fallback: just use a good initial guess
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.clip(best_c, -10, 10).tolist()

    # Run differential evolution
    de_result = differential_evolution(
        de_objective,
        bounds=bounds,
        seed=42,
        maxiter=200,
        popsize=12,
        tol=1e-7,
        polish=True,
        init='sobol',
    )

    best_c = de_result.x
    best_err = de_result.fun

    # Verify with hard objective
    hard_val = hard_objective_vec(best_c)
    if hard_val < best_err:
        best_err = hard_val

    if time.time() > DEADLINE - 5:
        return np.clip(best_c, -10, 10).tolist()

    # Local polish with Nelder-Mead on the hard objective
    res = minimize(
        hard_objective_vec,
        best_c,
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    hard_val2 = hard_objective_vec(res.x)
    if hard_val2 < best_err:
        best_c = res.x
        best_err = hard_val2

    # One more L-BFGS-B polish if time allows
    if time.time() < DEADLINE - 2:
        res2 = minimize(
            hard_objective_vec,
            best_c,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        hard_val3 = hard_objective_vec(res2.x)
        if hard_val3 < best_err:
            best_c = res2.x
            best_err = hard_val3

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
