"""Growth-index optimization via global grid + differential evolution with constraint handling."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    def Om_of_a(a, Om0, w0, wa):
        a = np.asarray(a, dtype=np.float64)
        Om_de = 1.0 - Om0
        de_factor = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        E2 = Om0/a**3 + Om_de*de_factor
        return Om0/a**3 / E2

    def compute_reference(Om0, w0, wa, a_grid):
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

        # Use a moderately fine grid for speed
        x_fine = np.linspace(x_min, x_max, 300)
        a_min = np.exp(x_min)
        y_init = [a_min, 1.0]

        sol = solve_ivp(ode, [x_min, x_max], y_init,
                       t_eval=x_fine,
                       method='DOP853', rtol=1e-10, atol=1e-12)

        if not sol.success:
            return None, None

        D_vals = np.interp(x_grid, sol.t, sol.y[0])
        dD_vals = np.interp(x_grid, sol.t, sol.y[1])

        D_ref = D_vals / D_vals[-1]
        dD_ref = dD_vals / D_vals[-1]
        f_ref = dD_ref / D_ref

        return f_ref, D_ref

    def model_gamma(c, a, Om0, w0, wa):
        q = 1.0 - Om_of_a(a, Om0, w0, wa)
        u = 1.0 + w0 + wa*(1.0-a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma

    def model_f(c, a, Om0, w0, wa):
        gamma = model_gamma(c, a, Om0, w0, wa)
        Om_m = Om_of_a(a, Om0, w0, wa)
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        return Om_m**gamma_clipped

    def model_D(c, a, Om0, w0, wa):
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

    # Use a dense a-grid for better reference
    a_grid = np.linspace(0.05, 1.0, 80)

    # Precompute references
    refs = {}
    for (Om0, w0, wa) in cosmologies:
        f_ref, D_ref = compute_reference(Om0, w0, wa, a_grid)
        if f_ref is not None:
            refs[(Om0, w0, wa)] = (f_ref, D_ref)

    if time.time() > DEADLINE:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    def fast_objective(c):
        max_err = 0.0
        for (Om0, w0, wa) in cosmologies:
            if (Om0, w0, wa) not in refs:
                continue
            f_ref, D_ref = refs[(Om0, w0, wa)]
            f_mod = model_f(c, a_grid, Om0, w0, wa)
            D_mod = model_D(c, a_grid, Om0, w0, wa)

            err_f = np.abs((f_mod - f_ref) / f_ref)
            err_D = np.abs((D_mod - D_ref) / D_ref)

            max_err = max(max_err, np.max(err_f), np.max(err_D))

        return max_err

    # Strategy: Global grid search to find good basins, then differential evolution for refinement
    # Coarse grid over the 6D parameter space

    # First, do a coarse grid search to identify promising regions
    # Focus on c0 in [0.3, 0.8], c1 in [-0.5, 0.5], c2 in [-0.3, 0.3],
    # c3 in [-0.3, 0.3], c4 in [-0.5, 0.5], c5 in [-0.3, 0.3]

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = fast_objective(best_c)

    if time.time() > DEADLINE - 20:
        return np.clip(best_c, -10, 10).tolist()

    # Coarse grid: 4 points per dimension for c1-c5, 5 for c0
    c0_vals = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
    c1_vals = np.array([-0.3, -0.1, 0.1, 0.3])
    c2_vals = np.array([-0.2, 0.0, 0.2])
    c3_vals = np.array([-0.2, 0.0, 0.2])
    c4_vals = np.array([-0.3, -0.1, 0.1, 0.3])
    c5_vals = np.array([-0.2, 0.0, 0.2])

    # Iterate through grid combinations (limit to stay within time)
    grid_iter = 0
    max_grid_iter = 500  # limit grid points to evaluate

    for c0 in c0_vals:
        for c1 in c1_vals:
            for c2 in c2_vals:
                for c3 in c3_vals:
                    for c4 in c4_vals:
                        for c5 in c5_vals:
                            if grid_iter >= max_grid_iter:
                                break
                            c = np.array([c0, c1, c2, c3, c4, c5])
                            err = fast_objective(c)
                            if err < best_err:
                                best_err = err
                                best_c = c.copy()
                            grid_iter += 1
                            if time.time() > DEADLINE - 15:
                                break
                        if time.time() > DEADLINE - 15:
                            break
                    if time.time() > DEADLINE - 15:
                        break
                if time.time() > DEADLINE - 15:
                    break
            if time.time() > DEADLINE - 15:
                break
        if time.time() > DEADLINE - 15:
            break

    if time.time() > DEADLINE - 10:
        return np.clip(best_c, -10, 10).tolist()

    # Now use differential evolution for global refinement
    bounds = [(-10, 10)] * 6
    # Constrain search to reasonable ranges
    bounds = [(0.1, 1.2), (-2.0, 2.0), (-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0)]

    try:
        result = differential_evolution(
            fast_objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mut=(0.5, 1.0),
            recomb=0.7,
            polish=True,
            init='sobol'
        )
        if result.fun < best_err:
            best_c = result.x
            best_err = result.fun
    except Exception:
        pass

    if time.time() > DEADLINE - 5:
        return np.clip(best_c, -10, 10).tolist()

    # Final polish: try a few random perturbations around best_c
    np.random.seed(123)
    for _ in range(5):
        if time.time() > DEADLINE - 3:
            break
        perturbation = np.random.normal(0, 0.05, 6)
        c_trial = best_c + perturbation
        c_trial = np.clip(c_trial, -10, 10)
        err = fast_objective(c_trial)
        if err < best_err:
            best_err = err
            best_c = c_trial.copy()

    best_c = np.clip(best_c, -10.0, 10.0)
    return best_c.tolist()
# EVOLVE-BLOCK-END
