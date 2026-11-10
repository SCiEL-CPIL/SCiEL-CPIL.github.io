import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

# ---------------- physics helpers (search-time only) ----------------

def _om(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    u = 1.0 + w0 + wa * (1.0 - a)
    de = np.exp(-3.0 * a * u)
    return Om0 * a**3 / (Om0 * a**3 + (1.0 - Om0) * de)

def _gamma(a, c, Om0, w0, wa):
    c0, c1, c2, c3, c4, c5 = c
    om = _om(a, Om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _f(a, c, Om0, w0, wa):
    om = _om(a, Om0, w0, wa)
    return om ** _gamma(a, c, Om0, w0, wa)

def _fref(a, Om0, w0, wa):
    om = _om(a, Om0, w0, wa)
    q = 1.0 - om
    return om ** (0.55 + 0.05 * q)

def _D(a, c, Om0, w0, wa):
    x = np.linspace(np.log(a), 0.0, 1200)
    f = _f(np.exp(x), c, Om0, w0, wa)
    return np.exp(-np.trapz(f, x))

def _Dref(a, Om0, w0, wa):
    x = np.linspace(np.log(a), 0.0, 1200)
    f = _fref(np.exp(x), Om0, w0, wa)
    return np.exp(-np.trapz(f, x))

def _solve_ode(Om0, w0, wa):
    """Solve the GR growth ODE to get f and D at a grid of a."""
    from scipy.integrate import solve_ivp
    a_min = 0.05
    a_max = 1.0
    def rhs(x, y):
        a = np.exp(x)
        om = _om(a, Om0, w0, wa)
        u = 1.0 + w0 + wa * (1.0 - a)
        w = -u
        H2 = om + (1.0 - om) * np.exp(-3.0 * a * u)
        Hp = 1.5 * om - 1.5 * (1.0 - om) * np.exp(-3.0 * a * u) * (w + 1.0)
        Hp2 = 1.5 * (om - (1.0 - om) * np.exp(-3.0 * a * u) * (w + 1.0)) * (2.0 * w + 1.0)
        Dp = y[1]
        Dpp = 1.5 * om * Dp - 0.5 * (Hp2 / H2) * Dp - Hp * D
        return [Dp, Dpp]
    D0 = 0.02
    Dp0 = _fref(a_min, Om0, w0, wa) * D0
    sol = solve_ivp(rhs, [np.log(a_min), 0.0], [D0, Dp0],
                    dense_output=True, rtol=1e-9, atol=1e-12,
                    max_step=0.02, method='DOP853')
    a_grid = np.linspace(a_min, a_max, 120)
    x_grid = np.log(a_grid)
    Y = sol.sol(x_grid)
    D = Y[0]
    Dp = Y[1]
    f_ode = Dp / D
    Dn = D / D[-1]
    return a_grid, f_ode, Dn

def _build_eval_points():
    models = [(0.30, -1.0, 0.0), (0.30, -1.0, 0.3), (0.30, -1.2, 0.4),
              (0.25, -1.0, 0.2), (0.35, -0.9, 0.1), (0.30, -1.1, 0.35)]
    a_grid = np.linspace(0.05, 1.0, 120)
    pts = []
    for m in models:
        Om0, w0, wa = m
        try:
            ag, f_ode, Dn = _solve_ode(Om0, w0, wa)
        except Exception:
            continue
        # interpolate to a_grid
        f_interp = np.interp(a_grid, ag, f_ode)
        D_interp = np.interp(a_grid, ag, Dn)
        for i in range(len(a_grid)):
            pts.append((a_grid[i], f_interp[i], D_interp[i], Om0, w0, wa))
    return np.array(pts)

# ---------------- objective ----------------

def _objective(c, pts):
    a = pts[:, 0]; f_ref = pts[:, 1]; D_ref = pts[:, 2]
    Om0 = pts[:, 3]; w0 = pts[:, 4]; wa = pts[:, 5]
    om = _om(a, Om0, w0, wa)
    g = _gamma(a, c, Om0, w0, wa)
    f = om ** g
    D = _D(a, c, Om0, w0, wa)
    rel_f = np.abs(f - f_ref) / np.maximum(np.abs(f_ref), 1e-9)
    rel_D = np.abs(D - D_ref) / np.maximum(np.abs(D_ref), 1e-9)
    return np.max(np.concatenate([rel_f, rel_D]))

def construct():
    t0 = time.time()
    DEADLINE = 90.0
    MARGIN = 8.0

    # Build evaluation points (ODE reference)
    pts = _build_eval_points()
    if len(pts) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Start from the seed
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = x0.copy()
    best_val = _objective(x0, pts)

    def remaining():
        return DEADLINE - MARGIN - (time.time() - t0)

    # Global optimization (bounded, no constraints needed)
    bounds = [(-1.0, 1.5), (-1.0, 1.0), (-1.0, 1.0),
              (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    if remaining() > 10:
        try:
            res = differential_evolution(_objective, bounds, args=(pts,),
                                         seed=42, maxiter=200, tol=1e-7,
                                         popsize=15, init='sobol',
                                         polish=False, updating='deferred',
                                         workers=1)
            if res.success or res.fun < best_val:
                best_c = res.x
                best_val = res.fun
        except Exception:
            pass

    # Local polish with Nelder-Mead / L-BFGS-B
    if remaining() > 5:
        try:
            res = minimize(_objective, best_c, args=(pts,), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_val:
                best_c = res.x
                best_val = res.fun
        except Exception:
            pass

    # Clamp into required coefficient range [-10, 10]
    best_c = np.clip(best_c, -10.0, 10.0)

    # Safety: ensure gamma stays in [0.1, 1.2] over a broad check grid; if not,
    # fall back toward the seed by scaling the deviation.
    check_a = np.linspace(0.05, 1.0, 400)
    check_models = [(0.30, -1.0, 0.0), (0.30, -1.0, 0.3), (0.30, -1.2, 0.4),
                    (0.25, -1.0, 0.2), (0.35, -0.9, 0.1), (0.30, -1.1, 0.35)]
    ok = True
    for (Om0, w0, wa) in check_models:
        g = _gamma(check_a, best_c, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            ok = False
            break
    if not ok:
        # shrink deviation from seed until gamma is in range
        dev = best_c - x0
        scale = 1.0
        for _ in range(60):
            cand = x0 + scale * dev
            good = True
            for (Om0, w0, wa) in check_models:
                g = _gamma(check_a, cand, Om0, w0, wa)
                if np.any(g < 0.1) or np.any(g > 1.2):
                    good = False
                    break
            if good:
                best_c = cand
                break
            scale *= 0.8
        else:
            best_c = x0.copy()

    return [float(v) for v in best_c]
