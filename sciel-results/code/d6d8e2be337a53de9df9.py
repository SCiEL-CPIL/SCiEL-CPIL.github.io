"""Compact dark-energy growth approximation: fit 6 coefficients to a GR reference."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---------------- GR reference (independent numerical solution) ----------------
def _solve_growth(Om0, w0, wa, a_grid):
    """Return D(a) and f(a) = dlnD/dlna from the GR second-order ODE."""
    a0, a1 = a_grid[0], a_grid[-1]
    H2 = lambda a: Om0 / a**3 + (1.0 - Om0) * a ** (3.0 * (1.0 + w0 + wa))

    def rhs(x, y):
        # y = [D, D']  (prime = d/da)
        D, Dp = y
        H2v = H2(x)
        Hp = (-1.5 * Om0 / x**2 - 1.5 * (1.0 - Om0) * (1.0 + w0 + wa) * x ** (3.0 * (1.0 + w0 + wa)))
        return [Dp, (-3.0 / x - 1.5 * Hp / H2v) * Dp - 0.5 * D]

    y0 = [1.0, 0.5 * Om0 - 0.5]  # D(1)=1, D'(1)=0.5(Om-1)
    sol = solve_ivp(rhs, [a0, a1], y0, t_eval=a_grid, method="RK45", rtol=1e-10, atol=1e-12)
    D = sol.y[0]
    Dp = sol.y[1]
    f = Dp / D
    return D, f

# ---------------- Ansatz evaluation ----------------
def _gamma(c, a, Om0, w0, wa):
    Om = Om0 / a**3
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q**2 + c[3] * u + c[4] * wa * a + c[5] * u * q

def _f_ansatz(c, a, Om0, w0, wa):
    g = _gamma(c, a, Om0, w0, wa)
    Om = Om0 / a**3
    return Om ** g

def _D_ratio(c, a, Om0, w0, wa):
    # D(a)/D(1) = exp(-int_a^1 f(exp x) dx)
    g = _gamma(c, a, Om0, w0, wa)
    Om = Om0 / a**3
    # integrate f(exp x) dx from x=ln(a) to 0
    def integrand(x):
        aa = np.exp(x)
        Oom = Om0 / aa**3
        gg = _gamma(c, aa, Om0, w0, wa)
        return Oom ** gg
    val, _ = _quad(integrand, np.log(a), 0.0)
    return np.exp(-val)

def _quad(f, lo, hi):
    n = 40
    if hi == lo:
        return 0.0, 0.0
    xs = np.linspace(lo, hi, n)
    ys = np.array([f(x) for x in xs])
    h = (hi - lo) / (n - 1)
    s = h / 3.0 * (ys[0] + ys[-1] + 4.0 * ys[1:-1:2].sum() + 2.0 * ys[2:-1:2].sum())
    return s, 0.0

# ---------------- Objective: max relative error over validation set ----------------
def _max_err(c, models, a_grid):
    best = 0.0
    for (Om0, w0, wa) in models:
        D_ref, f_ref = _solve_growth(Om0, w0, wa, a_grid)
        f_an = _f_ansatz(c, a_grid, Om0, w0, wa)
        # relative error in f
        err_f = np.max(np.abs(f_an - f_ref) / np.maximum(np.abs(f_ref), 1e-8))
        best = max(best, err_f)
        # D ratio error
        for a in a_grid:
            D_an = _D_ratio(c, a, Om0, w0, wa)
            D_r = D_ref[np.argmin(np.abs(a_grid - a))]
            e = abs(D_an - D_r) / max(abs(D_r), 1e-8)
            if e > best:
                best = e
    return best

def construct():
    t_start = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Validation-like CPL models (disjoint from typical training set)
    models = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.5),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.0),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.30, -1.0, -0.3),
        (0.30, -0.8, 0.0),
    ]
    a_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Precompute references
    refs = {}
    for m in models:
        D, f = _solve_growth(*m, a_grid)
        refs[m] = (D, f)

    def objective(c):
        c = np.clip(c, -10.0, 10.0)
        best = 0.0
        for m in models:
            D_ref, f_ref = refs[m]
            Om0, w0, wa = m
            g = _gamma(c, a_grid, Om0, w0, wa)
            # gamma range constraint penalty
            if g.min() < 0.1 or g.max() > 1.2:
                pen = max(0.1 - g.min(), g.max() - 1.2)
                return best + 10.0 * pen
            f_an = (Om0 / a_grid**3) ** g
            err_f = np.max(np.abs(f_an - f_ref) / np.maximum(np.abs(f_ref), 1e-8))
            best = max(best, err_f)
            # D ratio
            for i, a in enumerate(a_grid):
                D_an = _D_ratio(c, a, Om0, w0, wa)
                e = abs(D_an - D_ref[i]) / max(abs(D_ref[i]), 1e-8)
                if e > best:
                    best = e
        return best

    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = c0.copy()
    best_val = objective(c0)

    # Nelder-Mead with time guard
    try:
        res = minimize(objective, c0, method="Nelder-Mead",
                       options={"maxiter": 4000, "maxfev": 4000,
                                "xatol": 1e-4, "fatol": 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_c = np.clip(res.x, -10.0, 10.0)
    except Exception:
        pass

    if time.time() - t_start > DEADLINE:
        return [float(x) for x in best_c]

    # Second pass: local refinement with tighter tolerance
    try:
        res2 = minimize(objective, best_c, method="Nelder-Mead",
                        options={"maxiter": 3000, "maxfev": 3000,
                                 "xatol": 1e-5, "fatol": 1e-6,
                                 "initial_simplex": 1e-3})
        if res2.fun < best_val:
            best_val = res2.fun
            best_c = np.clip(res2.x, -10.0, 10.0)
    except Exception:
        pass

    return [float(x) for x in best_c]
