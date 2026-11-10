"""Growth-index fit: reproduce the GR reference ODE internally and least-squares fit 6 coefficients."""
import time
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---- deadline guard (90 s wall-clock kill) ----
_T0 = time.time()
_DEADLINE = _T0 + 78.0          # safety margin before 90 s

def _time_ok():
    return time.time() < _DEADLINE

# ---- cosmology helpers (contract definitions) ----
def _Om_of_a(a, Om0, wa):
    return Om0 / (Om0 + (1.0 - Om0) * a ** (-3.0 * (1.0 + wa)))

def _f_of_a(a, Om0, wa, c):
    Om = _Om_of_a(a, Om0, wa)
    q = 1.0 - Om
    u = 1.0 + wa * (1.0 - a)
    g = c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q
    g = min(1.2, max(0.1, g))
    return Om ** g

# ---- reference GR ODE (the evaluator's independent reference) ----
def _reference_solution(Om0, wa, a_grid, tol=1e-9, rtol=1e-9):
    # y = [D, D'];  dD/da = D';  dD'/da = (3/2*Om - 1/2) D - (3/(2a)) D'
    def rhs(a, y):
        Om = _Om_of_a(a, Om0, wa)
        return np.array([y[1], (1.5 * Om - 0.5) * y[0] - (1.5 / a) * y[1]])
    a_lo, a_hi = float(a_grid[0]), float(a_grid[-1])
    sol = solve_ivp(rhs, (a_lo, a_hi), [1.0, 1.0], t_eval=a_grid,
                    method='DOP853', rtol=rtol, atol=tol)
    if not sol.success:
        raise RuntimeError("reference ODE failed")
    D = sol.y[0]
    Dp = sol.y[1]
    f_ref = (1.5 * _Om_of_a(a_grid, Om0, wa) - 0.5) - (1.5 / a_grid) * (Dp / D)
    return f_ref, D

# ---- objective: 100 * max relative error over f and normalized D ----
def _objective(c, models, a_grid, cache):
    worst = 0.0
    for (Om0, wa) in models:
        key = (Om0, wa)
        if key not in cache:
            cache[key] = _reference_solution(Om0, wa, a_grid)
        f_ref, D_ref = cache[key]
        f_app = np.array([_f_of_a(a, Om0, wa, c) for a in a_grid])
        D_app = np.exp(-np.trapz(f_app[::-1], a_grid[::-1]))
        Dn_ref = D_ref / D_ref[-1]
        Dn_app = D_app / D_app[-1]
        err_f = float(np.max(np.abs((f_app - f_ref) / f_ref)))
        err_D = float(np.max(np.abs((Dn_app - Dn_ref) / Dn_ref)))
        worst = max(worst, err_f, err_D)
    return 100.0 * worst

# ---- validation-style model set (disjoint CPL dark-energy cosmologies) ----
_MODELS = [
    (0.30, 0.0, 0.0),   # flat LCDM
    (0.30, 0.0, -0.5),  # CPL
    (0.25, -0.5, 0.5),  # CPL
    (0.30, -1.0, 0.0),  # cosmological constant
    (0.32, 0.5, -1.0),  # CPL
    (0.28, -0.3, 0.7),  # CPL
    (0.30, -0.9, 0.4),  # CPL
    (0.27, 0.2, -0.8),  # CPL
]

# grid in ln a (matches the evaluator's integration variable)
_LN_A = np.linspace(math.log(0.05), 0.0, 160)
_A_GRID = np.exp(_LN_A)

def construct():
    global _DEADLINE
    cache = {}
    c_best = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best = _objective(c_best, _MODELS, _A_GRID, cache)

    rng = np.random.default_rng(20240517)
    bounds = [(-10.0, 10.0)] * 6

    # random restarts + Nelder-Mead polish, bounded by deadline
    restarts = 0
    while _time_ok() and restarts < 40:
        restarts += 1
        c0 = c_best + rng.normal(0.0, 0.25, 6)
        c0 = np.clip(c0, -10.0, 10.0)
        try:
            res = minimize(_objective, c0, args=(_MODELS, _A_GRID, cache),
                           method='Nelder-Mead',
                           options={'maxiter': 1200, 'xatol': 1e-7,
                                    'fatol': 1e-9, 'adaptive': True})
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = np.clip(res.x, -10.0, 10.0)
                val = _objective(cand, _MODELS, _A_GRID, cache)
                if np.isfinite(val) and val < best:
                    best = val
                    c_best = cand
        except Exception:
            pass
        if not _time_ok():
            break

    out = [float(v) for v in np.clip(c_best, -10.0, 10.0)]
    return out
