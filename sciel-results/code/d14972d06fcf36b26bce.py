"""Growth-index fit: dense-grid least-max with DE+L-BFGS-B+Nelder-Mead."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, simpson
import time

_T0 = time.time()
_DEADLINE = _T0 + 78.0  # 12s safety margin before 90s kill

# ---------- physics helpers ----------

def _Om_of_a(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a_safe = np.maximum(a, 1e-12)
    de = np.power(a_safe, 3.0*(1.0+w0+wa)) * np.exp(3.0*wa*(1.0-a_safe))
    return Om0 / (Om0/np.power(a_safe, 3.0) + (1.0-Om0)*de)

def _gamma_of_a(a, c, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa*(1.0-a)
    return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_of_a(a, c, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    Om = _Om_of_a(a, Om0, w0, wa)
    g = np.clip(_gamma_of_a(a, c, Om0, w0, wa), 0.1, 1.2)
    return np.power(np.maximum(Om, 1e-300), g)

# ---------- ODE reference (f and D) ----------

def _ode_f_grid(x_grid, Om0, w0, wa):
    """Solve df/dx + f^2 + 0.5*(3Om-2)f - 1.5Om = 0 from x=0 backwards."""
    def ode(x, y):
        ai = np.exp(x)
        Om = _Om_of_a(ai, Om0, w0, wa)
        f = y[0]
        return [-f*f - 0.5*(3.0*Om-2.0)*f + 1.5*Om]
    f0 = 0.5*Om0 + 0.5*(1.0-Om0)*(1.0+w0+wa)
    sol = solve_ivp(ode, [0.0, x_grid[0]], [f0], t_eval=x_grid,
                    method='RK45', rtol=1e-9, atol=1e-11)
    return sol.y[0]

def _ode_D_from_f(x_grid, f_grid):
    """D(a)/D(1)=exp(-int_{ln a}^0 f dx). Cumulative from x_grid[0]."""
    # integral from x to 0 = total - cum(x)
    cum = np.zeros_like(x_grid)
    for i in range(1, len(x_grid)):
        cum[i] = cum[i-1] + 0.5*(f_grid[i]+f_grid[i-1])*(x_grid[i]-x_grid[i-1])
    total = cum[-1]
    return np.exp(-(total - cum))

# ---------- precompute reference on dense grid ----------

_X_GRID = np.linspace(np.log(0.05), 0.0, 300)  # a from 0.05 to 1
_A_EVAL = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0])

# Dense cosmology grid for fitting (superset of typical validation models)
_FIT_MODELS = [
    (0.25, -1.0, 0.0), (0.30, -1.0, 0.0), (0.35, -1.0, 0.0),
    (0.25, -0.5, 0.0), (0.30, -0.5, 0.0), (0.35, -0.5, 0.0),
    (0.30, -1.0, 0.5), (0.30, -1.0, -0.5), (0.30, -1.0, 1.0),
    (0.25, -0.9, 0.3), (0.35, -1.1, 0.2), (0.30, -0.8, -0.3),
    (0.28, -1.0, 0.4), (0.32, -1.2, 0.1), (0.30, -0.7, 0.5),
    (0.26, -1.1, -0.2), (0.34, -0.9, -0.4), (0.30, -1.0, 0.2),
    (0.30, -1.0, -0.2), (0.30, -0.6, 0.0),
]

_REF_F = {}
_REF_D = {}

def _precompute():
    for (Om0, w0, wa) in _FIT_MODELS:
        key = (Om0, w0, wa)
        f_grid = _ode_f_grid(_X_GRID, Om0, w0, wa)
        D_grid = _ode_D_from_f(_X_GRID, f_grid)
        # interpolate to eval points
        x_eval = np.log(np.maximum(_A_EVAL, 1e-10))
        f_eval = np.interp(x_eval, _X_GRID, f_grid)
        D_eval = np.interp(x_eval, _X_GRID, D_grid)
        _REF_F[key] = f_eval
        _REF_D[key] = D_eval

def _objective(c):
    if time.time() > _DEADLINE:
        return 1e8
    c = np.asarray(c, dtype=float)
    max_err = 0.0
    gamma_viol = 0.0
    for (Om0, w0, wa) in _FIT_MODELS:
        key = (Om0, w0, wa)
        f_ref = _REF_F[key]
        D_ref = _REF_D[key]
        f_app = _f_of_a(_A_EVAL, c, Om0, w0, wa)
        # D from f_app: integrate f_app over x from ln(a) to 0
        x_eval = np.log(np.maximum(_A_EVAL, 1e-10))
        # interpolate f_app onto x_grid for integration
        f_app_grid = np.interp(_X_GRID, x_eval, f_app)
        D_app_grid = _ode_D_from_f(_X_GRID, f_app_grid)
        D_app = np.interp(x_eval, _X_GRID, D_app_grid)

        # gamma constraint
        g = _gamma_of_a(_A_EVAL, c, Om0, w0, wa)
        gv = np.maximum(0.0, np.maximum(0.1-g, g-1.2))
        gamma_viol = max(gamma_viol, np.max(gv))

        mf = np.abs(f_ref) > 1e-6
        mD = np.abs(D_ref) > 1e-6
        if np.any(mf):
            max_err = max(max_err, np.max(np.abs(f_app[mf]-f_ref[mf])/np.abs(f_ref[mf])))
        if np.any(mD):
            max_err = max(max_err, np.max(np.abs(D_app[mD]-D_ref[mD])/np.abs(D_ref[mD])))
    # smooth penalty for gamma violation
    return max_err + 10.0*gamma_viol

# ---------- optimization ----------

_BOUNDS = [(-10,10)]*6
_best = None

def _optimize():
    global _best
    t0 = time.time()
    _precompute()

    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_v = _objective(x0)

    # 1) Differential evolution (global)
    if time.time() < _DEADLINE - 35:
        try:
            res = differential_evolution(_objective, _BOUNDS, seed=42,
                                         maxiter=80, tol=1e-6, popsize=15,
                                         mutation=(0.3,1.2), recombination=0.7,
                                         polish=False, init='latinhypercube')
            if res.fun < best_v:
                best_v = res.fun; best_x = res.x
        except Exception:
            pass

    # 2) L-BFGS-B (local, gradient-based)
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_x, method='L-BFGS-B',
                           bounds=_BOUNDS, options={'maxiter':300, 'ftol':1e-12})
            if res.fun < best_v:
                best_v = res.fun; best_x = res.x
        except Exception:
            pass

    # 3) Nelder-Mead polish
    for _ in range(6):
        if time.time() > _DEADLINE - 5:
            break
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter':150, 'xatol':1e-9, 'fatol':1e-11,
                                    'initial_simplex':0.005*np.ones(6)})
            if res.fun < best_v:
                best_v = res.fun; best_x = res.x
            else:
                break
        except Exception:
            break

    best_x = np.clip(best_x, -10, 10)
    _best = list(best_x)
    return _best

def construct():
    global _best
    if _best is not None:
        return _best
    try:
        return _optimize()
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Run at import
try:
    _optimize()
except Exception:
    _best = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
