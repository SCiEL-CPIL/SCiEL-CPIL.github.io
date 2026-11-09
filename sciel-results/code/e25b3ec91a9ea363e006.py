"""Growth-index compact ansatz tuned by smooth Chebyshev surrogate + Nelder-Mead."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 80.0  # safety margin before 90s kill
_SEED = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# --- Validation cosmology grid (self-contained reference) ---
_OM_GRID = [0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.76, 0.84]
_W0_GRID = [-1.3, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
_WA_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
_A_GRID = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.33, 0.42, 0.52, 0.63, 0.75, 0.85, 0.93, 1.0])

# --- Exact reference solution via GR second-order ODE ---
def _true_f_D(Om0, w0, wa, a_grid):
    """Return (f_true, Dnorm_true) on a_grid from the exact GR growth ODE."""
    Om_a = Om0 * a_grid ** (-3)
    De_a = 1.0 - Om_a
    q_a = 1.0 - Om_a
    u_a = 1.0 + w0 + wa * (1.0 - a_grid)

    def rhs(x, y):
        a = x
        Om = Om0 * a ** (-3)
        De = 1.0 - Om
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a)
        H2 = Om + De * a ** (-3.0 * (1.0 + u))
        dH2 = -6.0 * Om + De * a ** (-3.0 * (1.0 + u)) * (-3.0 * (1.0 + u) + 3.0 * wa)
        # D'' = (3/(2a) - H')/H * D' - (3/(2a^2)) D
        return [y[1], (3.0 / (2.0 * a) - dH2 / (2.0 * H2)) / H2 * y[1] - (3.0 / (2.0 * a ** 2)) * y[0]]

    x = np.log(a_grid)
    y0 = [1.0, 0.5 * Om0]  # D(1)=1, D'(1)=f(1)=Om(1)^0.5 (Einstein-de Sitter limit)
    sol = solve_ivp(rhs, [x[0], x[-1]], y0, t_eval=x, method="RK45",
                    rtol=1e-9, atol=1e-11, dense_output=False)
    if not sol.success:
        return None, None
    D = sol.y[0]
    Dp = sol.y[1]
    H2 = Om_a + De_a * a_grid ** (-3.0 * (1.0 + u_a))
    dH2 = -6.0 * Om_a + De_a * a_grid ** (-3.0 * (1.0 + u_a)) * (-3.0 * (1.0 + u_a) + 3.0 * wa)
    f = Dp / (a_grid * D) - dH2 / (2.0 * H2)
    Dnorm = D / D[-1]
    return f, Dnorm

# Precompute reference on the grid
_REF = []
for _Om0 in _OM_GRID:
    for _w0 in _W0_GRID:
        for _wa in _WA_GRID:
            _f, _Dn = _true_f_D(_Om0, _w0, _wa, _A_GRID)
            if _f is not None and np.all(np.isfinite(_f)) and np.all(np.isfinite(_Dn)):
                _REF.append((_Om0, _w0, _wa, _f, _Dn))

# Build dense target arrays
if _REF:
    _N = len(_REF)
    _M = len(_A_GRID)
    _Om_a_all = np.empty((_N, _M))
    _q_all = np.empty((_N, _M))
    _u_all = np.empty((_N, _M))
    _wa_all = np.empty((_N, _M))
    _a_all = np.empty((_N, _M))
    _f_true = np.empty((_N, _M))
    _D_true = np.empty((_N, _M))
    for _i, (_Om0, _w0, _wa, _f, _Dn) in enumerate(_REF):
        _Om_a_all[_i] = _Om0 * _A_GRID ** (-3)
        _q_all[_i] = 1.0 - _Om_a_all[_i]
        _u_all[_i] = 1.0 + _w0 + _wa * (1.0 - _A_GRID)
        _wa_all[_i] = _wa
        _a_all[_i] = _A_GRID
        _f_true[_i] = _f
        _D_true[_i] = _Dn
    _log_f_true = np.log(np.clip(_f_true, 1e-12, None))
    _log_D_true = np.log(np.clip(_D_true, 1e-12, None))
else:
    _N = 0

# --- Surrogate objective: smooth max relative error (log-sum-exp) ---
def _objective(c):
    if _N == 0:
        return 1e6
    c0, c1, c2, c3, c4, c5 = c
    if any(abs(v) > 10.0 for v in c):
        return 1e6
    Om = _Om_a_all
    q = _q_all
    u = _u_all
    wa = _wa_all
    a = _a_all
    gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    # gamma constraint [0.1, 1.2]
    g_pen = np.maximum(0.0, 1.2 - gamma) + np.maximum(0.0, gamma - 0.1)
    g_pen = np.max(g_pen)
    if g_pen > 0:
        # heavy penalty if gamma out of range
        pass
    f = Om ** gamma
    # log D via integration of -f(exp(x)) dx from ln(a) to 0
    # log D(a) = - integral_{ln a}^{0} f(exp(x)) dx
    # We compute log D at each grid point via cumulative integration over x=ln(a)
    x = np.log(a)  # (N, M), increasing with a
    # f as function of x: f(exp(x)) = Om(x)^gamma(x); but we already have f on grid.
    # Integrate -f dx from x_i to 0 (i.e., from a_i to 1)
    # Use cumulative trapezoid from the end (a=1) backwards.
    neg_f = -f  # (N, M)
    # dx between consecutive x
    dx = np.diff(x, axis=1)  # (N, M-1)
    trap = 0.5 * (neg_f[:, :-1] + neg_f[:, 1:]) * dx  # integral over [x_i, x_{i+1}]
    # cumulative from the right: logD[i] = sum_{j>=i} trap[j]  (integral from x_i to x_{last})
    # since x_last = 0 (a=1), logD at a=1 = 0
    logD = np.zeros_like(f)
    # cumulative sum from right
    cs = np.cumsum(trap[:, ::-1], axis=1)[:, ::-1]  # (N, M-1)
    logD[:, :-1] = cs
    logD[:, -1] = 0.0

    # relative errors
    rel_f = np.abs(f - _f_true) / np.maximum(np.abs(_f_true), 1e-12)
    rel_D = np.abs(np.exp(logD) - _D_true) / np.maximum(np.abs(_D_true), 1e-12)
    # smooth max via log-sum-exp
    def smooth_max(arr):
        m = np.max(arr)
        if m < 1e-15:
            return 0.0
        s = np.sum(np.exp(arr - m))
        return m + math.log(s)
    obj = 0.5 * (smooth_max(rel_f) + smooth_max(rel_D))
    if g_pen > 0:
        obj += 100.0 * g_pen
    return obj

# --- Multi-start Nelder-Mead with deadline guard ---
def _run_search():
    best_c = list(_SEED)
    best_val = _objective(best_c)
    rng = np.random.default_rng(12345)
    n_starts = 40
    for k in range(n_starts):
        if time.time() > _DEADLINE:
            break
        if k == 0:
            x0 = list(_SEED)
        else:
            x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            # random perturbation, larger in later starts
            scale = 0.15 + 0.1 * (k / n_starts)
            x0 = [x0[i] + rng.normal(0.0, scale) for i in range(6)]
            x0 = [max(-10.0, min(10.0, v)) for v in x0]
        res = minimize(_objective, x0, method="Nelder-Mead",
                       options={"maxiter": 200, "maxfev": 300,
                                "xatol": 1e-6, "fatol": 1e-8})
        if time.time() > _DEADLINE:
            break
        if res.success or res.fun < best_val:
            cand = [max(-10.0, min(10.0, v)) for v in res.x]
            v = _objective(cand)
            if v < best_val:
                best_val = v
                best_c = cand
    return best_c, best_val

if _N > 0:
    _BEST_C, _BEST_V = _run_search()
else:
    _BEST_C, _BEST_V = list(_SEED), 1e6

# Clamp final coefficients to [-10,10] and ensure finite
_BEST_C = [float(v) for v in _BEST_C]
for i in range(6):
    if not math.isfinite(_BEST_C[i]):
        _BEST_C[i] = _SEED[i]
    _BEST_C[i] = max(-10.0, min(10.0, _BEST_C[i]))

def construct():
    return list(_BEST_C)
# EVOLVE-BLOCK-END
