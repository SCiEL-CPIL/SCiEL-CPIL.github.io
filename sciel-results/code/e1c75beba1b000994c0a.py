"""Growth-index fit for CPL dark energy; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

_DEADLINE = time.time() + 82.0  # safety margin under 90s wall-clock

def _H2(a, Om, w0, wa):
    u = 1.0 + w0 + wa * (1.0 - a)
    de = 1.0 - Om
    if de <= 0.0:
        return 1.0 + Om
    return Om * a ** (-3.0) + de * a ** (-3.0 * (1.0 + u))

def _Om_a(a, Om, w0, wa):
    h2 = _H2(a, Om, w0, wa)
    return Om * a ** (-3.0) / h2

def _solve_D(a_min, Om, w0, wa):
    # D(1)=1, D'(1)=f(1)=Om^gamma. Integrate backward.
    def rhs(x, y):
        a = math.exp(x)
        O = _Om_a(a, Om, w0, wa)
        g = 0.55  # only needed for initial slope
        f = O ** g
        dD = y[1]
        ddD = 2.0 * (1.0 - 0.5 * O) * dD - f * y[0]
        return [dD, ddD]
    f1 = Om ** 0.55
    sol = solve_ivp(rhs, [0.0, math.log(a_min)], [1.0, f1],
                    method="RK45", rtol=1e-10, atol=1e-13, max_step=0.05)
    if not sol.success:
        return None
    return sol

def _ref_model(a_grid, Om, w0, wa):
    a_min = a_grid[0]
    sol = _solve_D(a_min, Om, w0, wa)
    if sol is None:
        return None
    xs = np.log(a_grid)
    D = sol.sol(xs)
    Dn = D[0] / D[0][-1]
    f = np.array([_Om_a(a, Om, w0, wa) ** 0.55 for a in a_grid])
    return f, Dn

def _cosmos():
    out = []
    for Om in [0.22, 0.25, 0.27, 0.30, 0.33, 0.36]:
        for w0 in [-1.2, -1.1, -1.05, -1.0, -0.95, -0.9]:
            for wa in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                out.append((Om, w0, wa))
    return out

def _a_grid():
    return np.linspace(0.05, 1.0, 40)

def _eval(c, cosmos, a_grid):
    c0, c1, c2, c3, c4, c5 = c
    worst = 0.0
    for (Om, w0, wa) in cosmos:
        ref = _ref_model(a_grid, Om, w0, wa)
        if ref is None:
            continue
        f_ref, Dn_ref = ref
        for i, a in enumerate(a_grid):
            q = 1.0 - _Om_a(a, Om, w0, wa)
            u = 1.0 + w0 + wa * (1.0 - a)
            g = c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q
            f = _Om_a(a, Om, w0, wa) ** g
            err_f = abs(f - f_ref[i]) / abs(f_ref[i])
            if err_f > worst:
                worst = err_f
            err_D = abs(math.exp(np.log(Dn_ref[i]) - np.log(Dn_ref[-1])) - 1.0) / abs(1.0)
            if err_D > worst:
                worst = err_D
    return worst

def _objective(c, cosmos, a_grid):
    if time.time() > _DEADLINE:
        return 1e6
    # gamma bound penalty
    bad = 0.0
    for (Om, w0, wa) in cosmos:
        for a in a_grid:
            q = 1.0 - _Om_a(a, Om, w0, wa)
            u = 1.0 + w0 + wa * (1.0 - a)
            g = c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q
            if g < 0.1:
                bad += (0.1 - g)
            elif g > 1.2:
                bad += (g - 1.2)
    base = _eval(c, cosmos, a_grid)
    if bad > 0:
        return base + 10.0 * bad
    return base

# EVOLVE-BLOCK-END

def construct():
    global _DEADLINE
    try:
        cosmos = _cosmos()
        a_grid = _a_grid()
        bounds = [(-1.0, 1.5), (-1.0, 1.0), (-1.0, 1.0),
                  (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Coarse global search if time permits
        if time.time() < _DEADLINE - 40.0:
            try:
                res = differential_evolution(
                    _objective, bounds, args=(cosmos, a_grid),
                    seed=1, maxiter=6, popsize=8, tol=1e-3,
                    polish=False, init="sobol",
                    updating="deferred")
                if res.x is not None and time.time() < _DEADLINE - 20.0:
                    x0 = list(res.x)
            except Exception:
                pass

        # Local polish with tighter tolerance
        best = x0
        best_val = _objective(x0, cosmos, a_grid)
        for _ in range(4):
            if time.time() > _DEADLINE - 10.0:
                break
            try:
                res = minimize(_objective, best, args=(cosmos, a_grid),
                               method="Nelder-Mead",
                               options={"maxiter": 300, "xatol": 1e-7,
                                        "fatol": 1e-8, "adaptive": True})
                if res.x is not None and res.fun < best_val:
                    best = list(res.x)
                    best_val = res.fun
            except Exception:
                break

        # Clamp to [-10,10]
        best = [min(10.0, max(-10.0, float(v))) for v in best]
        return best
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
# EVOLVE-BLOCK-END
