"""Compact dark-energy growth-index fit: self-built fast GR reference + multi-start min-max fit."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Fixed internal deadline with a safety margin (evaluator kills at 90 s).
_DEADLINE = 75.0

# Cosmology grid: a in (0, 1] mapped to x = ln a in [-3, 0].
_A_GRID = np.exp(np.linspace(-3.0, 0.0, 240))
_X_GRID = np.log(_A_GRID)


def _omega_m(a, om0, w0, wa):
    a2 = a * a
    a3 = a2 * a
    return om0 / (om0 + (1.0 - om0) * a3 * np.exp(3.0 * (w0 + 1.0 + wa) * (1.0 - a) / (1.0 + wa)))


def _de_omega(a, om0, w0, wa):
    return 1.0 - _omega_m(a, om0, w0, wa)


def _gamma_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q)


def _gamma_feasible(c, om0, w0, wa):
    g = _gamma_of(c, _A_GRID, om0, w0, wa)
    return bool(np.all((g >= 0.1) & (g <= 1.2)))


def _f_of(c, a, om0, w0, wa):
    om = _omega_m(a, om0, w0, wa)
    return om ** _gamma_of(c, a, om0, w0, wa)


def _norm_d_of(c, om0, w0, wa):
    # D(a)/D(1) = exp(-integral_{ln a}^{0} f(exp x) dx)
    f = _f_of(c, _A_GRID, om0, w0, wa)
    # cumulative integral from x=0 (a=1) backward
    integ = np.zeros_like(_X_GRID)
    integ[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(_X_GRID))
    # D(a)/D(1): for a<1, integral from ln a to 0 = total - (integral up to ln a)
    # integral from ln a to 0 = sum over segments from idx to end
    total = integ[-1]
    # value at index i = total - integ[i]
    val = total - integ
    return np.exp(-val)


def _build_reference(om0, w0, wa):
    """Solve the second-order GR growth ODE for D(a) and f(a) = D'/(a D)."""
    N = 240
    x = np.linspace(-3.0, 0.0, N)
    a = np.exp(x)
    om = _omega_m(a, om0, w0, wa)
    ode = 1.0 - om
    # f(a) = D'/(aD); f(a) = (3/2) om + (3/2) om' - (3/2) om - ... use standard form:
    # f' + f^2 = (3/2) om + (3/2) om'  where derivative is d/da.
    # Work in x: d/dx = a d/da. Let g(x)=f. Then a f' = g - g^2.
    # a f' = (3/2) om + (3/2) a om' - (3/2) om? Standard: f' + f^2 = (3/2) om + (3/2) om' (d/d a).
    # a f' + f^2 = (3/2) om a + (3/2) a om'  -> not clean. Use d/dx form:
    # f' (d/da) = (1/a)(g - g^2). Equation: f' + f^2 = (3/2) om + (3/2) om'.
    # => (1/a)(g - g^2) + g^2 = (3/2) om + (3/2) om'
    # => g - g^2 + a g^2 = a*(3/2) om + a*(3/2) om'
    # => g*(1) + g^2*(a - 1) = (3/2) a (om + om')
    # Solve for g: (a-1) g^2 + g - (3/2) a (om+om') = 0.
    # Need om' = d om/d a.
    # om(a) = om0 / (om0 + (1-om0) a^3 E)  where E = exp(3 (w0+1+wa)(1-a)/(1+wa)).
    # Use numerical derivative for robustness.
    da = np.diff(a)
    om_prime = np.zeros_like(a)
    om_prime[1:] = (om[1:] - om[:-1]) / da
    om_prime[0] = om_prime[1]
    rhs = 1.5 * a * (om + om_prime)
    Acoef = (a - 1.0)
    # quadratic: Acoef g^2 + g - rhs = 0  => g = [-1 + sqrt(1 + 4 Acoef rhs)]/(2 Acoef)
    disc = 1.0 + 4.0 * Acoef * rhs
    disc = np.maximum(disc, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (-1.0 + np.sqrt(disc)) / (2.0 * Acoef)
    # fix a=1 limit: g(1) = 0.5 om(1) = 0.5 om0
    g[0] = 0.5 * om[0]
    # Now D(a)/D(1): d ln D / da = f => d ln D / dx = f * a = g
    lnD = np.zeros_like(x)
    lnD[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))
    D = np.exp(lnD)
    D = D / D[-1]
    return a, g, D


def _objective(c, models):
    # returns max relative error over f and normalized D across models and a
    worst = 0.0
    for (om0, w0, wa) in models:
        g = _gamma_of(c, _A_GRID, om0, w0, wa)
        if not (np.all(g >= 0.1) and np.all(g <= 1.2)):
            # feasibility penalty
            viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
            return 1.0 + float(np.max(viol))
        f = _f_of(c, _A_GRID, om0, w0, wa)
        D = _norm_d_of(c, om0, w0, wa)
        _, ref_f, ref_D = _build_reference(om0, w0, wa)
        # interpolate ref onto our grid (already same grid)
        er_f = np.max(np.abs((f - ref_f) / np.maximum(ref_f, 1e-12)))
        er_D = np.max(np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12)))
        worst = max(worst, er_f, er_D)
    return float(worst)


def _nelder_mead(c0, obj, max_iter=400, tol=1e-6):
    n = len(c0)
    # initial simplex
    simplex = [np.array(c0, dtype=float)]
    base = 0.05
    for i in range(n):
        v = np.array(c0, dtype=float)
        v[i] += base
        simplex.append(v)
    fvals = [obj(v) for v in simplex]
    best_i = int(np.argmin(fvals))
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = obj(xr)
        if fr < fvals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = obj(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = obj(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[i] + simplex[0])
                    fvals[i] = obj(simplex[i])
    best_i = int(np.argmin(fvals))
    return simplex[best_i], fvals[best_i]


def _fit(models, seed=12345):
    rng = np.random.default_rng(seed)
    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = _objective(best_c, models)
    starts = 0
    max_starts = 60
    while True:
        if time.time() > _DEADLINE - 8.0:
            break
        if starts >= max_starts:
            break
        starts += 1
        c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        c0 += rng.normal(0.0, 0.15, size=6)
        c0 = np.clip(c0, -10.0, 10.0)
        c, val = _nelder_mead(c0, lambda x: _objective(x, models), max_iter=300)
        c = np.clip(c, -10.0, 10.0)
        val = _objective(c, models)
        if val < best_val:
            best_val = val
            best_c = c
    return best_c, best_val


def construct():
    t0 = time.time()
    # Public validation-style cosmologies (disjoint, smooth CPL):
    models = [
        (0.30, -0.9, 0.0),
        (0.28, -1.0, 0.0),
        (0.31, -1.1, 0.3),
        (0.27, -0.8, -0.3),
        (0.30, -1.0, 0.5),
        (0.33, -0.95, 0.2),
    ]
    # Pre-compute references once to speed the objective.
    global _REF_CACHE
    try:
        _REF_CACHE
    except NameError:
        _REF_CACHE = {}
    for m in models:
        if m not in _REF_CACHE:
            _REF_CACHE[m] = _build_reference(*m)
    # Patch objective to use cache (redefine local wrapper).
    def obj(c):
        worst = 0.0
        for m in models:
            a, ref_f, ref_D = _REF_CACHE[m]
            om0, w0, wa = m
            g = _gamma_of(c, _A_GRID, om0, w0, wa)
            if not (np.all(g >= 0.1) and np.all(g <= 1.2)):
                viol = np.maximum(0.1 - g, 0.0) + np.maximum(g - 1.2, 0.0)
                return 1.0 + float(np.max(viol))
            f = _f_of(c, _A_GRID, om0, w0, wa)
            D = _norm_d_of(c, om0, w0, wa)
            er_f = np.max(np.abs((f - ref_f) / np.maximum(ref_f, 1e-12)))
            er_D = np.max(np.abs((D - ref_D) / np.maximum(np.abs(ref_D), 1e-12)))
            worst = max(worst, er_f, er_D)
        return float(worst)

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_val = obj(best_c)
    rng = np.random.default_rng(12345)
    starts = 0
    max_starts = 80
    while time.time() < _DEADLINE - 8.0 and starts < max_starts:
        starts += 1
        c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        c0 += rng.normal(0.0, 0.18, size=6)
        c0 = np.clip(c0, -10.0, 10.0)
        c, val = _nelder_mead(c0, obj, max_iter=350, tol=1e-7)
        c = np.clip(c, -10.0, 10.0)
        val = obj(c)
        if val < best_val:
            best_val = val
            best_c = c
    # Final safety: ensure finite and in range.
    if not np.all(np.isfinite(best_c)):
        best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = np.clip(best_c, -10.0, 10.0)
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END
