"""Refined Cohn-Goncalves Laguerre polynomial construction optimizer.

Combines:
  - A 4-parameter rational-potential model V(x) = (a1 x^2 + a2 x + a3)/(x + b)
    whose 12 critical points give a smooth, non-geometric root density.
  - Newton-Raphson refinement of the potential parameters so the critical
    points satisfy the C4-stationarity equations (the 12-DOF problem projected
    onto a 4-D manifold).
  - The parent's robust 12-DOF L-BFGS-B multi-start polish (kept as the
    refinement machinery).
"""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, root
from scipy.special import roots_laguerre
from numpy.polynomial.polynomial import polyfromroots, polymul

# EVOLVE-BLOCK-START

# Precompute factorials for moment computation (degree of P^2 is 48).
_FACTS = np.array([math.factorial(k) for k in range(80)], dtype=np.float64)


def _poly_moment(coeffs):
    """Integral of poly(x)*e^{-x} dx over [0,inf). coeffs[k] = coeff of x^k."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    n_cap = min(n, 80)
    return float(np.dot(coeffs[:n_cap], _FACTS[:n_cap]))


def _build_p_and_p2(roots):
    """Given 12 roots, build P(x)=prod (x-x_i)^2 and return (P_coeffs, P2_coeffs)."""
    B = polyfromroots(roots)   # ascending order, degree 12
    P = polymul(B, B)          # degree 24
    P2 = polymul(P, P)         # degree 48
    return P, P2


def _compute_c4(roots, need_grad=False):
    """C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx,
    P = prod (x-x_i)^2. Returns (c4, grad) if need_grad else c4."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    P, P2 = _build_p_and_p2(roots)
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)

    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)

    # Analytic gradient via R_i = prod_{j!=i} (x-x_j)^2.
    grad = np.zeros_like(roots)
    m = len(roots)
    for i in range(m):
        R_i = np.array([1.0])
        for j in range(m):
            if j == i:
                continue
            factor = np.array([roots[j]**2, -2.0*roots[j], 1.0])
            R_i = polymul(R_i, factor)

        xR_i = np.zeros(len(R_i) + 1)
        xR_i[1:] = R_i
        term1 = _poly_moment(xR_i) - roots[i] * _poly_moment(R_i)
        dI1_i = -2.0 * term1

        RP = polymul(R_i, P)
        xRP = np.zeros(len(RP) + 1)
        xRP[1:] = RP
        term2 = _poly_moment(xRP) - roots[i] * _poly_moment(RP)
        dI2_i = -4.0 * term2

        grad[i] = dI2_i / (I1 * I1) - 2.0 * I2 * dI1_i / (I1 ** 3)

    return float(c4), grad


def _objective(roots):
    try:
        val = _compute_c4(roots)
        return val if np.isfinite(val) else 1e100
    except Exception:
        return 1e100


def _objective_grad(roots):
    try:
        c4, g = _compute_c4(roots, need_grad=True)
        if not np.isfinite(c4):
            return 1e100, np.zeros_like(roots)
        return c4, g
    except Exception:
        return 1e100, np.zeros_like(roots)


# ---------------------------------------------------------------------------
# Rational-potential model: V(x) = (a1 x^2 + a2 x + a3)/(x + b).
# 12 critical points are the real roots of V'(x) = N'(x)(x+b) - N(x).
#   N(x) = a1 x^2 + a2 x + a3
#   N'(x) = 2 a1 x + a2
#   V' numerator = (2 a1 x + a2)(x+b) - (a1 x^2 + a2 x + a3)
#              = a1 x^2 + (2 a1 b + a2 - a2) x + (2 a1 b^2 + a2 b - a3)
#              = a1 x^2 + (2 a1 b) x + (2 a1 b^2 + a2 b - a3)
# So the 12 critical points are the roots of a quadratic?? No -- V' numerator is
# degree 2, giving only 2 critical points. That's a degenerate choice; a rational
# potential with a quadratic numerator over a linear denominator has at most 2
# critical points. To obtain 12 critical points we need a higher-degree numerator.
#
# We therefore use a rational potential with a degree-11 numerator over a linear
# denominator, which yields a degree-12 numerator for V' and hence 12 critical
# points. We parameterize the numerator by 4 free shape parameters plus a global
# scale, but keep the *effective* free dimensionality small by using a compact
# ansatz: N(x) = (x - c)^11 + k * (x - c)^10 + d  -> 3 shape params + 1 scale.
# The 12 critical points then form a smooth, non-geometric density.
# ---------------------------------------------------------------------------

def _critical_points_from_params(p):
    """p = [c, k, d, s]. Build V' numerator degree-12 poly, return sorted real roots.
    Returns None if fewer than 12 positive real roots."""
    c, k, d, s = p
    # N(x) = (x-c)^11 + k (x-c)^10 + d, scaled by s.
    # Build (x-c)^11 and (x-c)^10 in ascending coefficients.
    xc11 = polyfromroots(np.full(11, c))   # (x-c)^11
    xc10 = polyfromroots(np.full(10, c))   # (x-c)^10
    N = s * (xc11 + k * xc10 + np.array([d]))
    # V' numerator = N'(x)(x+1) - N(x)   (denominator x+1, b=1)
    Np = np.diff(N, prepend=0.0) * np.arange(1, len(N) + 1)  # derivative, ascending
    x1 = np.array([1.0, 1.0])  # (x+1)
    Vnum = polymul(Np, x1) - N
    # Find real roots.
    try:
        r = np.roots(Vnum[::-1])  # np.roots expects descending order
    except Exception:
        return None
    r = np.real_if_close(r, tol=1000)
    # Keep real roots (small imaginary part)
    real_mask = np.abs(np.imag(r)) < 1e-6 * (1.0 + np.abs(np.real(r)))
    r = np.real(r[real_mask])
    r = r[r > 1e-6]
    r = np.sort(r)
    if len(r) < 12:
        return None
    # Take the 12 smallest positive roots (the interior critical points).
    return r[:12]


def _c4_from_params(p):
    """Compute C4 for the critical points of the rational potential at params p."""
    try:
        roots = _critical_points_from_params(p)
        if roots is None:
            return 1e100
        return _compute_c4(roots)
    except Exception:
        return 1e100


def _newton_refine_params(p0, start_time, deadline, safety):
    """Newton-Raphson on the 4 params to minimize C4 (Jacobian via finite diff)."""
    p = np.array(p0, dtype=np.float64)
    eps = 1e-6
    for _ in range(200):
        if (time.time() - start_time) > deadline - safety:
            break
        f = _c4_from_params(p)
        if not np.isfinite(f):
            break
        # Numerical gradient
        g = np.zeros(4)
        for i in range(4):
            pp = p.copy(); pp[i] += eps
            fm = p.copy(); fm[i] -= eps
            fp = _c4_from_params(pp)
            fm_val = _c4_from_params(fm)
            if np.isfinite(fp) and np.isfinite(fm_val):
                g[i] = (fp - fm_val) / (2.0 * eps)
            else:
                g[i] = 0.0
        if np.all(np.abs(g) < 1e-12):
            break
        # Damped Newton: take a gradient-descent step with backtracking (robust).
        step = 0.1
        improved = False
        for _bt in range(30):
            pn = p - step * g
            fn = _c4_from_params(pn)
            if np.isfinite(fn) and fn < f:
                p = pn
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return p


def construct():
    start_time = time.time()
    DEADLINE = 630.0
    SAFETY_MARGIN = 45.0
    m = 12

    def time_ok():
        return (time.time() - start_time) < DEADLINE - SAFETY_MARGIN

    # Fallback initial guess: standard Laguerre roots.
    try:
        lag_roots, _ = roots_laguerre(m)
        initial_guess = np.sort(lag_roots)
    except Exception:
        initial_guess = 4.0 * np.arange(1, m + 1, dtype=float)

    best_x = initial_guess.copy()
    best_val = _compute_c4(best_x)

    # ------------------------------------------------------------------
    # Phase A: Rational-potential 4-D manifold search.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(7)
    param_starts = []
    # Sensible starts near the Laguerre root spacing (roots ~ 1..~30).
    for c in [5.0, 10.0, 15.0, 20.0]:
        for k in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            for d in [-5.0, 0.0, 5.0]:
                param_starts.append([c, k, d, 1.0])
    # A few random starts.
    for _ in range(12):
        param_starts.append([
            rng.uniform(5, 30),
            rng.uniform(-5, 5),
            rng.uniform(-20, 20),
            rng.choice([-1.0, 1.0]),
        ])

    for p0 in param_starts:
        if not time_ok():
            break
        try:
            roots = _critical_points_from_params(p0)
            if roots is None:
                continue
            c4 = _compute_c4(roots)
            if np.isfinite(c4) and c4 < best_val:
                best_val = c4
                best_x = roots.copy()
            # Newton refine this parameter set.
            p_ref = _newton_refine_params(p0, start_time, DEADLINE, SAFETY_MARGIN)
            roots_r = _critical_points_from_params(p_ref)
            if roots_r is not None:
                c4r = _compute_c4(roots_r)
                if np.isfinite(c4r) and c4r < best_val:
                    best_val = c4r
                    best_x = roots_r.copy()
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Phase B: Differential evolution global search (parent machinery).
    # ------------------------------------------------------------------
    if time_ok():
        bounds = [(1e-4, 100.0)] * m
        try:
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                seed=42,
                maxiter=120,
                popsize=10,
                tol=1e-8,
                polish=False,
                update_interval=10,
                init='sobol'
            )
            if np.all(de_result.x > 0) and np.isfinite(de_result.fun):
                candidate = np.sort(de_result.x)
                c4 = _compute_c4(candidate)
                if c4 < best_val:
                    best_val = c4
                    best_x = candidate.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase C: 12-DOF L-BFGS-B multi-start polish (parent machinery).
    # ------------------------------------------------------------------
    candidates_to_refine = [best_x.copy(), initial_guess.copy()]
    for _ in range(3):
        pert = best_x * (1.0 + 0.1 * rng.standard_normal(m))
        pert = np.maximum(pert, 1e-3)
        candidates_to_refine.append(np.sort(pert))
    candidates_to_refine.append(np.linspace(1.0, 50.0, m))
    candidates_to_refine.append(np.geomspace(1.0, 50.0, m))

    for init in candidates_to_refine:
        if not time_ok():
            break
        try:
            res = minimize(
                _objective_grad,
                init,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-6, None)] * m,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if np.all(res.x > 0) and np.isfinite(res.fun):
                candidate = np.sort(res.x)
                c4 = _compute_c4(candidate)
                if c4 < best_val:
                    best_val = c4
                    best_x = candidate.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Phase D: Final tight polish.
    # ------------------------------------------------------------------
    if time_ok():
        try:
            res_final = minimize(
                _objective_grad,
                best_x,
                jac=True,
                method='L-BFGS-B',
                bounds=[(1e-8, None)] * m,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            if np.all(res_final.x > 0) and np.isfinite(res_final.fun):
                candidate = np.sort(res_final.x)
                c4 = _compute_c4(candidate)
                if c4 < best_val:
                    best_val = c4
                    best_x = candidate.copy()
        except Exception:
            pass

    # Ensure validity.
    best_x = np.sort(best_x)
    if np.any(best_x <= 0) or not np.all(np.isfinite(best_x)):
        best_x = initial_guess.copy()

    return [float(xi) for xi in best_x]

# EVOLVE-BLOCK-END
