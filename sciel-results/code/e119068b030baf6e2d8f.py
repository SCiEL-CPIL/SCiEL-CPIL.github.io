"""Analytic Hermite-Gaussian uncertainty bound.

Design (kept from the current program): the Fourier variance is the constant
vF = 7/3 (given the g(0)=0 constraint), so the bound is sqrt((7/3) * (m2/m0)).
Minimizing m2/m0 is a Rayleigh quotient on the 2-sphere of the 3 free
coefficients. We combine the current program's generalized-eigenvalue + L-BFGS-B
machinery with the inspiration's structural idea of solving the *stationary-point
equations* (gradient of the product of variances = 0) with a robust hybrid
root-finder to escape local minima, plus a Newton high-precision polish and an
exact-rational verification of the seed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from scipy.optimize import minimize, root
from scipy.linalg import eigh

_DEADLINE = time.time() + 195  # 15s safety margin before the 210s kill


def _bound_and_grad(a0, a4, a8):
    """Uncertainty product and analytic gradient w.r.t. (a0, a4, a8).

    a12 is fixed by g(0)=0: a12 = -(a0 + 3*a4 + 105*a8)/945.
    Because vF = n2/n0 = 7/3 is constant, bound = sqrt((7/3) * m2/m0) and the
    gradient only sees the m2/m0 (Rayleigh) part.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    sqrt_pi = np.sqrt(np.pi)

    m0_inner = a0 * a0 + 3.0 * a4 * a4 + 105.0 * a8 * a8 + 945.0 * a12 * a12
    m0 = sqrt_pi * m0_inner

    m2_inner = (a0 * (3.0 * a0 + 12.0 * a4)
                + a4 * (12.0 * a0 + 105.0 * a4 + 840.0 * a8)
                + a8 * (840.0 * a4 + 2940.0 * a8)
                + 11340.0 * a8 * a12)
    m2 = sqrt_pi * m2_inner

    if m0 <= 0.0 or not np.isfinite(m0) or not np.isfinite(m2):
        return 1e18, np.array([0.0, 0.0, 0.0])

    vf = m2 / m0
    vF = 7.0 / 3.0  # constant

    if vf <= 0.0:
        return 1e18, np.array([0.0, 0.0, 0.0])

    bound = np.sqrt(vf * vF)

    dm0_da0 = sqrt_pi * (2.0 * a0 + 2.0 * 945.0 * a12 * (-1.0 / 945.0))
    dm0_da4 = sqrt_pi * (6.0 * a4 + 2.0 * 945.0 * a12 * (-3.0 / 945.0))
    dm0_da8 = sqrt_pi * (210.0 * a8 + 2.0 * 945.0 * a12 * (-105.0 / 945.0))

    dm2_inner_da0 = 6.0 * a0 + 24.0 * a4
    dm2_inner_da4 = 24.0 * a0 + 210.0 * a4 + 1680.0 * a8
    dm2_inner_da8 = 1680.0 * a4 + 4620.0 * a8 + 11340.0 * a12

    dm2_da0 = sqrt_pi * dm2_inner_da0
    dm2_da4 = sqrt_pi * dm2_inner_da4
    dm2_da8 = sqrt_pi * dm2_inner_da8

    dvf_da0 = (dm2_da0 * m0 - m2 * dm0_da0) / (m0 * m0)
    dvf_da4 = (dm2_da4 * m0 - m2 * dm0_da4) / (m0 * m0)
    dvf_da8 = (dm2_da8 * m0 - m2 * dm0_da8) / (m0 * m0)

    grad = (0.5 / bound) * vF * np.array([dvf_da0, dvf_da4, dvf_da8])
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params[0], params[1], params[2])
    return val if np.isfinite(val) else 1e18


def _objective_grad(params):
    val, grad = _bound_and_grad(params[0], params[1], params[2])
    if not np.isfinite(val):
        return 1e18, np.array([0.0, 0.0, 0.0])
    return val, grad


def _stationary(a):
    """Gradient of the product of variances (unconstrained, scale-invariant)."""
    _, g = _bound_and_grad(a[0], a[1], a[2])
    return g


def _newton_polish(x0):
    """Exact Newton steps on the stationary-point equations (Hessian by FD)."""
    x = np.asarray(x0, dtype=float)
    for _ in range(60):
        if time.time() > _DEADLINE - 8:
            break
        g = _stationary(x)
        if not np.all(np.isfinite(g)):
            break
        if np.linalg.norm(g) < 1e-16:
            break
        hess = np.zeros((3, 3))
        eps = 1e-6
        for i in range(3):
            for j in range(3):
                if i == j:
                    pp = x.copy(); pm = x.copy()
                    pp[i] += eps; pm[i] -= eps
                    hess[i, j] = (_stationary(pp)[i] - _stationary(pm)[i]) / (2 * eps)
                else:
                    pp = x.copy(); pm = x.copy(); mp = x.copy(); mm = x.copy()
                    pp[i] += eps; pp[j] += eps
                    pm[i] += eps; pm[j] -= eps
                    mp[i] -= eps; mp[j] += eps
                    mm[i] -= eps; mm[j] -= eps
                    hess[i, j] = ((_stationary(pp)[i] - _stationary(pm)[i]
                                  - _stationary(mp)[i] + _stationary(mm)[i])
                                 / (4 * eps * eps))
        try:
            step = np.linalg.solve(hess + 1e-12 * np.eye(3), g)
        except Exception:
            break
        if not np.all(np.isfinite(step)):
            break
        # damped step
        t = 1.0
        for _ in range(30):
            x_new = x - t * step
            if np.all(np.isfinite(x_new)) and _objective(x_new) <= _objective(x) * (1 + 1e-12):
                break
            t *= 0.5
        x = x - t * step
        if np.linalg.norm(step) < 1e-14:
            break
    return x


def construct():
    """Return the three Hermite coefficients (degrees 0, 4, 8) minimizing the
    upper bound on C4. The degree-12 weight is fixed so g(0)=0."""
    # Symmetric matrices for the Rayleigh quotient m2/m0 = x^T B x / x^T A x.
    A = np.array([
        [1.0 + 1.0 / 945.0,       3.0 / 945.0,       105.0 / 945.0],
        [3.0 / 945.0,              3.0 + 9.0 / 945.0, 315.0 / 945.0],
        [105.0 / 945.0,            315.0 / 945.0,     105.0 + 11025.0 / 945.0]
    ])
    B = np.array([
        [3.0,     12.0,   -6.0],
        [12.0,    105.0,  822.0],
        [-6.0,    822.0,  1680.0]
    ])

    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240.0])
    best = seed.copy()
    best_val, _ = _bound_and_grad(*seed)

    # ---- 1. Generalized eigenvalue candidates (Rayleigh quotient critical pts) ----
    eigenvalue_solutions = []
    try:
        eigenvalues, eigenvectors = eigh(B, A)
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            g2_v = 2.0 * v[0] + 210.0 * v[2]
            if abs(g2_v) > 1e-12:
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-15:
                    v_scaled = v / norm_v
                    eigenvalue_solutions.append(v_scaled)
                    val_check, _ = _bound_and_grad(*v_scaled)
                    if np.isfinite(val_check) and val_check < best_val:
                        best_val = val_check
                        best = v_scaled.copy()
    except Exception:
        pass

    norm_seed = np.linalg.norm(seed)
    if norm_seed > 1e-15:
        seed_unit = seed / norm_seed
        val_seed_unit, _ = _bound_and_grad(*seed_unit)
        if np.isfinite(val_seed_unit) and val_seed_unit < best_val:
            best_val = val_seed_unit
            best = seed_unit.copy()

    # ---- 2. Build a diverse set of starting points (unit sphere) ----
    rng = np.random.default_rng(42)
    direct_starts = [best.copy()]
    for v in eigenvalue_solutions:
        direct_starts.append(v.copy())
    for i in range(60):
        rand_dir = rng.normal(0.0, 1.0, 3)
        norm_rd = np.linalg.norm(rand_dir)
        if norm_rd > 1e-15:
            rand_unit = rand_dir / norm_rd
            if abs(2.0 * rand_unit[0] + 210.0 * rand_unit[2]) > 1e-12:
                direct_starts.append(rand_unit)
    for i in range(40):
        scale = 0.4 / (i + 1)
        perturbed = best + rng.normal(0.0, scale, 3)
        norm_p = np.linalg.norm(perturbed)
        if norm_p > 1e-15:
            perturbed_unit = perturbed / norm_p
            if abs(2.0 * perturbed_unit[0] + 210.0 * perturbed_unit[2]) > 1e-12:
                direct_starts.append(perturbed_unit)

    # ---- 3. L-BFGS-B from each start (kept machinery) ----
    for s in direct_starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                           options={'maxiter': 4000, 'gtol': 1e-15, 'ftol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best = res.x.copy()
        except Exception:
            pass

    # ---- 4. Robust hybrid root-finder on stationary-point equations (new) ----
    # This directly targets exact critical points and escapes local-minima traps.
    for s in direct_starts:
        if time.time() > _DEADLINE - 25:
            break
        for method in ('hybr', 'lm'):
            try:
                sol = root(_stationary, s, method=method,
                           options={'xtol': 1e-14, 'maxfev': 20000})
                if sol.success and np.all(np.isfinite(sol.x)):
                    val, _ = _bound_and_grad(*sol.x)
                    if np.isfinite(val) and val < best_val:
                        best_val = val
                        best = sol.x.copy()
            except Exception:
                pass

    # ---- 5. Newton high-precision polish (tighter than L-BFGS-B) ----
    if time.time() < _DEADLINE - 12:
        try:
            polished = _newton_polish(best)
            if np.all(np.isfinite(polished)):
                val, _ = _bound_and_grad(*polished)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best = polished.copy()
        except Exception:
            pass

    # ---- 6. Exact-rational verification of the seed (structural sanity) ----
    try:
        a0 = Fraction(-113, 100)
        a4 = Fraction(1, 25)
        a8 = Fraction(1, 3240)
        a12 = -(a0 + 3 * a4 + 105 * a8) / 945
        m0_inner = a0 * a0 + 3 * a4 * a4 + 105 * a8 * a8 + 945 * a12 * a12
        m2_inner = (a0 * (3 * a0 + 12 * a4)
                    + a4 * (12 * a0 + 105 * a4 + 840 * a8)
                    + a8 * (840 * a4 + 2940 * a8)
                    + 11340 * a8 * a12)
        ratio = m2_inner / m0_inner
        # bound^2 = (7/3) * ratio
        bound_sq = Fraction(7, 3) * ratio
        if bound_sq < 0:
            bound_sq = abs(bound_sq)
        exact_bound = float(np.sqrt(float(bound_sq)))
        if np.isfinite(exact_bound) and exact_bound < best_val:
            best_val = exact_bound
            best = seed.copy()
    except Exception:
        pass

    # ---- Final verification ----
    if time.time() < _DEADLINE - 5:
        val_final, _ = _bound_and_grad(*best)
        if np.isfinite(val_final) and val_final < best_val:
            best_val = val_final

    return np.array(best)
# EVOLVE-BLOCK-END
