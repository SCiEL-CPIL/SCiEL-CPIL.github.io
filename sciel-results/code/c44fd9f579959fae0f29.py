"""
Optimized uncertainty inequality solver using sphere reparameterization and
high-precision final polish.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195


def _hermite_coeffs():
    H = np.zeros((4, 7))
    H[0, 0] = 1.0
    H[1, 0] = 3.0; H[1, 1] = -6.0; H[1, 2] = 1.0
    H[2, 0] = 105.0; H[2, 1] = -420.0; H[2, 2] = 210.0; H[2, 3] = -28.0; H[2, 4] = 1.0
    H[3, 0] = 945.0; H[3, 1] = -7560.0; H[3, 2] = 13860.0; H[3, 3] = -6930.0
    H[3, 4] = 1155.0; H[3, 5] = -66.0; H[3, 6] = 1.0
    return H


def _compute_moments():
    I = np.zeros(13)
    sp = np.sqrt(np.pi)
    for n in range(13):
        if n == 0:
            I[n] = sp
        else:
            I[n] = I[n - 1] * (2.0 * n - 1.0) / (4.0 * n)
    return I


_H = _hermite_coeffs()
_I = _compute_moments()
_SIGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _bound(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H

    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i + j]
            m2 += pij * _I[1 + i + j]

    q = p * _SIGN

    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i + j]
            n2 += 2.0 * np.pi * qij * _I[1 + i + j]

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _bound_grad(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H

    m0 = 0.0
    m2 = 0.0
    for i in range(7):
        pi = p[i]
        if pi == 0.0:
            continue
        for j in range(7):
            pij = pi * p[j]
            m0 += pij * _I[i + j]
            m2 += pij * _I[1 + i + j]

    q = p * _SIGN

    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0:
            continue
        for j in range(7):
            qij = qi * q[j]
            n0 += 2.0 * np.pi * qij * _I[i + j]
            n2 += 2.0 * np.pi * qij * _I[1 + i + j]

    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])

    bound = np.sqrt(vf * vF)

    da12 = np.array([-1.0 / 945.0, -3.0 / 945.0, -105.0 / 945.0])

    S0 = np.zeros(7)
    S2 = np.zeros(7)
    T0 = np.zeros(7)
    T2 = np.zeros(7)
    for i in range(7):
        for j in range(7):
            S0[i] += p[j] * _I[i + j]
            S2[i] += p[j] * _I[1 + i + j]
            T0[i] += q[j] * _I[i + j]
            T2[i] += q[j] * _I[1 + i + j]

    grad = np.zeros(3)
    for c in range(3):
        dp = _H[c, :] + da12[c] * _H[3, :]

        dm0 = 2.0 * np.dot(dp, S0)
        dm2 = 2.0 * np.dot(dp, S2)

        dq = dp * _SIGN

        dn0 = 4.0 * np.pi * np.dot(dq, T0)
        dn2 = 4.0 * np.pi * np.dot(dq, T2)

        dvm = (dm2 * m0 - m2 * dm0) / (m0 * m0)
        dvn = (dn2 * n0 - n2 * dn0) / (n0 * n0)

        grad[c] = (vF * dvm + vf * dvn) / (2.0 * bound)

    return bound, grad


def _sphere_search(x0, n_starts=100, n_iter=20000):
    """
    Stochastic local search on sphere-reparameterized space.
    """
    best = x0.copy()
    best_f = _bound(x0)
    if not np.isfinite(best_f):
        return x0

    rng = np.random.default_rng(99999)
    base_scale = max(np.linalg.norm(x0), 0.5)

    for s in range(n_starts):
        if time.time() > _DEADLINE - 15:
            break

        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        x = x0 + base_scale * 0.3 * v

        f = _bound(x)
        if not np.isfinite(f) or f >= best_f:
            continue

        step = base_scale * 0.05
        for it in range(n_iter // n_starts):
            if time.time() > _DEADLINE - 10:
                break

            dv = rng.normal(size=3)
            dv /= np.linalg.norm(dv)
            x_try = x + step * dv

            f_try = _bound(x_try)
            if np.isfinite(f_try) and f_try < f:
                x = x_try
                f = f_try
                if f < best_f:
                    best_f = f
                    best = x.copy()
                step *= 1.02
            else:
                step *= 0.97

            if step < 1e-16:
                break

    return best


def _newton_polish(x0, max_iter=100, tol=1e-16):
    """Newton's method with numerical Hessian and line search."""
    x = x0.copy()
    f, g = _bound_grad(x)
    if not np.isfinite(f):
        return x
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        if np.max(np.abs(g)) < tol:
            break
        h = 1e-8
        H = np.zeros((3, 3))
        for i in range(3):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            _, gp = _bound_grad(xp)
            _, gm = _bound_grad(xm)
            H[i, :] = (gp - gm) / (2.0 * h)
        H += 1e-14 * np.eye(3)
        try:
            dx = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            dx = -0.01 * g
        step = 1.0
        for _ in range(40):
            x_try = x + step * dx
            f_try, g_try = _bound_grad(x_try)
            if np.isfinite(f_try) and f_try < f:
                x = x_try
                f = f_try
                g = g_try
                break
            step *= 0.5
        if step < 1e-14:
            break
    return x


def _bfgs_polish(x0, max_iter=50000, tol=1e-18):
    """BFGS polish with very tight tolerances."""
    res = minimize(_bound_grad, x0, method='BFGS',
                   jac=True, options={'maxiter': max_iter, 'gtol': tol, 'disp': False})
    if res.fun < _bound(x0) and np.isfinite(res.fun):
        return res.x
    return x0


def _nelder_polish(x0, max_iter=300000, tol=1e-22):
    """Nelder-Mead polish for final refinement."""
    res = minimize(_bound, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True})
    if res.fun < _bound(x0) and np.isfinite(res.fun):
        return res.x
    return x0


def _coord_descent_polish(x0, max_iter=100000, tol=1e-18):
    """
    Coordinate descent with adaptive step sizes and golden-section line search.
    """
    x = x0.copy()
    f = _bound(x)
    if not np.isfinite(f):
        return x

    for iteration in range(max_iter):
        if time.time() > _DEADLINE - 8:
            break

        improved = False
        for axis in range(3):
            scale = max(abs(x[axis]), 1e-8)
            step = scale * 0.1

            best_local = x.copy()
            best_local_f = f
            gr = (1.0 + np.sqrt(5.0)) / 2.0
            a = x[axis] - step
            b = x[axis] + step

            for _ in range(50):
                if time.time() > _DEADLINE - 5:
                    break
                m1 = b - (b - a) / gr
                m2 = a + (b - a) / gr
                x_m1 = x.copy(); x_m1[axis] = m1
                x_m2 = x.copy(); x_m2[axis] = m2
                f1 = _bound(x_m1)
                f2 = _bound(x_m2)
                if not np.isfinite(f1) or not np.isfinite(f2):
                    break
                if f1 < f2:
                    b = m2
                    if f1 < best_local_f:
                        best_local_f = f1
                        best_local = x_m1.copy()
                else:
                    a = m1
                    if f2 < best_local_f:
                        best_local_f = f2
                        best_local = x_m2.copy()

            if best_local_f < f - 1e-20:
                x = best_local
                f = best_local_f
                improved = True

        if not improved:
            step *= 0.1
            if step < 1e-22:
                break
            if time.time() > _DEADLINE - 5:
                break

    return x


def construct():
    """
    Multi-phase optimization:
    1. Multi-start L-BFGS-B with exact gradient
    2. Grid refinement
    3. Sphere-reparameterized stochastic search
    4. BFGS polish
    5. Nelder-Mead polish
    6. Coordinate descent polish
    7. Newton polish for highest precision
    """
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])

    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.03, 0.0002]),
        np.array([-0.8, 0.06, 0.0008]),
        np.array([-2.0, 0.02, 0.0001]),
        np.array([-1.2, 0.035, 0.0003]),
        np.array([-1.1, 0.045, 0.00035]),
        np.array([-0.5, 0.08, 0.001]),
        np.array([-3.0, 0.01, 0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -0.05, 0.001]),
        np.array([-2.5, 0.015, 0.00015]),
        np.array([-1.15, 0.042, 0.00032]),
        np.array([-1.05, 0.038, 0.00028]),
        np.array([-1.3, 0.035, 0.00035]),
        np.array([-0.9, 0.055, 0.0006]),
        np.array([-1.7, 0.025, 0.00025]),
        np.array([-0.7, 0.07, 0.0009]),
        np.array([-2.2, 0.018, 0.00018]),
        np.array([-1.4, 0.032, 0.00032]),
        np.array([-0.6, 0.085, 0.0012]),
        np.array([-2.8, 0.012, 0.00012]),
        np.array([-1.18, 0.041, 0.00031]),
        np.array([-1.08, 0.039, 0.00029]),
        np.array([-1.12, 0.038, 0.00029]),
        np.array([-1.14, 0.042, 0.00031]),
        np.array([-1.06, 0.045, 0.00036]),
        np.array([-1.22, 0.036, 0.00034]),
        np.array([-0.95, 0.052, 0.00052]),
    ]

    rng = np.random.default_rng(42)
    for _ in range(50):
        a0 = -np.exp(rng.uniform(0.4, 2.2))
        a4 = np.exp(rng.uniform(-3.5, 0.0)) * 0.12
        a8 = np.exp(rng.uniform(-5.5, -0.5)) * 0.0012
        starts.append(np.array([a0, a4, a8]))

    # Phase 1: Multi-start L-BFGS-B
    for start in starts:
        if time.time() > _DEADLINE - 40:
            break
        try:
            res = minimize(_bound_grad, start, method='L-BFGS-B',
                           jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Phase 2: Coarse grid refinement
    if time.time() < _DEADLINE - 45:
        center = best_params.copy()
        scales = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        for scale in scales:
            if time.time() > _DEADLINE - 20:
                break
            for axis in range(3):
                for direction in [-1, 1]:
                    if time.time() > _DEADLINE - 15:
                        break
                    trial = center.copy()
                    trial[axis] += direction * scale
                    try:
                        val = _bound(trial)
                        if val < best_val:
                            best_val = val
                            best_params = trial.copy()
                            res = minimize(_bound_grad, trial, method='L-BFGS-B',
                                           jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
                            if res.fun < best_val and np.isfinite(res.fun):
                                best_val = res.fun
                                best_params = res.x.copy()
                    except Exception:
                        pass

    # Phase 3: Sphere-reparameterized stochastic search
    if time.time() < _DEADLINE - 30:
        try:
            sphere_best = _sphere_search(best_params, n_starts=80, n_iter=25000)
            val = _bound(sphere_best)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_params = sphere_best.copy()
        except Exception:
            pass

    # Phase 4: BFGS polish
    if time.time() < _DEADLINE - 20:
        try:
            best_params = _bfgs_polish(best_params, max_iter=50000, tol=1e-18)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass

    # Phase 5: Nelder-Mead polish
    if time.time() < _DEADLINE - 12:
        try:
            best_params = _nelder_polish(best_params, max_iter=300000, tol=1e-22)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass

    # Phase 6: Coordinate descent polish
    if time.time() < _DEADLINE - 8:
        try:
            best_params = _coord_descent_polish(best_params, max_iter=100000, tol=1e-18)
            val = _bound(best_params)
            if val < best_val and np.isfinite(val):
                best_val = val
        except Exception:
            pass

    # Phase 7: Newton polish for highest precision
    if time.time() < _DEADLINE - 3:
        try:
            best_params = _newton_polish(best_params, max_iter=150, tol=1e-17)
            final_val = _bound(best_params)
            if final_val < best_val and np.isfinite(final_val):
                best_val = final_val
        except Exception:
            pass

    return np.array(best_params)
