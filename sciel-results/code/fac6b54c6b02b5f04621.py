"""Dual-space Chebyshev projection fit to the precomputed GR reference."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_DEADLINE = time.time() + 78.0


def _om_of_a(Om0, w0, wa, a):
    a = np.asarray(a, dtype=float)
    x = a ** (3.0 * (1.0 + w0 + wa))
    return Om0 / (Om0 + (1.0 - Om0) * x)


def _gamma_vec(c, a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    q = 1.0 - _om_of_a(Om0, w0, wa, a)
    u = 1.0 + w0 + wa * (1.0 - a)
    return (c[0] + c[1] * q + c[2] * q * q + c[3] * u
            + c[4] * wa * a + c[5] * u * q)


def _f_ref(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    out = np.empty(a.size)
    for i in range(a.size):
        ai = float(a[i])
        if ai >= 0.9999999:
            out[i] = 1.0
            continue
        def ode(x, y):
            Om = float(_om_of_a(Om0, w0, wa, np.array([math.exp(x)]))[0])
            w = w0 + wa * (1.0 - math.exp(x))
            de = -3.0 * Om * (1.0 + w)
            return [y[1], 1.5 * Om - 0.5 * de - y[1] - y[1] * y[1]]
        sol = solve_ivp(ode, [math.log(ai), 0.0], [0.0, 1.0],
                        method="RK45", rtol=1e-9, atol=1e-11)
        out[i] = float(sol.y[1, -1])
    return out


def _build_grid(Om0s, w0s, was, n_a):
    """Precompute reference f and the design arrays once."""
    k = np.arange(n_a)
    a_base = np.exp(-1.0 * np.cos(np.pi * (k + 0.5) / n_a))
    a_base = np.clip(np.sort(a_base), 1e-3, 0.99999)
    combos = [(O, w0, wa) for O in Om0s for w0 in w0s for wa in was]
    n_combo = len(combos)
    F = np.empty(n_combo * n_a)
    A = np.empty(n_combo * n_a)
    idx = 0
    for (O, w0, wa) in combos:
        F[idx:idx + n_a] = _f_ref(a_base, O, w0, wa)
        A[idx:idx + n_a] = a_base
        idx += n_a
    w = 1.0 / (1.0 + 2.0 * np.abs(np.log(A)))
    return combos, A, F, w


def _predict(c, combos, A):
    """Return approx f and gamma on the flat array A (tiled by combo)."""
    n_combo = len(combos)
    n_a = A.size // n_combo
    a_base = A[:n_a]
    f = np.empty(A.size)
    g = np.empty(A.size)
    for ci in range(n_combo):
        O, w0, wa = combos[ci]
        sl = slice(ci * n_a, (ci + 1) * n_a)
        gc = _gamma_vec(c, a_base, O, w0, wa)
        gc = np.clip(gc, 0.1, 1.2)
        g[sl] = gc
        f[sl] = _om_of_a(O, w0, wa, a_base) ** gc
    return f, g


def _make_obj(combos, A, F, w, deriv_w):
    def obj(c):
        f, g = _predict(c, combos, A)
        err = (f - F) / F
        s = float(np.sum(w * err * err))
        if deriv_w > 0.0:
            h = 1e-5
            # derivative of relative error via finite difference on ln a
            f2, _ = _predict(c, combos, A * np.exp(h))
            # reference at perturbed a: reuse F only if a_base matches; here a differs
            # so we must have precomputed F2 as well -> handled by caller via closure
            # (we pass a precomputed F2 through the closure below)
            err2 = (f2 - F2) / F2
            derr = (err2 - err) / h
            s += deriv_w * float(np.sum(w * derr * derr))
        return s
    return obj


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 78.0

    Om0s = np.array([0.2, 0.3, 0.4, 0.5])
    w0s = np.array([-1.2, -1.0, -0.8, -0.5])
    was = np.array([-0.5, 0.0, 0.5])

    # ---- Stage 1: coarse, plain least squares ----
    combos, A, F, w = _build_grid(Om0s, w0s, was, n_a=10)

    def obj1(c):
        f, g = _predict(c, combos, A)
        err = (f - F) / F
        return float(np.sum(w * err * err))

    best = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_obj = obj1(best)
    for _ in range(3):
        if time.time() > _DEADLINE - 25.0:
            break
        try:
            res = minimize(obj1, best, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-5,
                                    "fatol": 1e-7, "adaptive": True})
            if res.fun < best_obj:
                best_obj = res.fun
                best = list(np.clip(res.x, -10.0, 10.0))
        except Exception:
            break

    # ---- Stage 2: fine grid with derivative (smoothness) penalty ----
    if time.time() < _DEADLINE - 20.0:
        combos2, A2, F2, w2 = _build_grid(Om0s, w0s, was, n_a=14)
        # precompute perturbed reference for derivative
        A2p = A2 * np.exp(1e-5)
        F2p = np.empty(A2p.size)
        n_combo = len(combos2)
        n_a2 = A2.size // n_combo
        a_base2 = A2[:n_a2]
        a_base2p = a_base2 * np.exp(1e-5)
        for ci in range(n_combo):
            O, w0, wa = combos2[ci]
            sl = slice(ci * n_a2, (ci + 1) * n_a2)
            F2p[sl] = _f_ref(a_base2p, O, w0, wa)

        def obj2(c):
            f, g = _predict(c, combos2, A2)
            err = (f - F2) / F2
            s = float(np.sum(w2 * err * err))
            f2, _ = _predict(c, combos2, A2p)
            err2 = (f2 - F2p) / F2p
            derr = (err2 - err) / 1e-5
            s += 0.05 * float(np.sum(w2 * derr * derr))
            return s

        best2 = list(best)
        best2_obj = obj2(best2)
        for _ in range(3):
            if time.time() > _DEADLINE - 5.0:
                break
            try:
                res = minimize(obj2, best2, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-6,
                                        "fatol": 1e-8, "adaptive": True})
                if res.fun < best2_obj:
                    best2_obj = res.fun
                    best2 = list(np.clip(res.x, -10.0, 10.0))
            except Exception:
                break
        best = best2

    # ---- Enforce gamma bounds over a wide sample ----
    best = [float(x) for x in best]
    a_s = np.linspace(0.01, 0.99999, 150)
    for _ in range(50):
        ok = True
        for O in Om0s:
            for w0 in w0s:
                for wa in was:
                    g = _gamma_vec(best, a_s, O, w0, wa)
                    if np.any(g < 0.102) or np.any(g > 1.198):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            break
        best = [best[0] + 0.5 * (best[i] - best[0]) if i > 0 else best[0]
                for i in range(6)]
    best = [max(-10.0, min(10.0, x)) for x in best]
    return best
# EVOLVE-BLOCK-END
