# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = time.time() + 170.0  # wall-clock guard; evaluator kills at 210s

# Hermite polynomials at 0 (physicists', exact)
_H00 = 1.0
_H40 = 12.0
_H80 = 1680.0
_H120 = -10395000.0


def _value(c0, c1, c2):
    """Reproduce the evaluator's value exactly (float arithmetic)."""
    c3 = -(_H00 * c0 + _H40 * c1 + _H80 * c2) / _H120
    # q = g / x^2 in z = x^2, degree 5:
    # q = c0 + c1*(z^2 + 2*6*z + 42) + c2*(z^4 + 8*14*z^3 + 8*35*z^2 + 28*15*z + 2520) + c3*(z^6 + ...)
    a6 = c3
    a5 = 12.0 * 11.0 * c3
    a4 = (8.0 * 10.0 * 9.0 * c3) + c2
    a3 = 8.0 * 8.0 * 7.0 * 10.0 * c3 + 8.0 * 14.0 * c2
    a2 = 8.0 * 6.0 * 5.0 * 4.0 * 3.0 * 2.0 * c3 + 8.0 * 35.0 * c2 + 2.0 * 6.0 * c1
    a1 = 8.0 * 4.0 * 3.0 * 2.0 * 1.0 * c3 + 28.0 * 15.0 * c2 + 2.0 * c1
    a0 = c0 + 42.0 * c1 + 2520.0 * c2
    if a6 < 0:  # g tends to negative limit -> replace by -g
        a6, a5, a4, a3, a2, a1, a0 = -a6, -a5, -a4, -a3, -a2, -a1, -a0
    roots = np.roots([a6, a5, a4, a3, a2, a1, a0])
    best = None
    for rr in roots:
        if abs(rr.imag) > 1e-9:
            continue
        x = rr.real
        if x <= 0:
            continue
        eps = 1e-8 * max(1.0, abs(x))
        if qsign(a6, a5, a4, a3, a2, a1, a0, x - eps) * qsign(a6, a5, a4, a3, a2, a1, a0, x + eps) < 0:
            if best is None or x > best:
                best = x
    if best is None:
        return float("inf")
    return best * best / (2.0 * math.pi)


def _qsign(a6, a5, a4, a3, a2, a1, a0, x):
    return ((a6 * x + a5) * x + a4) * x * x + (a3 * x * x + a2) * x + (a1 * x + a0)


def construct():
    global _DEADLINE
    t0 = time.time()
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best = seed.copy()
    best_val = _value(best[0], best[1], best[2])

    def obj(p):
        return _value(p[0], p[1], p[2])

    # ---- multi-start Nelder-Mead over the 2-DOF landscape ----
    rng = np.random.default_rng(12345)
    starts = [seed]
    # structured grid of starts
    for c0 in np.linspace(-1.5, 1.5, 13):
        for c1 in np.linspace(-1.0, 1.0, 9):
            for c2 in np.linspace(-0.05, 0.05, 5):
                starts.append(np.array([c0, c1, c2]))
    # random starts
    for _ in range(220):
        starts.append(np.array([rng.uniform(-2.0, 2.0), rng.uniform(-1.5, 1.5), rng.uniform(-0.1, 0.1)]))

    for s in starts:
        if time.time() > _DEADLINE - 40.0:
            break
        try:
            res = _nelder_mead(obj, s.copy(), maxiter=1200, tol=1e-10, t0=t0)
        except Exception:
            continue
        v = obj(res)
        if v < best_val:
            best_val = v
            best = res.copy()

    # ---- high-precision polish on the best point ----
    if time.time() < _DEADLINE - 10.0:
        for _ in range(3):
            try:
                res = _nelder_mead(obj, best.copy(), maxiter=4000, tol=1e-13, t0=t0)
            except Exception:
                break
            v = obj(res)
            if v < best_val:
                best_val = v
                best = res.copy()
            else:
                break

    return best


def _nelder_mead(f, x0, maxiter=2000, tol=1e-10, t0=None):
    n = len(x0)
    simplex = [x0.astype(float)]
    for i in range(n):
        step = x0.copy()
        step[i] = step[i] + (0.05 if step[i] == 0 else 0.05 * step[i])
        simplex.append(step)
    vals = [f(s) for s in simplex]
    for _ in range(maxiter):
        if t0 is not None and time.time() > _DEADLINE - 12.0:
            break
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * max(1.0, abs(vals[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = 0.5 * (simplex[i] + simplex[0])
                    vals[i] = f(simplex[i])
    order = np.argsort(vals)
    return simplex[order[0]]
# EVOLVE-BLOCK-END
