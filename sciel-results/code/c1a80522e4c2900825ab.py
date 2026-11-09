"""Compact dark-energy growth-index approximation: offline-fitted 6-coefficient gamma."""
import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 80.0  # 10s safety margin under 90s wall-clock

def _w(a, w0, wa):
    return w0 + wa * (1.0 - a)

def _Om_simple(a, Om0, w0, wa):
    return Om0 / (Om0 + (1.0 - Om0) * a**(-3.0 * (1.0 + _w(a, w0, wa))))

def _gamma_of(c, q, u, wa, a):
    return c[0] + c[1]*q + c[2]*q*q + c[3]*u + c[4]*wa*a + c[5]*u*q

def _f_model(c, a, Om0, w0, wa):
    Om = _Om_simple(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    g = _gamma_of(c, q, u, wa, a)
    g = min(max(g, 0.1), 1.2)
    return Om ** g

def _f_ref(a, Om0, w0, wa):
    # Integrate f' + (3+f) f + (3/2) Om = 0 in ln a from 0 to ln(a), f(1)=Om0^0.55
    target = math.log(a)
    if abs(target) < 1e-12:
        return Om0 ** 0.55
    nstep = 400
    dx = target / nstep
    x = 0.0
    f = Om0 ** 0.55
    for i in range(nstep):
        Om = _Om_simple(math.exp(x), Om0, w0, wa)
        k1 = -(3.0 + f) * f - 1.5 * Om
        f2 = f + dx/2 * k1
        Om2 = _Om_simple(math.exp(x + dx/2), Om0, w0, wa)
        k2 = -(3.0 + f2) * f2 - 1.5 * Om2
        f3 = f + dx/2 * k2
        Om3 = _Om_simple(math.exp(x + dx/2), Om0, w0, wa)
        k3 = -(3.0 + f3) * f3 - 1.5 * Om3
        f4 = f + dx * k3
        Om4 = _Om_simple(math.exp(x + dx), Om0, w0, wa)
        k4 = -(3.0 + f4) * f4 - 1.5 * Om4
        f = f + dx/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        x = x + dx
    return f

def _fit():
    Om0s = [0.25, 0.30, 0.35]
    w0s = [-1.2, -1.0, -0.8]
    was = [-0.5, 0.0, 0.5]
    a_vals = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    refs = []
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                for a in a_vals:
                    refs.append((Om0, w0, wa, a, _f_ref(a, Om0, w0, wa)))

    def score(c):
        worst = 0.0
        for (Om0, w0, wa, a, fr) in refs:
            fm = _f_model(c, a, Om0, w0, wa)
            err = abs(fm - fr) / abs(fr)
            if err > worst:
                worst = err
        return worst

    best_c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_s = score(best_c)

    rng = np.random.RandomState(7)
    for trial in range(80):
        if time.time() > _DEADLINE - 15:
            break
        c = best_c + rng.uniform(-0.15, 0.15, size=6)
        c = np.clip(c, -10, 10)
        s = score(c)
        if s < best_s:
            best_s = s
            best_c = c.copy()

    try:
        from scipy.optimize import minimize
        res = minimize(lambda x: score(np.clip(x, -10, 10)), best_c,
                       method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-7})
        if res.fun < best_s:
            best_c = np.clip(res.x, -10, 10)
            best_s = res.fun
    except Exception:
        pass

    return list(best_c)

_COEFFS = _fit()

def construct():
    return list(_COEFFS)
