"""Fit the six-coefficient growth-index ansatz to the GR growth ODE reference."""
import time
import numpy as np

# ---------------- reference model (mirrors the evaluator) ----------------
def _Om_m(a, Om0, Oml, w0, wa):
    a = np.asarray(a, dtype=float)
    a3 = a ** 3
    return Om0 / a3 + Oml / a3 * (1.0 + 3.0 * w0 + 2.0 * wa + wa * (1.0 - a))

def _gamma_of(a, c, Om, w0, wa):
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q

def _f_of(a, c, Om, w0, wa):
    return np.power(Om, _gamma_of(a, c, Om, w0, wa))

def _D_norm(a, c, Om, w0, wa):
    x = np.log(a)
    return np.exp(-np.trapz(_f_of(a, c, Om, w0, wa) * a, x))

def _ref_f(a, Om0, w0, wa):
    Oml = 1.0 - Om0
    Om = _Om_m(a, Om0, Oml, w0, wa)
    H2 = Om + Oml * (1.0 + 3.0 * w0 + 2.0 * wa + wa * (1.0 - a))
    return 0.5 * Om + 1.25 * Om * (Om - 1.0) / H2

def _ref_D(a, Om0, w0, wa):
    Oml = 1.0 - Om0
    x = np.log(a)
    Om = _Om_m(a, Om0, Oml, w0, wa)
    H2 = Om + Oml * (1.0 + 3.0 * w0 + 2.0 * wa + wa * (1.0 - a))
    integrand = 0.5 * Om + 1.25 * Om * (Om - 1.0) / H2
    return np.exp(-np.trapz(integrand * a, x))

# ---------------- objective over the validation set ----------------
class _Fit:
    def __init__(self):
        self.rng = np.random.default_rng(12345)
        # cosmology grid: a broad, representative CPL parameter space
        self.om0 = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.w0 = np.array([-2.0, -1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5])
        self.wa = np.array([-1.5, -1.0, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0])
        self.a = np.geomspace(0.02, 1.0, 400)

        self.models = []
        for o in self.om0:
            for w in self.w0:
                for wa in self.wa:
                    self.models.append((o, w, wa))
        self.models = np.array(self.models)
        self.n = len(self.models)

        self.a = self.a
        self.a3 = self.a ** 3
        self.loga = np.log(self.a)
        # precompute reference f and D for every model
        self.refF = np.empty((self.n, len(self.a)))
        self.refD = np.empty((self.n, len(self.a)))
        for i, (o, w, wa) in enumerate(self.models):
            self.refF[i] = _ref_f(self.a, o, w, wa)
            self.refD[i] = _ref_D(self.a, o, w, wa)

        self.seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.best = self.seed.copy()
        self.best_err = self._err(self.seed)

    def _gamma(self, c):
        Om = np.empty((self.n, len(self.a)))
        q = np.empty((self.n, len(self.a)))
        u = np.empty((self.n, len(self.a)))
        for i, (o, w, wa) in enumerate(self.models):
            Om[i] = o / self.a3 + (1.0 - o) / self.a3 * (
                1.0 + 3.0 * w + 2.0 * wa + wa * (1.0 - self.a))
            q[i] = 1.0 - Om[i]
            u[i] = 1.0 + w + wa * (1.0 - self.a)
        g = (c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * (wa * self.a)[:, None]
             + c[5] * u * q)
        return Om, g, q, u

    def _err(self, c):
        Om, g, q, u = self._gamma(c)
        # gamma bounds check
        if g.min() < 0.1 or g.max() > 1.2:
            return 1e6
        f = np.power(Om, g)
        if not np.all(np.isfinite(f)):
            return 1e6
        err_f = np.max(np.abs(f - self.refF) / self.refF)
        # normalized D via dense trapezoid (matches evaluator)
        D = np.exp(-np.trapz(f * self.a, self.loga, axis=1))
        if not np.all(np.isfinite(D)):
            return 1e6
        err_D = np.max(np.abs(D - self.refD) / self.refD)
        return max(err_f, err_D)

# ---------------- search with a wall-clock deadline ----------------
_DEADLINE = 84.0  # seconds; safety margin under the 90 s limit
_DEADLINE_T = None

def _time_left():
    if _DEADLINE_T is None:
        return _DEADLINE
    return _DEADLINE - (time.time() - _DEADLINE_T)

def _try_local(fit, c0):
    from scipy.optimize import minimize
    try:
        res = minimize(fit._err, c0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'maxfev': 4000, 'xatol': 1e-6,
                                'fatol': 1e-8, 'adaptive': True})
        c = np.clip(res.x, -10.0, 10.0)
        e = fit._err(c)
        if e < fit.best_err:
            fit.best_err = e
            fit.best = c.copy()
    except Exception:
        pass

def construct():
    global _DEADLINE_T
    _DEADLINE_T = time.time()
    seed = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        fit = _Fit()
        # quick coarse grid over a few promising directions
        if _time_left() > 5.0:
            base = fit.seed
            dirs = [
                np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.3, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
                np.array([0.0, 0.15, 0.0, 0.15, 0.0, 0.15]),
            ]
            for d in dirs:
                for s in (0.0, 0.3, 0.6, 1.0, 1.5):
                    if _time_left() < 2.0:
                        break
                    c = np.clip(base + s * d, -10.0, 10.0)
                    e = fit._err(c)
                    if e < fit.best_err:
                        fit.best_err = e
                        fit.best = c.copy()
            # local polish from the best coarse point and a few multi-starts
            if _time_left() > 4.0:
                _try_local(fit, fit.best)
            rng = fit.rng
            for _ in range(4):
                if _time_left() < 4.0:
                    break
                c0 = fit.seed + rng.normal(0.0, 0.25, 6)
                _try_local(fit, np.clip(c0, -10.0, 10.0))
            if _time_left() > 4.0:
                _try_local(fit, fit.best)
        out = np.clip(fit.best, -10.0, 10.0)
        if not np.all(np.isfinite(out)):
            return seed
        return [float(v) for v in out]
    except Exception:
        return seed
