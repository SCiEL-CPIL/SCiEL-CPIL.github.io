"""Growth-index fit: compute true f and D via GR ODE, then fit the 6-coeff ansatz."""
import time
import numpy as np

_DEADLINE = time.time() + 85.0  # safety margin under 90 s wall-clock
_T = np.linspace(0.0, 1.0, 2001)
_LOGA = np.log(np.clip(_T, 1e-6, None))
_TAU = -_LOGA


def _Om_m(Om0, w0, wa, a):
    return Om0 * a**3 / (Om0 * a**3 + (1.0 - Om0) * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a)))


def _H2(Om0, w0, wa, a):
    return Om0 * a**3 + (1.0 - Om0) * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a))


def _ref_f(Om0, w0, wa):
    """True growth rate f(a)=Om(a)*H'/H via GR second-order ODE on D(a)."""
    N = len(_T)
    a = _T
    Om = _Om_m(Om0, w0, wa, a)
    h2 = _H2(Om0, w0, wa, a)
    a2 = a * a
    dOmda = -3.0 * Om / a
    h2p = 3.0 * h2 * (Om - 1.0) / a
    h2pp = (3.0 / a**2) * (3.0 * Om * (2.0 - Om) + (Om - 1.0) * (Om - 1.0))
    d2 = np.zeros(N)
    d1 = np.zeros(N)
    d1[N - 1] = 1.0
    d2[N - 1] = Om[N - 1]
    for i in range(N - 2, -1, -1):
        da = a[i + 1] - a[i]
        if da <= 0.0:
            d2[i] = d2[i + 1]
            d1[i] = d1[i + 1]
            continue
        # f = (3/2)(1+Om) - (2/3) h2'/(h2^2) - (1/2) h2''/(h2^2) - (1/2) Om' / h2
        f = (1.5 * (1.0 + Om[i])
             - (2.0 / 3.0) * h2p[i] / (h2[i] * h2[i])
             - 0.5 * h2pp[i] / (h2[i] * h2[i])
             - 0.5 * dOmda[i] / h2[i])
        g = 1.5 * (1.0 + Om[i]) + 0.5 * dOmda[i] / h2[i]
        d2[i] = (d2[i + 1] + da * f * d1[i + 1]) / (1.0 + da * g)
        d1[i] = d1[i + 1] + 0.5 * da * (d2[i] + d2[i + 1])
    f = 1.5 * (1.0 + Om) - 0.5 * dOmda / h2
    return f


def _ref_D(Om0, w0, wa, f):
    """Normalized growth factor D(a)/D(1) = exp(-int_a^1 f dln a)."""
    dlog = np.diff(_LOGA)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dlog)))
    return np.exp(-(cum[-1] - cum))


def _eval(c, Om0, w0, wa, f_ref, D_ref):
    a = _T
    q = 1.0 - Om0 * a**3 / (Om0 * a**3 + (1.0 - Om0) * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a)))
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    gam = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
    if np.any(gam < 0.1) or np.any(gam > 1.2):
        return np.inf
    f = (Om0 * a**3 / (Om0 * a**3 + (1.0 - Om0) * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a))))**gam
    dlog = np.diff(_LOGA)
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * dlog)))
    D = np.exp(-(cum[-1] - cum))
    ef = np.max(np.abs(f / f_ref - 1.0))
    eD = np.max(np.abs(D / D_ref - 1.0))
    return max(ef, eD)


def _fit():
    grid = [
        (0.30, -1.0, 0.0), (0.30, -1.0, 0.5), (0.30, -1.0, -0.5),
        (0.30, -0.9, 0.0), (0.30, -1.1, 0.0), (0.30, -1.0, 1.0),
        (0.30, -1.0, -1.0), (0.30, -0.5, 0.0), (0.30, -1.5, 0.0),
        (0.30, -1.0, 0.0),
    ]
    refs = []
    for Om0, w0, wa in grid:
        f = _ref_f(Om0, w0, wa)
        D = _ref_D(Om0, w0, wa, f)
        refs.append((Om0, w0, wa, f, D))

    def obj(c):
        if np.any(~np.isfinite(c)) or np.any(np.abs(c) > 10.0):
            return 1e6
        m = 0.0
        for Om0, w0, wa, f, D in refs:
            m = max(m, _eval(c, Om0, w0, wa, f, D))
        return m

    def nelder_mead(f, x0, step=0.15, maxiter=4000, tol=1e-9):
        n = len(x0)
        simplex = [np.array(x0, dtype=float)]
        f0 = f(simplex[0])
        for i in range(n):
            v = simplex[0].copy()
            v[i] += step
            simplex.append(v)
        for _ in range(maxiter):
            if time.time() > _DEADLINE:
                break
            vals = [f(s) for s in simplex]
            order = np.argsort(vals)
            simplex = [simplex[j] for j in order]
            vals = [vals[j] for j in order]
            if abs(vals[-1] - vals[0]) < tol:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = f(xr)
            if fr < vals[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                else:
                    simplex[-1] = xr
            elif fr < vals[-2]:
                simplex[-1] = xr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < vals[-1]:
                    simplex[-1] = xc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = 0.5 * (simplex[i] + simplex[0])
        best = min(simplex, key=lambda s: f(s))
        return best, f(best)

    starts = [
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.05, 0.0, 0.0, 0.0, 0.0],
        [0.6, -0.05, 0.0, 0.0, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.55, 0.0, 0.0, 0.0, 0.0, 0.05],
        [0.55, 0.0, 0.0, 0.0, 0.0, -0.05],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    best = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    bestv = obj(best)
    for s in starts:
        if time.time() > _DEADLINE:
            break
        x, v = nelder_mead(obj, s)
        if v < bestv:
            bestv = v
            best = x
    return np.clip(best, -10.0, 10.0)


_COEFFS = _fit()


def construct():
    return [float(x) for x in _COEFFS]
