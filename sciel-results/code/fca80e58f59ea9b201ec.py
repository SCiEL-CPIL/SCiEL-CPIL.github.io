# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NPHI = 130
_NTH = 70


def _build_quadrature():
    x, w = np.polynomial.legendre.leggauss(_NTH)
    phi = np.linspace(0.0, 2.0 * np.pi, _NPHI, endpoint=False)
    dphi = 2.0 * np.pi / _NPHI
    W = np.outer(w, np.full(_NPHI, dphi))

    Y = np.zeros((_NTH, _NPHI, 2 * _K + 1))
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        denom = 1.0
        for n in range(_K - am + 1, _K + am + 1):
            denom *= n
        fact_ratio = 1.0 / denom
        Nm = np.sqrt((2 * _K + 1) / (4.0 * np.pi) * fact_ratio)
        P = lpmv(am, _K, x)
        if m == 0:
            Y[:, :, idx] = Nm * P[:, None]
        elif m > 0:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.cos(m * phi)[None, :]
        else:
            Y[:, :, idx] = np.sqrt(2.0) * Nm * P[:, None] * np.sin(am * phi)[None, :]
    return Y, W


_Y, _W = _build_quadrature()
_Wflat = _W.ravel()
_Yflat = _Y.reshape(-1, 2 * _K + 1)


def _ratio_from_coeffs(c):
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 0.0:
        return np.inf
    l4sq = float(np.dot(_Wflat, f2 * f2))
    if l4sq <= 0.0:
        return 1.0
    return np.sqrt(l4sq) / np.sqrt(l2sq)


def _negE(c):
    """Negative L2-normalized L4^4 (maximizing E = f^4 integral / (4pi)^2)."""
    f = _Yflat @ c
    f2 = f * f
    l2sq = float(np.dot(_Wflat, f2))
    if l2sq <= 1e-30:
        return 1e30
    f4 = f2 * f2
    l4sq = float(np.dot(_Wflat, f4))
    return -l4sq / (l2sq * l2sq)


def _negE_numgrad(c):
    c = np.asarray(c, dtype=np.float64)
    f0 = _negE(c)
    g = np.empty_like(c)
    for i in range(c.size):
        e = 1e-6 * max(1.0, abs(c[i]))
        cp = c.copy(); cp[i] += e
        cm = c.copy(); cm[i] -= e
        g[i] = (_negE(cp) - _negE(cm)) / (2.0 * e)
    return f0, g


def _lbfgs(c0, deadline):
    c = np.asarray(c0, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm

    def fun(x):
        return _negE(x)

    def jac(x):
        return _negE_numgrad(x)[1]

    try:
        res = minimize(fun, c, jac=jac, method="L-BFGS-B",
                       options={"maxiter": 250, "maxfun": 1500,
                                "ftol": 1e-15, "gtol": 1e-9})
    except Exception:
        return None
    c = np.asarray(res.x, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm
    return c, float(_negE(c))


def _nm(c0, deadline):
    c = np.asarray(c0, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm
    try:
        res = minimize(_negE, c, method="Nelder-Mead",
                       options={"maxiter": 1500, "xatol": 1e-10,
                                "fatol": 1e-12, "adaptive": True})
    except Exception:
        return None
    c = np.asarray(res.x, dtype=np.float64)
    nrm = np.linalg.norm(c)
    if nrm < 1e-14:
        return None
    c = c / nrm
    return c, float(_negE(c))


def _run(c0, deadline):
    out = _lbfgs(c0, deadline)
    if out is None:
        return None
    c, val = out
    if time.time() > deadline - 15.0:
        return c, val
    out2 = _nm(c, deadline)
    if out2 is not None:
        c2, val2 = out2
        if val2 < val:
            return c2, val2
    return c, val


def rudin_problem10() -> np.ndarray:
    start = time.time()
    deadline = start + 600.0
    rng = np.random.default_rng(2024)

    def _candidate(i):
        c = np.zeros(2 * _K + 1)
        for m in range(-_K, _K + 1):
            idx = m + _K
            c[idx] = np.cos(i * 0.7 + m * 0.3) + 0.3 * rng.standard_normal()
        return c

    best_c = None
    best_val = np.inf
    n_restarts = 120
    for i in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        c0 = _candidate(i)
        try:
            out = _run(c0, deadline)
        except Exception:
            continue
        if out is None:
            continue
        c, val = out
        if val < best_val:
            best_val = val
            best_c = c.copy()
        if time.time() > deadline - 30.0:
            break

    if best_c is None:
        best_c = np.ones(2 * _K + 1, dtype=np.float64)

    best_c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(2 * _K + 1, dtype=np.float64)
    mx = float(np.max(np.abs(best_c)))
    if mx < 1e-3:
        best_c = best_c * (1.0 / mx)
    if mx > 1e5:
        best_c = best_c * (1e5 / mx)
    return best_c


# EVOLVE-BLOCK-END
