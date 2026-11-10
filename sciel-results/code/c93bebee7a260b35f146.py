import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution
from math import factorial, pi
import time

_K = 10
_NTH = 70
_NPH = 130


def _build_basis():
    """Return (B, Wf): B is (Npts, 21) basis values, Wf flattened quadrature weights."""
    x, w = np.polynomial.legendre.leggauss(_NTH)          # x in [-1,1], sum w = 2
    phi = 2.0 * np.pi * np.arange(_NPH) / _NPH
    dphi = 2.0 * np.pi / _NPH
    X, PHI = np.meshgrid(x, phi, indexing='ij')           # (NTH, NPH)
    W = np.outer(w, np.full(_NPH, dphi))
    Wf = W.ravel()
    pts = X.size
    B = np.zeros((pts, 2 * _K + 1), dtype=np.float64)
    for idx, m in enumerate(range(-_K, _K + 1)):
        am = abs(m)
        N_m = np.sqrt(21.0 / (4.0 * pi) * factorial(_K - am) / factorial(_K + am))
        P = lpmv(am, _K, X)                               # includes Condon-Shortley phase
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * PHI)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(am * PHI)
        B[:, idx] = Y.ravel()
    return B, Wf


def _make_obj(B, Wf):
    def J(c):
        f = B @ c
        A = np.dot(Wf, f ** 4)
        B2 = np.dot(c, c)
        return A / (B2 * B2)
    def Jg(c):
        f = B @ c
        f3 = f ** 3
        A = np.dot(Wf, f ** 4)
        B2 = np.dot(c, c)
        Jv = A / (B2 * B2)
        dA = 4.0 * (B.T @ (Wf * f3))
        grad = 4.0 * (dA * B2 - A * c) / (B2 ** 3)
        return Jv, grad
    return J, Jg


def _polish(Jg, c0, maxiter=4000):
    try:
        res = minimize(Jg, c0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-12})
        if res.success or res.fun < Jg(c0)[0]:
            return res.x, res.fun
    except Exception:
        pass
    return c0, Jg(c0)[0]


def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    DEADLINE = 580.0  # safety margin vs 600s kill

    B, Wf = _build_basis()
    J, Jg = _make_obj(B, Wf)

    # seed
    c_seed = np.ones(2 * _K + 1, dtype=np.float64)
    best_c = c_seed.copy()
    best_J = J(c_seed)

    def _consider(c, Jv):
        nonlocal best_c, best_J
        if Jv < best_J:
            best_J = Jv
            best_c = c.copy()

    # 1) polish the seed
    c, Jv = _polish(Jg, c_seed)
    _consider(c, Jv)

    # 2) diverse structured / random restarts, each polished
    rng = np.random.default_rng(12345)
    starts = [c_seed.copy()]
    # alternating signs per |m|
    alt = np.array([(-1.0) ** ((i + _K) // 2) for i in range(2 * _K + 1)])
    starts.append(alt)
    # only m=0 and even m
    ev = np.array([1.0 if (i + _K) % 2 == 0 else 0.0 for i in range(2 * _K + 1)])
    starts.append(ev)
    for _ in range(12):
        starts.append(rng.normal(0.0, 1.0, 2 * _K + 1))

    for s in starts:
        if time.time() > DEADLINE - 60.0:
            break
        c, Jv = _polish(Jg, s)
        _consider(c, Jv)

    # 3) global Differential Evolution, then polish its best
    if time.time() < DEADLINE - 40.0:
        try:
            bounds = [(-5.0, 5.0)] * (2 * _K + 1)
            de = differential_evolution(J, bounds, seed=7, popsize=20,
                                        maxiter=400, tol=1e-12,
                                        init='latinhypercube', polish=False,
                                        updating='deferred', workers=1)
            if de.fun < best_J:
                c, Jv = _polish(Jg, de.x)
                _consider(c, Jv)
        except Exception:
            pass

    # 4) a couple more random-restart polishes if time remains
    n_extra = 0
    while time.time() < DEADLINE - 20.0 and n_extra < 10:
        s = rng.normal(0.0, 1.0, 2 * _K + 1)
        c, Jv = _polish(Jg, s)
        _consider(c, Jv)
        n_extra += 1

    # scale so max|c| = 1 (keeps validity 1e-3 <= max|c| <= 1e5)
    mx = np.max(np.abs(best_c))
    if mx > 0:
        best_c = best_c / mx
    return best_c.astype(np.float64)
