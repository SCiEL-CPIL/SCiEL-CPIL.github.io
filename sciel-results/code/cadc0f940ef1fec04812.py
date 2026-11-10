# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

_K = 10
_NTH = 70
_NPHI = 130
_NPTS = _NTH * _NPHI


def _build_quadrature():
    """Return (Y, w) where Y has shape (NPTS, 2K+1) and w has shape (NPTS,)
    such that sum_p w_p * (Y_p . c)^p  ==  integral over S^2 of |f|^p dOmega
    for p = 2, 4.  Uses 70-pt Gauss-Legendre in mu=cos(theta) x 130 eq phi."""
    mu, w_mu = np.polynomial.legendre.leggauss(_NTH)
    phi = np.arange(_NPHI, dtype=np.float64) * (2.0 * np.pi / _NPHI)
    mu2, phi2 = np.meshgrid(mu, phi, indexing="ij")
    phi_f = phi2.ravel()
    mu_f = mu2.ravel()
    w_f = np.repeat(w_mu, _NPHI) * (2.0 * np.pi / _NPHI)

    # associated Legendre with Condon-Shortley phase (scipy lpmv includes it)
    P = np.empty((_K + 1, _NPTS), dtype=np.float64)
    for m in range(_K + 1):
        P[m] = lpmv(m, _K, mu_f)

    # normalization N_m = sqrt(21/(4pi) * (K-m)!/(K+m)!)
    logfac = np.log(np.arange(1, _K + 2, dtype=np.float64)).cumsum()
    logfac = np.concatenate(([0.0], logfac))
    N = np.empty(_K + 1, dtype=np.float64)
    for m in range(_K + 1):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * np.exp(logfac[_K - m] - logfac[_K + m]))

    Y = np.empty((_NPTS, 2 * _K + 1), dtype=np.float64)
    s2 = np.sqrt(2.0)
    for m in range(_K + 1):
        Nm = N[m]
        Pm = P[m]
        if m == 0:
            Y[:, _K] = Nm * Pm
        else:
            Y[:, _K + m] = s2 * Nm * Pm * np.cos(m * phi_f)
            Y[:, _K - m] = s2 * Nm * Pm * np.sin(m * phi_f)
    return Y, w_f


_Y, _W = _build_quadrature()
# precompute the L2 Gram matrix: G_ij = sum_p w_p Y_pi Y_pj = delta_ij (orthonormal)
# and weight-squared for L4
_W2 = _W * _W
# Precompute Y^T Y weighted (should be identity). Also useful for gradient of L4.
_G = np.einsum("p,pi,pj->ij", _W, _Y, _Y)  # ~ identity
# For L4 gradient we need sum w_p (Y_p c)^3 Y_p = (Y^T diag((Yc)^3 w) Y) c
# Precompute per-point Y^T Y? That's 9100 x 21 x 21 -> too big. Instead compute
# gradient on the fly with a two-step tensordot.


def _ratios(c):
    """Return (R, L4, L2) for coefficient vector c (length 21)."""
    f = _Y @ c                       # (NPTS,)
    L2 = float(np.dot(_W, f * f))
    if L2 <= 0.0:
        return np.inf, 0.0, 0.0
    L4 = float(np.dot(_W2, f * f * f * f))
    R = np.sqrt(L4 / L2)
    return R, L4, L2


def _objective(c):
    """Smooth proxy for R = ||f||_4/||f||_2.  Minimize R; use R^2 (=L4/L2)
    which is smooth and monotone in R, plus a tiny L2 penalty to avoid
    the degenerate c=0 (where the ratio is undefined)."""
    f = _Y @ c
    L2 = float(np.dot(_W, f * f))
    L4 = float(np.dot(_W2, f * f * f * f))
    # R^2 = L4 / L2 ; add small ridge to keep away from 0
    return L4 / (L2 + 1e-12) + 1e-6 * L2


def _gradient(c):
    """Gradient of objective (R^2 + 1e-6 L2)."""
    f = _Y @ c
    L2 = float(np.dot(_W, f * f))
    # d(L4)/dc = 4 * sum w_p (Yc)^3 Y_p
    f3 = f * f * f
    L4grad = 4.0 * (_Y.T @ (_W2 * f3))
    # d(L2)/dc = 2 * sum w_p f Y_p
    L2grad = 2.0 * (_Y.T @ (_W * f))
    # d(L4/L2)/dc = (L4grad * L2 - L4 * L2grad) / L2^2
    L4 = float(np.dot(_W2, f * f * f * f))
    objgrad = (L4grad * L2 - L4 * L2grad) / (L2 * L2)
    objgrad += 1e-6 * L2grad
    return objgrad


def _run():
    t0 = time.time()
    deadline = t0 + 600.0
    best_c = np.ones(2 * _K + 1, dtype=np.float64)
    best_R = _ratios(best_c)[0]

    rng = np.random.default_rng(20240517)
    dim = 2 * _K + 1

    # ---- diverse initializations ----
    inits = []
    # all ones
    inits.append(np.ones(dim))
    # single modes
    for m in range(-_K, _K + 1):
        v = np.zeros(dim); v[m + _K] = 1.0; inits.append(v)
    # pairs of modes (cos/sin and real combos)
    for m1 in range(0, _K + 1):
        for m2 in range(m1, _K + 1):
            v = np.zeros(dim); v[_K + m1] = 1.0; v[_K + m2] = 1.0
            inits.append(v.copy())
            v2 = np.zeros(dim); v2[_K + m1] = 1.0; v2[_K - m2] = 1.0
            inits.append(v2.copy())
    # sparse random (2-4 nonzero)
    for _ in range(40):
        v = np.zeros(dim)
        idx = rng.choice(dim, size=rng.integers(2, 5), replace=False)
        v[idx] = rng.normal(size=len(idx))
        inits.append(v)
    # fully random
    for _ in range(30):
        inits.append(rng.normal(size=dim))
    # structured: alternating signs, symmetric, antisymmetric
    base = np.ones(dim)
    inits.append(base * np.cos(np.linspace(0, np.pi, dim)))
    inits.append(base * np.sin(np.linspace(0, np.pi, dim)))
    sym = np.zeros(dim); sym[:_K + 1] = 1.0; sym[_K + 1:] = 1.0
    inits.append(sym)
    asym = np.zeros(dim); asym[_K] = 0.0; asym[:_K] = 1.0; asym[_K + 1:] = -1.0
    inits.append(asym)

    # ---- score initializations, keep top for optimization ----
    scored = []
    for v in inits:
        v = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(v)) or np.max(np.abs(v)) < 1e-8:
            continue
        R = _ratios(v)[0]
        scored.append((R, v))
    scored.sort(key=lambda x: x[0])

    # ---- L-BFGS-B from top candidates + random restarts ----
    top = scored[:12]
    for R0, v0 in top:
        if time.time() > deadline - 25.0:
            break
        try:
            res = minimize(_objective, v0, jac=_gradient, method="L-BFGS-B",
                           options=dict(maxiter=300, ftol=1e-14, gtol=1e-10,
                                        maxfun=3000))
            if res.success or res.fun < _objective(v0):
                Rn = _ratios(res.x)[0]
                if Rn < best_R:
                    best_R = Rn
                    best_c = res.x.copy()
        except Exception:
            pass

    # random restarts
    n_restarts = 25
    for i in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        v0 = rng.normal(size=dim)
        try:
            res = minimize(_objective, v0, jac=_gradient, method="L-BFGS-B",
                           options=dict(maxiter=250, ftol=1e-14, gtol=1e-10,
                                        maxfun=2500))
            if res.success or res.fun < _objective(v0):
                Rn = _ratios(res.x)[0]
                if Rn < best_R:
                    best_R = Rn
                    best_c = res.x.copy()
        except Exception:
            pass

    # ---- high-precision polish pass from the best ----
    if time.time() < deadline - 20.0:
        try:
            res = minimize(_objective, best_c, jac=_gradient, method="L-BFGS-B",
                           options=dict(maxiter=600, ftol=1e-16, gtol=1e-12,
                                        maxfun=8000))
            if res.success or res.fun < _objective(best_c):
                Rn = _ratios(res.x)[0]
                if Rn < best_R:
                    best_R = Rn
                    best_c = res.x.copy()
        except Exception:
            pass

    # ---- validity guard: scale into (1e-3, 1e5) range ----
    c = np.asarray(best_c, dtype=np.float64)
    if not np.all(np.isfinite(c)):
        c = np.ones(dim)
    c = c / np.max(np.abs(c))
    c = c * 1.0
    if np.max(np.abs(c)) < 1e-3:
        c = c * (1e-2 / max(np.max(np.abs(c)), 1e-30))
    if np.max(np.abs(c)) > 1e5:
        c = c * (1e4 / np.max(np.abs(c)))
    return c


def rudin_problem10() -> np.ndarray:
    return _run()


# EVOLVE-BLOCK-END
