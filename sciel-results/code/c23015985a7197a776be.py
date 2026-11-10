# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize

# ----------------------------- constants -----------------------------
_K = 10
_NTH = 70          # Gauss-Legendre nodes in cos(theta)
_NPHI = 130        # equally spaced phi samples
_N = _NTH * _NPHI  # total quadrature nodes (9100)

_DEADLINE = time.time() + 600.0
_SAFE = 5.0        # seconds of margin before the hard limit

# ----------------------------- quadrature -----------------------------
# Gauss-Legendre nodes/weights on [-1,1]
_x, _w = np.polynomial.legendre.leggauss(_NTH)
# phi grid
_phi = 2.0 * np.pi * np.arange(_NPHI) / _NPHI
# flatten node arrays: rows = nodes, cols = (theta, phi)
_COS = np.repeat(_x, _NPHI)                 # (N,)
_PHI = np.tile(_phi, _NTH)                  # (N,)
_W = np.repeat(_w, _NPHI) * (2.0 * np.pi / _NPHI)   # (N,) surface weights
_WN = _W / (4.0 * np.pi)                     # normalized (sum = 1)


def _build_matrix():
    """M[m, i] = Y_m(x_i) for real orthonormal degree-10 harmonics."""
    M = np.empty((2 * _K + 1, _N), dtype=np.float64)
    for m in range(0, _K + 1):
        # normalization N_m = sqrt(21/(4pi) * (10-m)!/(10+m)!)
        logN = 0.5 * (np.log(21.0 / (4.0 * np.pi))
                      + np.sum(np.log(np.arange(1, _K - m + 1)))
                      - np.sum(np.log(np.arange(1, m + 1))))
        N = np.exp(logN)
        # P_10^m(cos theta) with Condon-Shortley phase
        P = lpmv(m, _K, _COS)
        if m == 0:
            M[m] = N * P
        else:
            s = np.sqrt(2.0) * N * P
            M[_K + m] = s * np.cos(m * _PHI)   # m > 0
            M[_K - m] = s * np.sin(m * _PHI)   # m < 0
    return M


_M = _build_matrix()


# ----------------------------- objective -----------------------------
def _objective(c):
    """Return (R(c), grad R(c)).  R = ||f||_4 / ||f||_2."""
    f = _M @ c
    f2 = f * f
    f4 = f2 * f2
    mean2 = float(np.dot(_WN, f2))
    mean4 = float(np.dot(_WN, f4))
    if mean2 <= 1e-300:
        return 1e300, np.zeros_like(c)
    R = (mean4 ** 0.25) / np.sqrt(mean2)
    # dR/dc = (1/(4 R mean2)) * ( f^3 - R^2 f )
    #        * ( 4*mean4 - mean2 )
    g = (f2 * f - R * R * f) * (4.0 * mean4 - mean2)
    scale = (1.0 / (4.0 * R * mean2))
    grad = scale * (_M.T @ g)
    return R, grad


# ----------------------------- validity -----------------------------
def _valid(c):
    if c is None:
        return False
    c = np.asarray(c)
    if c.shape != (2 * _K + 1,):
        return False
    if not np.all(np.isfinite(c)):
        return False
    mx = float(np.max(np.abs(c)))
    return (mx >= 1e-3) and (mx <= 1e5)


# ----------------------------- initializations -----------------------------
def _init_pool():
    rng = np.random.default_rng(20240517)
    K = _K
    pool = []

    # all-ones (seed baseline)
    pool.append(np.ones(2 * K + 1))

    # single modes
    for m in range(-K, K + 1):
        v = np.zeros(2 * K + 1)
        v[m + K] = 1.0
        pool.append(v)

    # pairs
    for a in range(-K, K + 1):
        for b in range(a + 1, K + 1):
            v = np.zeros(2 * K + 1)
            v[a + K] = 1.0
            v[b + K] = 1.0
            pool.append(v)

    # even m only (symmetry-friendly)
    v = np.zeros(2 * K + 1)
    for m in range(-K, K + 1, 2):
        v[m + K] = 1.0
    pool.append(v)
    # odd m only
    v = np.zeros(2 * K + 1)
    for m in range(-K, K + 1, 2):
        if m % 2 != 0:
            v[m + K] = 1.0
    pool.append(v)

    # cos-only (m>0 and m=0)
    v = np.zeros(2 * K + 1)
    for m in range(0, K + 1):
        v[m + K] = 1.0
    pool.append(v)
    # sin-only (m<0)
    v = np.zeros(2 * K + 1)
    for m in range(-K, 0):
        v[m + K] = 1.0
    pool.append(v)

    # |m| weighting: c_{+m} = c_{-m} = w_m
    for pattern in (lambda m: 1.0,
                    lambda m: (-1) ** m,
                    lambda m: 1.0 / (1.0 + m)):
        v = np.zeros(2 * K + 1)
        for m in range(0, K + 1):
            w = pattern(m)
            v[m + K] = w
            v[-m + K] = w if m != 0 else 0.0
        pool.append(v)

    # random unit-norm
    for _ in range(14):
        v = rng.standard_normal(2 * K + 1)
        v /= (np.linalg.norm(v) + 1e-300)
        pool.append(v)

    return pool


# ----------------------------- main -----------------------------
def rudin_problem10() -> np.ndarray:
    K = _K
    n = 2 * K + 1

    best = None
    bestR = np.inf

    def consider(c):
        nonlocal best, bestR
        if not _valid(c):
            return
        val, _ = _objective(c)
        if val < bestR:
            bestR = val
            best = np.asarray(c, dtype=np.float64).copy()

    pool = _init_pool()
    rng = np.random.default_rng(20240517)

    # Phase 1: evaluate pool, keep top candidates
    scored = []
    for c in pool:
        if time.time() > _DEADLINE - _SAFE:
            break
        if not _valid(c):
            continue
        val, _ = _objective(c)
        scored.append((val, np.asarray(c, dtype=np.float64).copy()))
    scored.sort(key=lambda t: t[0])

    # Phase 2: L-BFGS-B from top candidates + random restarts
    top = scored[:10]
    n_random = 15
    for it in range(len(top) + n_random):
        if time.time() > _DEADLINE - _SAFE:
            break
        if it < len(top):
            x0 = top[it][1]
        else:
            x0 = rng.standard_normal(n)
            x0 /= (np.linalg.norm(x0) + 1e-300)
        try:
            res = minimize(_objective, x0, method="L-BFGS-B",
                           jac=True,
                           options={"maxiter": 400,
                                    "maxfun": 4000,
                                    "gtol": 1e-10})
        except Exception:
            continue
        if res is not None and hasattr(res, "x") and res.x is not None:
            consider(res.x)

    # Phase 3: if spare time, a few extra restarts from around best
    if best is not None:
        for _ in range(30):
            if time.time() > _DEADLINE - _SAFE:
                break
            x0 = best + 0.2 * rng.standard_normal(n)
            try:
                res = minimize(_objective, x0, method="L-BFGS-B",
                               jac=True,
                               options={"maxiter": 250, "maxfun": 2500,
                                        "gtol": 1e-10})
            except Exception:
                continue
            if res is not None and hasattr(res, "x") and res.x is not None:
                consider(res.x)

    # fallback
    if best is None:
        best = np.ones(n, dtype=np.float64)

    return best


# EVOLVE-BLOCK-END
