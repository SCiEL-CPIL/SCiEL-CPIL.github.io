"""Optimized tuner for the Fourier-invariant Hermite-Gaussian uncertainty bound."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from scipy.special import factorial
from scipy.optimize import minimize

_T0 = time.time()
_HARD_DEADLINE = 200.0  # safety margin before the 210 s kill
_SEED = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])


def _h(x, n):
    """Physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _coeffs(a, b, c):
    """Full coefficient vector [H0, H4, H8, H12] with g(0)=0 enforced."""
    d = -(a + b / 6.0 + c / 108.0)
    return np.array([a, b, c, d])


def _build_basis(n, x):
    """Cache the Hermite polynomials and Gaussian on a grid."""
    hs = [_h(x, k) for k in (0, 4, 8, 12)]
    g = np.exp(-0.5 * x * x)
    return hs, g


def _build_fourier_coeffs(n, x):
    """Fourier transform of the basis, in the same Hermite-Gaussian basis.

    F[g](x) = sqrt(2*pi) * sum_i c_i * H_{n_i}(x) * exp(-x^2/2).
    """
    scale = np.sqrt(2.0 * np.pi)
    c = _coeffs(n[0], n[1], n[2])
    degs = (0, 4, 8, 12)
    fcoeffs = [scale * c[i] for i in range(4)]
    return fcoeffs, degs


def _value(n, x, hs, g, fcoeffs, degs, fhs):
    """Upper bound on C4 for a given coefficient triple."""
    gval = g * (n[0] * hs[0] + n[1] * hs[1] + n[2] * hs[2] +
                (-(n[0] + n[1] / 6.0 + n[2] / 108.0)) * hs[3])
    g2 = gval * gval
    A = np.sqrt(np.sum(g2) / np.sum(g2[0]))  # placeholder; recomputed below
    # A(f) = sqrt( sum g^2 / sum g^2 ) -- but g is normalized? Use L2 norm.
    A = np.sqrt(np.sum(g2))
    fval = g * sum(fcoeffs[i] * fhs[i] for i in range(4))
    f2 = fval * fval
    AF = np.sqrt(np.sum(f2))
    return A * AF


def _objective(n, x, hs, g, fcoeffs_cache):
    fval = g * sum(fcoeffs_cache[i] * _h(x, (0, 4, 8, 12)[i]) for i in range(4))
    gval = g * (n[0] * hs[0] + n[1] * hs[1] + n[2] * hs[2] +
                (-(n[0] + n[1] / 6.0 + n[2] / 108.0)) * hs[3])
    A = np.sqrt(np.sum(gval * gval))
    AF = np.sqrt(np.sum(fval * fval))
    return A * AF


def _softmax_upper(n, x, hs, g, fcoeffs, degs, fhs, scale):
    """Smooth surrogate: log-sum-exp over grid of per-point upper bounds."""
    gval = g * (n[0] * hs[0] + n[1] * hs[1] + n[2] * hs[2] +
                (-(n[0] + n[1] / 6.0 + n[2] / 108.0)) * hs[3])
    fval = g * sum(fcoeffs[i] * fhs[i] for i in range(4))
    A2 = np.sum(gval * gval)
    AF2 = np.sum(fval * fval)
    return np.sqrt(A2 * AF2)


def construct():
    best_n = _SEED.copy()
    best_val = _objective(best_n, None, None, None, None) if False else None

    # Build a fine frequency grid for the upper bound.
    x = np.linspace(-6.0, 6.0, 1201)
    hs = [_h(x, k) for k in (0, 4, 8, 12)]
    g = np.exp(-0.5 * x * x)
    fhs = hs  # same basis for the Fourier side

    def eval_n(n):
        fcoeffs = [np.sqrt(2.0 * np.pi) * v for v in _coeffs(n[0], n[1], n[2])]
        gval = g * (n[0] * hs[0] + n[1] * hs[1] + n[2] * hs[2] +
                    (-(n[0] + n[1] / 6.0 + n[2] / 108.0)) * hs[3])
        fval = g * (fcoeffs[0] * fhs[0] + fcoeffs[1] * fhs[1] +
                    fcoeffs[2] * fhs[2] + fcoeffs[3] * fhs[3])
        A = np.sqrt(np.sum(gval * gval))
        AF = np.sqrt(np.sum(fval * fval))
        return A * AF

    best_val = eval_n(best_n)

    def objective(n):
        return eval_n(n)

    # Multi-start local optimization with a smooth objective.
    rng = np.random.default_rng(12345)
    starts = [best_n.copy()]
    # Deterministic box starts around the seed.
    for delta in (0.2, 0.5, 1.0, 2.0):
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sz in (-1, 0, 1):
                    if sx == sy == sz == 0:
                        continue
                    cand = best_n.copy()
                    cand[0] += sx * delta
                    cand[1] += sy * delta * 0.1
                    cand[2] += sz * delta * 0.01
                    starts.append(cand)
    # Random starts.
    for _ in range(20):
        starts.append(np.array([
            best_n[0] + rng.uniform(-3, 3),
            best_n[1] + rng.uniform(-0.5, 0.5),
            best_n[2] + rng.uniform(-0.1, 0.1),
        ]))

    for s in starts:
        if time.time() > _HARD_DEADLINE:
            break
        try:
            res = minimize(objective, s, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-8,
                                    'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            pass

    # High-precision polish pass with tighter tolerances.
    for _ in range(3):
        if time.time() > _HARD_DEADLINE:
            break
        try:
            res = minimize(objective, best_n, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-11,
                                    'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_n = res.x.copy()
        except Exception:
            break

    return best_n
# EVOLVE-BLOCK-END
