# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 8
_S = 1.0 / _N
_CS = np.arange(_N, dtype=np.float64) * _S
_DEADLINE = 630.0
_MARGIN = 20.0


def _union_length_1d(a, c):
    """Exact union length (sum of disjoint intervals) of [a_i + c*i/N, a_i + 1/N + c*i/N]."""
    L = a.shape[0]
    lo = a + c * _CS
    hi = lo + _S
    order = np.argsort(lo)
    lo = lo[order]
    hi = hi[order]
    total = 0.0
    cur_lo = lo[0]
    cur_hi = hi[0]
    for k in range(1, L):
        if lo[k] > cur_hi:
            total += cur_hi - cur_lo
            cur_lo = lo[k]
            cur_hi = hi[k]
        else:
            if hi[k] > cur_hi:
                cur_hi = hi[k]
    total += cur_hi - cur_lo
    return total


def _volume_1d(a, cgrid):
    """Exact integral (trapezoidal) of union length over c in [0,1]."""
    L = a.shape[0]
    lo = a[None, :] + cgrid[:, None] * _CS[None, :]
    hi = lo + _S
    order = np.argsort(lo, axis=1)
    lo = np.take_along_axis(lo, order, axis=1)
    hi = np.take_along_axis(hi, order, axis=1)
    total = hi[:, 0] - lo[:, 0]
    cur_hi = hi[:, 0]
    for k in range(1, L):
        gap = lo[:, k] - cur_hi
        cont = gap < 0
        total = total + np.where(cont, 0.0, cur_hi - lo[:, k])
        cur_hi = np.where(cont, np.maximum(cur_hi, hi[:, k]), hi[:, k])
    total = total + (cur_hi - lo[:, 0])
    return 0.5 * (total[0] + total[-1]) + total[1:-1].sum()


def _volume_3d(a, b, cgrid):
    """Exact volume of the product union = Lx * Ly (fast exact objective)."""
    return _volume_1d(a, cgrid) * _volume_1d(b, cgrid)


def _soft_volume_3d(a, b, cgrid, beta):
    """Smooth (log-sum-exp) surrogate for Lx*Ly, differentiable everywhere."""
    L = a.shape[0]
    lo = a[None, :] + cgrid[:, None] * _CS[None, :]
    hi = lo + _S
    lo = np.sort(lo, axis=1)
    hi = np.sort(hi, axis=1)
    cur_hi = hi[:, 0].copy()
    for k in range(1, L):
        cont = lo[:, k] < cur_hi
        cur_hi = np.where(cont, np.maximum(cur_hi, hi[:, k]), hi[:, k])
    length = cur_hi - lo[:, 0]
    mx = length.max()
    Lx = mx + np.log(np.exp(length - mx).sum())
    return Lx * Lx  # symmetric: Lx == Ly for identical a, b


def _grad_soft_volume_3d(a, b, cgrid, beta):
    """Gradient of the soft surrogate w.r.t. (a, b)."""
    L = a.shape[0]
    lo = a[None, :] + cgrid[:, None] * _CS[None, :]
    hi = lo + _S
    lo = np.sort(lo, axis=1)
    hi = np.sort(hi, axis=1)
    cur_hi = hi[:, 0].copy()
    for k in range(1, L):
        cont = lo[:, k] < cur_hi
        cur_hi = np.where(cont, np.maximum(cur_hi, hi[:, k]), hi[:, k])
    length = cur_hi - lo[:, 0]
    mx = length.max()
    w = np.exp(length - mx)
    wsum = w.sum()
    Lx = mx + np.log(wsum)
    # d(Lx)/d(lo_j) = -w_j / wsum
    dlo = -w / wsum
    # d(lo_j)/d(a_j) = c_i
    grad_a = (dlo * cgrid[:, None]).sum(axis=0)
    return np.concatenate([grad_a, grad_a])  # symmetric


def _keich_init():
    pos = int(np.log2(_N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(_N, dtype=np.float64)
    for idx in range(_N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _build_positions(a, b):
    i_grid, j_grid = np.meshgrid(np.arange(_N), np.arange(_N), indexing="ij")
    base_x = a[i_grid.flatten()]
    base_y = b[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _is_valid(positions):
    if positions.shape != (64, 2):
        return False
    if not np.all(np.isfinite(positions)):
        return False
    x = positions[:, 0]
    y = positions[:, 1]
    i = (np.arange(64) // 8).astype(np.float64)
    j = (np.arange(64) % 8).astype(np.float64)
    X0 = x.min()
    X1 = (x + (i + 1) / 8).max()
    Y0 = y.min()
    Y1 = (y + (j + 1) / 8).max()
    return (X1 - X0) > 1e-9 and (Y1 - Y0) > 1e-9


def kakeya_needle_3d8() -> np.ndarray:
    start = time.time()
    cgrid = np.linspace(0.0, 1.0, 2001)

    keich = _keich_init()
    best = _build_positions(keich, keich)
    best_v = _volume_3d(keich, keich, cgrid)
    best_ab = (keich.copy(), keich.copy())

    rng = np.random.default_rng(7)
    beta = 200.0
    max_iters = 300

    n_restarts = 24
    for r in range(n_restarts):
        if time.time() > _DEADLINE - _MARGIN:
            break
        if r == 0:
            a = keich.copy()
            b = keich.copy()
        else:
            a = best_ab[0] + rng.normal(0.0, 0.08, _N)
            b = best_ab[1] + rng.normal(0.0, 0.08, _N)

        x0 = np.concatenate([a, b])

        def fun(xx):
            return _soft_volume_3d(xx[:_N], xx[_N:], cgrid, beta)

        def grad(xx):
            return _grad_soft_volume_3d(xx[:_N], xx[_N:], cgrid, beta)

        try:
            res = minimize(fun, x0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": max_iters, "gtol": 1e-8,
                                    "ftol": 1e-12, "maxls": 40})
        except Exception:
            continue
        a_opt = res.x[:_N]
        b_opt = res.x[_N:]
        v = _volume_3d(a_opt, b_opt, cgrid)
        if np.isfinite(v) and v < best_v:
            best_v = v
            best_ab = (a_opt.copy(), b_opt.copy())
            best = _build_positions(a_opt, b_opt)

    # High-precision polish: tighter grid, more L-BFGS iterations, lower beta.
    if time.time() < _DEADLINE - _MARGIN:
        cgrid2 = np.linspace(0.0, 1.0, 4001)
        beta2 = 800.0
        a = best_ab[0].copy()
        b = best_ab[1].copy()
        x0 = np.concatenate([a, b])

        def fun(xx):
            return _soft_volume_3d(xx[:_N], xx[_N:], cgrid2, beta2)

        def grad(xx):
            return _grad_soft_volume_3d(xx[:_N], xx[_N:], cgrid2, beta2)

        try:
            res = minimize(fun, x0, jac=grad, method="L-BFGS-B",
                           options={"maxiter": 600, "gtol": 1e-10,
                                    "ftol": 1e-14, "maxls": 50})
            a_p = res.x[:_N]
            b_p = res.x[_N:]
            v = _volume_3d(a_p, b_p, cgrid2)
            if np.isfinite(v) and v < best_v:
                best_v = v
                best = _build_positions(a_p, b_p)
        except Exception:
            pass

    if not _is_valid(best):
        best = _build_positions(keich, keich)
    return best


# EVOLVE-BLOCK-END
