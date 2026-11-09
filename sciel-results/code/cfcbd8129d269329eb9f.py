# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_TOL = 1e-7
_DEADLINE = 600.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _smooth_penalty(centers, sides, beta):
    """Smooth (log-sum-exp) surrogate for the max violation. Returns (value, logsumexp)."""
    n = sides.shape[0]
    cx = centers[:, 0]
    cy = centers[:, 1]

    # boundary violations
    b1 = sides / 2.0 - cx
    b2 = cx + sides / 2.0 - 1.0
    b3 = sides / 2.0 - cy
    b4 = cy + sides / 2.0 - 1.0
    b = np.concatenate([b1, b2, b3, b4])
    mb = np.max(b)
    lb = mb + np.log(np.sum(np.exp(b - mb))) if b.size else 0.0

    # pairwise overlap violations
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = (sides[i] + sides[j]) / 2.0 - abs(cx[i] - cx[j])
            u = (sides[i] + sides[j]) / 2.0 - abs(cy[i] - cy[j])
            d[i, j] = d[j, i] = max(v, u)
    mu = np.max(d)
    ld = mu + np.log(np.sum(np.exp(d - mu)))

    m = max(lb, ld)
    return m + np.log(np.exp(lb - m) + np.exp(ld - m))


def _objective(x, beta):
    n = _N
    centers = x[: 2 * n].reshape(n, 2)
    sides = x[2 * n:]
    val, _ = _smooth_penalty(centers, sides, beta)
    return -np.sum(sides) + beta * val


def _valid(centers, sides):
    n = sides.shape[0]
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    cx = centers[:, 0]
    cy = centers[:, 1]
    for i in range(n):
        if cx[i] - sides[i] / 2.0 < -_TOL or cx[i] + sides[i] / 2.0 > 1.0 + _TOL:
            return False
        if cy[i] - sides[i] / 2.0 < -_TOL or cy[i] + sides[i] / 2.0 > 1.0 + _TOL:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - _TOL
            if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                return False
    return True


def _penalty_value(centers, sides):
    n = sides.shape[0]
    cx = centers[:, 0]
    cy = centers[:, 1]
    p = 0.0
    for i in range(n):
        p = max(p, sides[i] / 2.0 - cx[i], cx[i] + sides[i] / 2.0 - 1.0,
                sides[i] / 2.0 - cy[i], cy[i] + sides[i] / 2.0 - 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            v = (sides[i] + sides[j]) / 2.0 - abs(cx[i] - cx[j])
            u = (sides[i] + sides[j]) / 2.0 - abs(cy[i] - cy[j])
            p = max(p, max(v, u))
    return p


def _project(x):
    """Hard projection back into the feasible box (keeps a candidate valid)."""
    n = _N
    centers = x[: 2 * n].reshape(n, 2).copy()
    sides = x[2 * n:].copy()
    sides = np.clip(sides, 1e-9, 1.0)
    centers[:, 0] = np.clip(centers[:, 0], sides / 2.0, 1.0 - sides / 2.0)
    centers[:, 1] = np.clip(centers[:, 1], sides / 2.0, 1.0 - sides / 2.0)
    return np.concatenate([centers.ravel(), sides])


def _coordinate_ascent(x, iters=30000, step0=0.02, decay=0.9998):
    """Greedy coordinate ascent on the exact (nonsmooth) objective."""
    n = _N
    x = x.copy()
    step = step0
    best = -np.sum(x[2 * n:])
    for _ in range(iters):
        improved = False
        for k in range(2 * n + n):
            d = 1.0
            while d > 1e-12:
                xp = x.copy()
                xp[k] += d
                if xp[2 * n:] is not None and np.all(xp[2 * n:] > 0):
                    vp = _penalty_value(xp[: 2 * n].reshape(n, 2), xp[2 * n:])
                    if vp <= _TOL:
                        fp = -np.sum(xp[2 * n:])
                    else:
                        fp = -np.sum(xp[2 * n:]) + 1e6 * vp
                else:
                    fp = 1e12
                xm = x.copy()
                xm[k] -= d
                if np.all(xm[2 * n:] > 0):
                    vm = _penalty_value(xm[: 2 * n].reshape(n, 2), xm[2 * n:])
                    if vm <= _TOL:
                        fm = -np.sum(xm[2 * n:])
                    else:
                        fm = -np.sum(xm[2 * n:]) + 1e6 * vm
                else:
                    fm = 1e12
                if fp < best - 1e-12:
                    x = xp
                    best = fp
                    improved = True
                elif fm < best - 1e-12:
                    x = xm
                    best = fm
                    improved = True
                else:
                    break
        if not improved:
            step *= decay
            if step < 1e-9:
                break
    return x


def _nm(x, beta, maxiter=4000, tol=1e-11):
    res = minimize(_objective, x, args=(beta,), method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": tol, "fatol": tol,
                            "adaptive": True})
    return res.x


def _lbfgs(x, beta, maxiter=2000, tol=1e-10):
    res = minimize(_objective, x, args=(beta,), method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-8,
                            "maxls": 30, "maxfun": 20000})
    return res.x


def _init_3x3():
    """3x3 grid with the 10th cell merged into a 2x2 block (bottom-right)."""
    s = 1.0 / 3.0
    centers = np.zeros((_N, 2))
    sides = np.zeros(_N)
    # 8 single squares: top two full rows + middle-left
    pos = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2),
    ]
    for i, (c, r) in enumerate(pos):
        centers[i] = [(c + 0.5) * s, (r + 0.5) * s]
        sides[i] = s
    # 2x2 block at columns 1,2 rows 2,3
    centers[7] = [1.5 * s, 2.5 * s]
    sides[7] = 2 * s
    centers[8] = [1.5 * s, 0.5 * s]
    sides[8] = s
    centers[9] = [0.5 * s, 0.5 * s]
    sides[9] = s
    return np.concatenate([centers.ravel(), sides])


def _init_grid():
    s = 1.0 / 3.0
    centers = np.zeros((_N, 2))
    sides = np.zeros(_N)
    for i in range(_N):
        c = i % 3
        r = i // 3
        centers[i] = [(c + 0.5) * s, (r + 0.5) * s]
        sides[i] = s
    return np.concatenate([centers.ravel(), sides])


def _init_random(rng):
    sides = 0.25 + 0.30 * rng.random(_N)
    centers = np.zeros((_N, 2))
    centers[:, 0] = sides / 2.0 + (1.0 - sides) * rng.random(_N)
    centers[:, 1] = sides / 2.0 + (1.0 - sides) * rng.random(_N)
    return np.concatenate([centers.ravel(), sides])


def _score(x):
    n = _N
    centers = x[: 2 * n].reshape(n, 2)
    sides = x[2 * n:]
    if not _valid(centers, sides):
        return None
    return float(np.sum(sides))


def _refine(x, deadline):
    """Multi-stage: coord ascent -> Nelder-Mead (annealed beta) -> L-BFGS-B -> coord polish."""
    n = _N
    x = _project(x)
    # Stage 1: coordinate ascent to remove gross overlaps
    if _remaining() > 60:
        x = _coordinate_ascent(x, iters=12000, step0=0.02, decay=0.9999)
    # Stage 2: Nelder-Mead with annealed smooth penalty
    for beta in [1000.0, 500.0, 200.0, 80.0, 30.0]:
        if _remaining() < 40:
            break
        x = _nm(x, beta, maxiter=3000, tol=1e-11)
    # Stage 3: L-BFGS-B
    if _remaining() > 60:
        x = _lbfgs(x, 200.0, maxiter=1500, tol=1e-10)
    # Stage 4: coordinate ascent polish (exact)
    if _remaining() > 40:
        x = _coordinate_ascent(x, iters=40000, step0=0.01, decay=0.99995)
    # Stage 5: final Nelder-Mead high-precision polish
    if _remaining() > 40:
        x = _nm(x, 30.0, maxiter=4000, tol=1e-12)
    if _remaining() > 30:
        x = _coordinate_ascent(x, iters=20000, step0=0.005, decay=0.99995)
    return x


def squares_in_square10() -> tuple:
    best_x = None
    best_score = -1.0
    rng = np.random.default_rng(12345)

    # Seed candidates
    seeds = [_init_3x3(), _init_grid()]
    for _ in range(4):
        seeds.append(_init_random(rng))

    for seed in seeds:
        if _remaining() < 90:
            break
        x = _refine(seed, _DEADLINE)
        sc = _score(x)
        if sc is not None and sc > best_score:
            best_score = sc
            best_x = x

    if best_x is None:
        # Fallback: always-valid 3x3 grid
        x = _init_3x3()
        best_x = x

    n = _N
    centers = best_x[: 2 * n].reshape(n, 2).astype(np.float64)
    sides = best_x[2 * n:].astype(np.float64)
    return centers, sides


# EVOLVE-BLOCK-END
