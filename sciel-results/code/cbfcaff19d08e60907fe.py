"""Symmetry-constrained smooth epigraph optimizer for 300-circle packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300
_BASE = 100          # circles per 120-degree sector (3-fold symmetry)
_EPS = 1e-12

_DEADLINE = 110.0    # seconds; leave a safety margin under the 120 s limit
_start = None


def _build_initial():
    """3-fold symmetric initial guess: 100 base circles in a hex grid,
    restricted to the 120-degree sector, then replicated by rotation."""
    # hex grid spacing chosen so ~100 points fall in the sector near the disk
    for spacing in np.linspace(0.10, 0.04, 60):
        s3 = spacing * np.sqrt(3.0) / 2.0
        k = int(1.0 / spacing) + 3
        base = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * s3
                base.append((x, y))
        base = np.array(base)
        r = np.hypot(base[:, 0], base[:, 1])
        base = base[r <= 1.0 - spacing / 2.0]
        # keep only points in the 120-degree sector: angle in [0, 2pi/3)
        ang = np.arctan2(base[:, 1], base[:, 0])
        ang = np.mod(ang, 2.0 * np.pi)
        base = base[ang < 2.0 * np.pi / 3.0]
        if len(base) >= _BASE:
            # take the _BASE points closest to the sector centroid for balance
            order = np.argsort(r, kind="stable")
            base = base[order[:_BASE]]
            break
    # replicate by 120 and 240 degree rotations
    th = 2.0 * np.pi / 3.0
    R1 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    R2 = np.array([[np.cos(2 * th), -np.sin(2 * th)], [np.sin(2 * th), np.cos(2 * th)]])
    pts = np.vstack([base, base @ R1.T, base @ R2.T])
    # enforce strict interior
    rr = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts * np.minimum(1.0, (1.0 - 1e-4) / np.maximum(rr, _EPS))[:, None]
    return pts


def _score(pts):
    """Largest feasible common radius for a set of centres."""
    n = len(pts)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d.min() / 2.0
    bound = 1.0 - np.hypot(pts[:, 0], pts[:, 1]).max()
    return min(pair, bound)


def construct():
    global _start
    _start = time.time()
    pts = _build_initial()
    n = len(pts)
    x0 = np.concatenate([pts.ravel(), [_score(pts) + 1e-3]])

    def objective(x):
        p = x[:2 * n].reshape(n, 2)
        t = x[2 * n]
        # pairwise distances
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        pair = d.ravel()
        pair = pair[pair < np.inf]
        # boundary distances
        bound = 1.0 - np.hypot(p[:, 0], p[:, 1])
        # soft-min of (pair - t) and (bound - t); large scale -> near min
        scale = 300.0
        a = (pair - t) / scale
        b = (bound - t) / scale
        m = np.concatenate([a, b])
        mmax = m.max()
        # log-sum-exp of exp(-(m - mmax)); want to MAXIMIZE the min => MINIMIZE -min
        # soft-min value = mmax - logsumexp(-(m-mmax))/scale
        lse = mmax - np.log(np.sum(np.exp(-(m - mmax))))
        soft_min = lse / scale
        # penalize infeasibility (soft_min < 0 means some constraint violated)
        return soft_min + 1e-6 * t  # tiny term to encourage larger t

    def objective_grad(x):
        p = x[:2 * n].reshape(n, 2)
        t = x[2 * n]
        d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        bound = 1.0 - np.hypot(p[:, 0], p[:, 1])
        scale = 300.0
        a = (d.ravel() - t) / scale
        a = a[np.isfinite(a)]
        b = (bound - t) / scale
        m = np.concatenate([a, b])
        mmax = m.max()
        w = np.exp(-(m - mmax))
        wsum = w.sum()
        gw = w / wsum
        # d soft_min / d (pair) = -gw_pair/scale ; d/d t = -(sum gw)/scale = -1/scale
        gpair = -gw[:len(a)] / scale
        gbound = -gw[len(a):] / scale
        # gradient wrt p from pairwise distances
        gp = np.zeros_like(p)
        # recompute to map back
        d2 = d.copy()
        np.fill_diagonal(d2, 1.0)
        # gp_i = sum_j gpair_ij * (p_i - p_j)/d_ij
        D = d2
        G = np.zeros((n, n))
        idx = ~np.eye(n, dtype=bool)
        G[idx] = gpair[idx]
        gp = np.sum((G[:, None, :] * (p[:, None, :] - p[None, :, :]) / D[:, None, :][:, :, None]), axis=1)
        gb = np.sum(gbound[:, None] * (-p / np.maximum(np.hypot(p[:, 0], p[:, 1]), _EPS)), axis=0)
        gptot = gp + gb
        gt = -1.0 / scale + 1e-6
        return np.concatenate([gptot.ravel(), [gt]])

    best = pts.copy()
    best_r = _score(pts)

    def _deadline_hit():
        return (time.time() - _start) > _DEADLINE

    # main optimization
    for it in range(1, 8):
        if _deadline_hit():
            break
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           jac=objective_grad,
                           options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8,
                                    "maxls": 40})
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = res.x[:2 * n].reshape(n, 2)
                rr = _score(cand)
                if np.isfinite(rr) and rr > best_r:
                    best_r = rr
                    best = cand.copy()
                    x0 = res.x.copy()
        except Exception:
            break

    # polish pass: tighter, more iterations, from best
    if not _deadline_hit():
        x0 = np.concatenate([best.ravel(), [best_r + 1e-3]])
        try:
            res = minimize(objective, x0, method="L-BFGS-B",
                           jac=objective_grad,
                           options={"maxiter": 1500, "ftol": 1e-15, "gtol": 1e-10,
                                    "maxls": 60})
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = res.x[:2 * n].reshape(n, 2)
                rr = _score(cand)
                if np.isfinite(rr) and rr > best_r:
                    best_r = rr
                    best = cand.copy()
        except Exception:
            pass

    # final safety: ensure strictly inside and finite
    rr = np.hypot(best[:, 0], best[:, 1])
    best = best * np.minimum(1.0, (1.0 - 1e-6) / np.maximum(rr, _EPS))[:, None]
    return best
