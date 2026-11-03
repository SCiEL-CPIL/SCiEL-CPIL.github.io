"""200 equal circles in the unit square [-1/2,1/2]^2.

Combines:
  * soft-min (log-sum-exp) surrogate + analytic gradient (current program)
  * KDTree force-directed pre-spread (inspiration)
  * targeted bottleneck elimination on the exact objective (measured 1.056 win)
  * final exact local solve + global limiting-pair expansion
Always returns a valid (N,2) array of finite, in-bounds centres.
"""
import numpy as np
import time
import math

from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 140.0  # evaluator kills at 150 s; keep a 10 s safety margin


def _now():
    return time.time()


# ---------------------------------------------------------------------------
# Exact objective
# ---------------------------------------------------------------------------
def _get_radius(pts):
    """Largest common radius for centres pts (in [-1/2,1/2]^2)."""
    if len(pts) == 0:
        return 0.0
    min_x = np.min(pts[:, 0]); max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1]); max_y = np.max(pts[:, 1])
    wall = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    return float(min(wall, min_pair))


# ---------------------------------------------------------------------------
# Soft-min surrogate + analytic gradient  (minimize -soft_min of constraints)
# ---------------------------------------------------------------------------
def _objective_and_grad(x, beta):
    pts = x.reshape((N, 2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)

    C = np.empty(4 * N + n_pairs)
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]

    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx * dx + dy * dy)
    C[4*N:] = dists / 2.0

    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return -np.min(C), np.zeros_like(x)
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp <= 0.0:
        return -np.min(C), np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    weights = exp_shifted / sum_exp

    grad_pts = np.zeros((N, 2))
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]

    valid = dists > 1e-10
    dx_s = np.where(valid, dx, 1.0)
    dy_s = np.where(valid, dy, 1.0)
    ds = np.where(valid, dists, 1.0)
    pw = weights[4*N:]
    np.add.at(grad_pts[:, 0], i_idx, np.where(valid, dx_s / (2.0 * ds), 0.0) * pw)
    np.add.at(grad_pts[:, 0], j_idx, np.where(valid, -dx_s / (2.0 * ds), 0.0) * pw)
    np.add.at(grad_pts[:, 1], i_idx, np.where(valid, dy_s / (2.0 * ds), 0.0) * pw)
    np.add.at(grad_pts[:, 1], j_idx, np.where(valid, -dy_s / (2.0 * ds), 0.0) * pw)

    grad_pts = -grad_pts / beta
    return obj_val, grad_pts.flatten()


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------
def _hex_initial(spacing=0.078):
    pts = []
    row_h = spacing * math.sqrt(3) / 2.0
    y = -0.5
    ridx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if ridx % 2 else 0.0
        x = -0.5 + shift
        while x < 0.5:
            pts.append([x, y])
            x += spacing
        y += row_h
        ridx += 1
    pts = np.array(pts)
    mask = (pts[:, 0] >= -0.499) & (pts[:, 0] <= 0.499) & \
           (pts[:, 1] >= -0.499) & (pts[:, 1] <= 0.499)
    pts = pts[mask]
    if len(pts) < N:
        rng = np.random.RandomState(0)
        return rng.uniform(-0.45, 0.45, (N, 2))
    d2c = np.sum(pts ** 2, axis=1)
    idx = np.argsort(d2c)
    return pts[idx[:N]].copy()


def _random_initial(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 2))


# ---------------------------------------------------------------------------
# Force-directed pre-spread (KDTree)
# ---------------------------------------------------------------------------
def _force_directed(pts, r_target, steps=150, lr=0.004):
    pts = pts.copy()
    rng = np.random.RandomState(1)
    for _ in range(steps):
        forces = np.zeros_like(pts)
        # walls
        lx = pts[:, 0] < -0.5 + r_target
        rx = pts[:, 0] > 0.5 - r_target
        ly = pts[:, 1] < -0.5 + r_target
        ry = pts[:, 1] > 0.5 - r_target
        forces[lx, 0] += 5.0 * (r_target - (pts[lx, 0] + 0.5))
        forces[rx, 0] -= 5.0 * (pts[rx, 0] - (0.5 - r_target))
        forces[ly, 1] += 5.0 * (r_target - (pts[ly, 1] + 0.5))
        forces[ry, 1] -= 5.0 * (pts[ry, 1] - (0.5 - r_target))
        # repulsion
        tree = KDTree(pts)
        cutoff = max(2.0 * r_target, 0.06)
        for i in range(N):
            for j in tree.query_ball_point(pts[i], cutoff):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                d = math.hypot(diff[0], diff[1])
                if d < 1e-10:
                    forces[i] += rng.randn(2) * 0.01
                    continue
                if d < 2.0 * r_target:
                    mag = 10.0 * (2.0 * r_target - d) / d
                    forces[i] += mag * diff
        pts += lr * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    return pts


# ---------------------------------------------------------------------------
# Exact targeted local refinement (bottleneck elimination)
# ---------------------------------------------------------------------------
def _exact_local(pts, steps=600, step=0.0006):
    pts = pts.copy()
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    for _ in range(steps):
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        pi, pj = np.unravel_index(np.argmin(D), D.shape)
        d = D[pi, pj]
        wall = min(pts[pi, 0] + 0.5, 0.5 - pts[pi, 0],
                   pts[pi, 1] + 0.5, 0.5 - pts[pi, 1])
        if d / 2.0 > wall + 1e-10:
            # wall is the bottleneck: push the closest-to-wall point inward
            wx = pts[pi, 0] + 0.5; wX = 0.5 - pts[pi, 0]
            wy = pts[pi, 1] + 0.5; wY = 0.5 - pts[pi, 1]
            m = min(wx, wX, wy, wY)
            if m == wx:
                pts[pi, 0] += step
            elif m == wX:
                pts[pi, 0] -= step
            elif m == wy:
                pts[pi, 1] += step
            else:
                pts[pi, 1] -= step
        else:
            # pair is the bottleneck: push the two points apart along their axis
            diff = pts[pi] - pts[pj]
            dd = math.hypot(diff[0], diff[1])
            if dd < 1e-12:
                pts[pi, 0] += step
                pts[pj, 0] -= step
            else:
                u = diff / dd
                pts[pi] += step * u
                pts[pj] -= step * u
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    return pts


# ---------------------------------------------------------------------------
# Global limiting-pair expansion (keeps all points in bounds, improves min pair)
# ---------------------------------------------------------------------------
def _expand_limiting(pts, steps=80, step=0.0004):
    pts = pts.copy()
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    for _ in range(steps):
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        pi, pj = np.unravel_index(np.argmin(D), D.shape)
        diff = pts[pi] - pts[pj]
        dd = math.hypot(diff[0], diff[1])
        if dd < 1e-12:
            break
        u = diff / dd
        npi = pts[pi] + step * u
        npj = pts[pj] - step * u
        if np.all(np.abs(npi) <= 0.5) and np.all(np.abs(npj) <= 0.5):
            pts[pi] = npi
            pts[pj] = npj
        else:
            # try a smaller step
            ok = False
            for f in (0.5, 0.25, 0.1):
                npi = pts[pi] + step * f * u
                npj = pts[pj] - step * f * u
                if np.all(np.abs(npi) <= 0.5) and np.all(np.abs(npj) <= 0.5):
                    pts[pi] = npi
                    pts[pj] = npj
                    ok = True
                    break
            if not ok:
                break
    return pts


# ---------------------------------------------------------------------------
# Per-initialization pipeline
# ---------------------------------------------------------------------------
def _optimize_init(pts, budget):
    t0 = _now()
    best = pts.copy()
    best_r = _get_radius(best)

    # 1) force-directed pre-spread to escape hex-grid local minima
    r0 = _get_radius(best)
    fd = _force_directed(best, max(r0, 0.03), steps=120, lr=0.004)
    r_fd = _get_radius(fd)
    if r_fd > best_r:
        best, best_r = fd.copy(), r_fd

    # 2) ramping soft-min gradient optimization
    x0 = best.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    for beta in (5, 10, 20, 50, 100, 200):
        if _now() - t0 > budget:
            break

        def fun_g(x, b=beta):
            v, g = _objective_and_grad(x, b)
            return v, g

        try:
            res = minimize(fun_g, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-9})
            cand = res.x.reshape(-1, 2)
        except Exception:
            continue
        cand = np.clip(cand, -0.5, 0.5)
        rc = _get_radius(cand)
        if rc > best_r:
            best, best_r = cand.copy(), rc
            x0 = cand.flatten()

    # 3) targeted exact local refinement
    if _now() - t0 < budget + 5:
        cand = _exact_local(best, steps=500, step=0.0006)
        rc = _get_radius(cand)
        if rc > best_r:
            best, best_r = cand.copy(), rc

    # 4) global limiting-pair expansion
    cand = _expand_limiting(best, steps=100, step=0.0004)
    rc = _get_radius(cand)
    if rc > best_r:
        best, best_r = cand.copy(), rc

    # 5) final exact polish
    cand = _exact_local(best, steps=400, step=0.0002)
    rc = _get_radius(cand)
    if rc > best_r:
        best, best_r = cand.copy(), rc

    return best, best_r


def construct():
    start = time.time()
    deadline = start + DEADLINE

    inits = [
        _hex_initial(0.075),
        _hex_initial(0.078),
        _hex_initial(0.081),
        _random_initial(42),
        _random_initial(123),
    ]

    best_pos = inits[0].copy()
    best_r = _get_radius(best_pos)

    for k, init in enumerate(inits):
        if _now() > deadline - 25:
            break
        budget = max(10.0, deadline - _now() - 25.0)
        try:
            pos, r = _optimize_init(init, budget)
        except Exception:
            continue
        if not (np.all(np.isfinite(pos)) and pos.shape == (N, 2)):
            continue
        pos = np.clip(pos, -0.5, 0.5)
        r = _get_radius(pos)
        if r > best_r:
            best_pos, best_r = pos.copy(), r

    # final safety polish if time remains
    if _now() < deadline - 3:
        try:
            cand = _exact_local(best_pos, steps=300, step=0.0002)
            rc = _get_radius(cand)
            if rc > best_r:
                best_pos, best_r = cand.copy(), rc
        except Exception:
            pass

    best_pos = np.clip(best_pos, -0.5, 0.5)
    if not np.all(np.isfinite(best_pos)):
        rng = np.random.RandomState(0)
        best_pos = rng.uniform(-0.4, 0.4, (N, 2))
    return best_pos
