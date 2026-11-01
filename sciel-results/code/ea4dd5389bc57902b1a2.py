"""Packing 50 equal circles in a unit circle using a multi-start approach:
   1) hexagonal lattice initialisations (randomly rotated/scaled)
   2) a smooth "pressure" gradient-descent / annealing phase that grows r
   3) L-BFGS-B polish on a smooth max-min objective
   Deadline-guarded."""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # evaluator kills at 150s; keep margin
START = time.perf_counter()

def _time_left():
    return DEADLINE - (time.perf_counter() - START)

def _score(points):
    """Return the feasible common radius; negative if infeasible."""
    if points is None or points.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # distance to boundary
    r_boundary = 1.0 - np.max(np.hypot(points[:, 0], points[:, 1]))
    if r_boundary <= 0:
        return -1.0
    # pairwise min distance / 2
    from scipy.spatial.distance import pdist
    dmin = pdist(points)
    if dmin.size == 0:
        return r_boundary
    r_pair = dmin.min() / 2.0
    return min(r_boundary, r_pair)

def _hex_initial(n, rot=0.0, scale=1.0):
    """Generate ~n points on a hex lattice inside unit disk, return first n closest to origin."""
    spacing = 2.0 / (math.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * math.sqrt(3.0) / 2.0
            pts.append((x, y))
    pts = np.array(pts)
    # rotate
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T * scale
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r <= 1.0 - spacing / 2.0
    pts = pts[mask]
    r = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(r, kind='stable')
    pts = pts[order]
    if len(pts) < n:
        # pad with random points near origin
        extra = np.random.randn(n - len(pts), 2) * spacing * 0.5
        pts = np.vstack([pts, extra])
    return pts[:n]

def _pressure_objective(pts, r_target, mu=50.0):
    """Smooth objective: minimize overlap violations + boundary violations.
       Lower is better. Use as energy to be minimized."""
    x = pts.reshape(-1)
    # pairwise
    P = x.reshape(-1, 2)
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    # violation: max(0, r_target - d)
    viol = np.maximum(0.0, r_target - d)
    E_pair = np.sum(viol ** 2)
    # boundary
    r = np.hypot(P[:, 0], P[:, 1])
    bviol = np.maximum(0.0, r - (1.0 - r_target))
    E_bound = np.sum(bviol ** 2)
    return E_pair + mu * E_bound

def _pressure_gradient(pts, r_target, mu=50.0):
    x = pts.reshape(-1)
    P = x.reshape(-1, 2)
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    # grad of sum max(0, r_target - d)^2 w.r.t. positions
    g = np.zeros_like(P)
    mask = d < r_target
    # for pairs where d < r_target, contribution: -2*(r_target - d) * (diff_i - diff_j)/d
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if d[i, j] < r_target:
                f = -2.0 * (r_target - d[i, j]) / d[i, j]
                g[i] += f * diff[i, j]
                g[j] -= f * diff[i, j]
    # boundary
    r = np.hypot(P[:, 0], P[:, 1])
    mask_b = r > (1.0 - r_target)
    for i in range(len(P)):
        if mask_b[i]:
            f = 2.0 * mu * (r[i] - (1.0 - r_target)) / r[i]
            g[i] += f * P[i]
    return g.reshape(-1)

def _refine(pts, r_start, time_budget):
    """Grow r via annealed gradient steps. Returns best (pts, r)."""
    best_r = r_start
    best_pts = pts.copy()
    cur = pts.copy()
    r = r_start
    lr = 0.01
    t_end = time.perf_counter() + time_budget
    step = 0
    while time.perf_counter() < t_end:
        # try to increase r slightly
        r_try = r * 1.005 + 1e-6
        # do a few gradient steps at r_try
        for it in range(30):
            if time.perf_counter() > t_end:
                break
            g = _pressure_gradient(cur, r_try, mu=50.0)
            cur = cur - lr * g
            cur = cur.reshape(-1, 2)
        # accept if energy decreased
        E_old = _pressure_objective(best_pts, r, mu=50.0)
        E_new = _pressure_objective(cur, r_try, mu=50.0)
        if E_new < E_old:
            best_pts = cur.copy()
            best_r = r_try
            r = r_try
        else:
            # revert and reduce learning rate
            cur = best_pts.copy()
            lr *= 0.9
        step += 1
    return best_pts, best_r

def _lbfgs_polish(pts, r_init):
    """Polish using L-BFGS-B on smooth max-min surrogate."""
    def obj(x):
        P = x.reshape(-1, 2)
        # smooth max-min: use -log-sum-exp of distances
        from scipy.spatial.distance import pdist
        d = pdist(P)
        if d.size == 0:
            return 0.0
        # maximize min d  <=> minimize -min d, but smooth:
        beta = 50.0
        logsumexp = np.log(np.sum(np.exp(-beta * d))) / (-beta)
        r_b = 1.0 - np.max(np.hypot(P[:, 0], P[:, 1]))
        # penalize boundary
        return logsumexp + 10.0 * max(0.0, 0.1 - r_b)
    x0 = pts.reshape(-1)
    res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-8})
    if res.success or res.fun < obj(x0):
        return res.x.reshape(-1, 2)
    return pts

def construct():
    global START, DEADLINE
    START = time.perf_counter()
    DEADLINE = START + 140.0

    np.random.seed(42)

    best_pts = None
    best_r = -1.0

    # Start with hex lattice
    pts0 = _hex_initial(N)
    r0 = _score(pts0)
    best_pts = pts0.copy()
    best_r = r0

    # Multi-start with random rotations and scales
    n_starts = 8
    for s in range(n_starts):
        if _time_left() < 10:
            break
        rot = np.random.uniform(0, 2 * math.pi)
        scale = np.random.uniform(0.85, 1.0)
        pts = _hex_initial(N, rot=rot, scale=scale)
        # small random jitter
        pts += np.random.randn(*pts.shape) * 0.005
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Refine best with pressure growth
    if _time_left() > 15:
        budget = min(30.0, max(5.0, _time_left() * 0.3))
        pts, r = _refine(best_pts, best_r, budget)
        if r > best_r:
            best_r = r
            best_pts = pts

    # L-BFGS polish
    if _time_left() > 5:
        pts = _lbfgs_polish(best_pts, best_r)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts

    # Final validity check
    final = _score(best_pts)
    if final <= 0:
        # fallback: return hex lattice
        best_pts = _hex_initial(N)
        best_r = _score(best_pts)

    return best_pts
