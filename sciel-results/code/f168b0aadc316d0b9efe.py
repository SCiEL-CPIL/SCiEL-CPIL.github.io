"""
Smooth surrogate (log-sum-exp soft-min) + L-BFGS-B strategy for packing 50 equal
spheres in the unit cube [-1/2, 1/2]^3.

Structural change vs. the seed program:
  - The seed uses a hand-rolled repulsion force loop + coordinate descent on the
    RAW (nonsmooth) min objective. That is a 0th-order / 1st-order heuristic that
    is slow per-iteration (O(N^2) python loops) and gets stuck because it only
    moves along single coordinates.
  - This program instead builds a SMOOTH, differentiable surrogate of the max-min
    radius problem and hands it to a real quasi-Newton gradient solver (L-BFGS-B).
    The soft-min via log-sum-exp is differentiable everywhere, so the solver can
    take large, well-directed steps in the full 150-dim space at once, and we
    anneal the sharpness (beta) to drive the surrogate toward the true min.
  - A final short EXACT subgradient polish refines the true (nonsmooth) objective.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5
# Wall clearance margin so spheres stay strictly inside the cube.
WALL = 0.49999
BOUNDS = [(-WALL, WALL)] * (3 * N)


def _compute_radius(pts):
    """True max common radius (nonsmooth). pts: (N,3)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    wall = HALF - np.abs(pts)
    min_wall = np.min(wall)
    return min(min_pair * 0.5, min_wall)


def _softmin_logsumexp(x, beta):
    """Smooth min of array x via log-sum-exp: -log(sum(exp(-beta x)))/beta."""
    x = np.asarray(x, dtype=float)
    m = np.min(x)
    s = np.sum(np.exp(-beta * (x - m)))
    return m - np.log(s) / beta


def _surrogate(pts_flat, beta):
    """
    Negative smooth radius (to be MINIMIZED).
    Returns (value, gradient) for L-BFGS-B.
    """
    pts = pts_flat.reshape(N, 3)
    # ---- pairwise distances ----
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    # avoid zero on diagonal / coincident
    d2 = np.maximum(d2, 1e-12)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, np.inf)
    # only upper triangle to count each pair once
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]

    # ---- wall distances ----
    wall = HALF - np.abs(pts)            # (N,3)
    wall_flat = wall.ravel()

    # ---- soft min of pair distances and wall distances ----
    min_pair = _softmin_logsumexp(pair_dist, beta)
    min_wall = _softmin_logsumexp(wall_flat, beta)

    # radius = min(min_pair/2, min_wall); smooth it with another soft min
    a = min_pair * 0.5
    b = min_wall
    r_soft = _softmin_logsumexp(np.array([a, b]), beta)

    value = -r_soft

    # ---- gradient ----
    # d r_soft / d a, d b  (soft min of [a,b])
    ea = np.exp(-beta * (a - np.minimum(a, b)))
    eb = np.exp(-beta * (b - np.minimum(a, b)))
    ea /= (ea + eb)
    eb /= (ea + eb)
    dr_da = ea
    dr_db = eb

    # d min_pair / d pair_dist  (soft min of pair_dist)
    mp = np.minimum(pair_dist, np.min(pair_dist))
    w_pair = np.exp(-beta * (pair_dist - mp))
    w_pair /= np.sum(w_pair)            # (M,)

    # d min_wall / d wall_flat
    mw = np.minimum(wall_flat, np.min(wall_flat))
    w_wall = np.exp(-beta * (wall_flat - mw))
    w_wall /= np.sum(w_wall)            # (3N,)

    grad = np.zeros_like(pts_flat)

    # ---- wall contribution ----
    # d wall / d pts: wall = HALF - |pts| -> d/dx = -sign(x)
    sign_pts = np.sign(pts)
    # w_wall is (3N,) aligned with pts.ravel(); d min_wall / d pts = -w_wall * sign
    dminwall_dpts = -(w_wall.reshape(N, 3) * sign_pts)
    grad += dr_db * dminwall_dpts.reshape(-1)

    # ---- pair contribution ----
    # pair_dist[m] = ||p_i - p_j|| ; d/d p_i = (p_i - p_j)/dist ; d/d p_j = -(p_i-p_j)/dist
    # We need d min_pair / d pts. Build via scattering.
    dminpair_dpts = np.zeros((N, 3))
    # w_pair[m] is d min_pair / d pair_dist[m]
    # d pair_dist[m] / d p_i = (p_i - p_j)/pair_dist[m]
    # vectorize: for each pair (i,j)
    ii = iu[0]
    jj = iu[1]
    vec = diff[iu]                       # p_i - p_j  (M,3)
    unit = vec / pair_dist[:, None]      # (M,3)
    # d min_pair / d p_i = w_pair[m] * unit[m]
    np.add.at(dminpair_dpts, ii, w_pair[:, None] * unit)
    np.add.at(dminpair_dpts, jj, -w_pair[:, None] * unit)
    # d (min_pair/2) / d pts = 0.5 * dminpair_dpts
    grad += dr_da * 0.5 * dminpair_dpts.reshape(-1)

    return value, grad


def _anneal_optimize(x0, time_budget):
    """Anneal beta from soft to sharp, running L-BFGS-B each stage."""
    start = time.time()
    x = x0.copy()
    betas = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
    for beta in betas:
        if time.time() - start > time_budget - 3.0:
            break
        stage_budget = time_budget - (time.time() - start) - 3.0
        if stage_budget < 2.0:
            break
        res = minimize(
            _surrogate, x, args=(beta,), jac=True,
            method='L-BFGS-B', bounds=BOUNDS,
            options={'maxiter': 400, 'maxfun': 4000,
                     'ftol': 1e-12, 'gtol': 1e-9,
                     'maxls': 50, 'maxcor': 30}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            x = res.x
    return x


def _subgradient_polish(pts, iters=200):
    """Exact subgradient ascent on the true nonsmooth radius."""
    start = time.time()
    pts = pts.copy()
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    step = 0.004
    for it in range(iters):
        if time.time() - start > 6.0:
            break
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        dist = np.sqrt(d2)
        wall = HALF - np.abs(pts)

        min_pair = np.min(dist)
        min_wall = np.min(wall)

        forces = np.zeros_like(pts)
        if min_pair * 0.5 <= min_wall:
            # active: pair constraint -> push apart the closest pairs
            for idx in np.argwhere(dist <= min_pair + 1e-6):
                i, j = idx[0], idx[1]
                if i == j:
                    continue
                d = pts[i] - pts[j]
                dn = np.linalg.norm(d)
                if dn < 1e-9:
                    d = np.random.randn(3) * 1e-3
                    dn = np.linalg.norm(d)
                f = d / dn
                forces[i] += f
                forces[j] -= f
        else:
            # active: wall constraint -> push inward
            for idx in np.argwhere(wall <= min_wall + 1e-6):
                i, k = idx[0], idx[1]
                forces[i][k] -= np.sign(pts[i][k])
        fnorm = np.linalg.norm(forces, axis=1, keepdims=True)
        fnorm[fnorm == 0] = 1.0
        new_pts = pts + step * forces / fnorm
        new_pts = np.clip(new_pts, -WALL, WALL)
        r = _compute_radius(new_pts)
        if r > best_r:
            best_r = r
            best_pts = new_pts.copy()
            pts = new_pts
            step = min(step * 1.1, 0.01)
        else:
            step *= 0.9
    return best_pts, best_r


def _make_starts():
    """Diverse, non-overlapping initial configurations."""
    starts = []

    # 1) jittered 4x4x4 grid (take 50 of 64)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(0)
    g = grid + rng.normal(0, 0.015, grid.shape)
    starts.append(np.clip(g, -WALL, WALL).copy())

    # 2) FCC-like lattice points inside cube
    pts = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for off in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + off[0], j + off[1], k + off[2]])
                    if np.all(np.abs(p) <= 0.5):
                        pts.append(p)
    pts = np.array(pts)
    if len(pts) > N:
        idx = rng.choice(len(pts), N, replace=False)
        pts = pts[idx]
    else:
        extra = rng.normal(0, 0.3, (N - len(pts), 3))
        pts = np.vstack([pts, extra])
    starts.append(np.clip(pts, -WALL, WALL).copy())

    # 3) random repulsive
    r3 = np.random.default_rng(7)
    p3 = r3.normal(0, 0.4, (N, 3))
    for _ in range(60):
        for i in range(N):
            for j in range(i + 1, N):
                d = p3[i] - p3[j]
                dn = np.linalg.norm(d)
                if dn < 0.18:
                    f = d / max(dn, 1e-9) * 0.01
                    p3[i] += f
                    p3[j] -= f
    starts.append(np.clip(p3, -WALL, WALL).copy())

    # 4) pure random
    r4 = np.random.default_rng(13)
    starts.append(np.clip(r4.uniform(-WALL, WALL, (N, 3)), -WALL, WALL).copy())

    return starts


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    starts = _make_starts()

    best_pts = starts[0].copy()
    best_r = _compute_radius(best_pts)

    # time budget per start
    n_starts = len(starts)
    per_start = (DEADLINE - (time.time() - t0) - 8.0) / n_starts
    per_start = max(per_start, 12.0)

    for s in starts:
        if time.time() - t0 > DEADLINE - 10.0:
            break
        budget = min(per_start, DEADLINE - (time.time() - t0) - 10.0)
        if budget < 6.0:
            break
        try:
            x = _anneal_optimize(s.ravel(), budget)
            pts = np.clip(x.reshape(N, 3), -WALL, WALL)
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
        except Exception:
            continue

    # final exact subgradient polish on the best
    if time.time() - t0 < DEADLINE - 4.0:
        try:
            best_pts, best_r = _subgradient_polish(best_pts, iters=200)
        except Exception:
            pass

    best_pts = np.clip(best_pts, -WALL, WALL)
    if not np.all(np.isfinite(best_pts)):
        best_pts = starts[0].copy()
    return best_pts
