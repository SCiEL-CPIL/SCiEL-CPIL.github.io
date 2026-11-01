"""
Packing 50 equal circles in the unit square [-0.5, 0.5]^2.
Combined approach:
- Fast vectorized scoring via cKDTree (from inspiration 2)
- Hexagonal lattice + grid initializations (from inspiration 2)
- Multi-restart local search with random perturbations (from inspiration 2)
- Final Nelder-Mead / Powell gradient-free polish (from current program)
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50


def _score(pts):
    """Return the achievable common radius for a centre array (N,2)."""
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # Boundary term: distance from each centre to the nearest side.
    d_bound = np.min(0.5 - np.abs(pts), axis=1)
    r = float(np.min(d_bound))
    # Pairwise term: half the minimum centre distance.
    if N > 1:
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        dmin = float(np.min(d[:, 1]))
        r = min(r, dmin / 2.0)
    return r if r > 0 else 0.0


def _valid(pts):
    return (pts.shape == (N, 2)
            and np.all(np.isfinite(pts))
            and np.all(pts >= -0.5 - 1e-9)
            and np.all(pts <= 0.5 + 1e-9))


def _neg_radius(params):
    """Objective: minimize negative radius (i.e., maximize radius)."""
    points = params.reshape(-1, 2)
    if np.any(np.abs(points) > 0.5 + 1e-8):
        return 1e6
    r = _score(points)
    if r <= 0:
        return 1e6
    return -r


def _hex_initial():
    """Hexagonal (triangular) lattice, rows staggered, clipped to square."""
    s = 0.152
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    xs = np.arange(-0.5, 0.5 + 1e-9, dx)
    pts = []
    row = 0
    y = -0.5
    while y <= 0.5 + 1e-9:
        offset = (row % 2) * (dx / 2.0)
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
        y += dy
        row += 1
    pts = np.array(pts, dtype=float)
    if len(pts) < N:
        grid = _grid_initial()
        pts = np.vstack([pts, grid[len(pts):]])
    else:
        # Keep N points closest to the centre to maximise boundary margin.
        d2 = np.sum(pts * pts, axis=1)
        order = np.argsort(d2)
        pts = pts[order[:N]]
    return pts.copy()


def _grid_initial():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _perturb_and_refine(pts, rng, step0, steps, cool, time_budget):
    """Local search: random perturbations accepted if score improves, with cooling."""
    best = pts.copy()
    best_s = _score(best)
    step = step0
    t0 = time.time()
    for it in range(steps):
        if time.time() - t0 > time_budget:
            break
        cand = best.copy()
        n_move = rng.integers(1, max(2, N // 4) + 1)
        idx = rng.choice(N, size=n_move, replace=False)
        cand[idx] += rng.normal(0.0, step, size=(n_move, 2))
        cand = np.clip(cand, -0.4999999, 0.4999999)
        s = _score(cand)
        if s > best_s + 1e-15:
            best = cand
            best_s = s
            step = max(step0 * 0.5, step * 1.0)
        else:
            step *= cool
        step = min(step, step0)
    return best, best_s


def _polish(pts, deadline):
    """Gradient-free polish using Nelder-Mead / Powell."""
    best_points = pts.copy()
    best_r = _score(best_points)
    x0 = best_points.flatten()
    try:
        result = minimize(
            _neg_radius, x0, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < _neg_radius(x0):
            new_points = result.x.reshape(-1, 2)
            new_points = np.clip(new_points, -0.49999999, 0.49999999)
            r_new = _score(new_points)
            if r_new > best_r:
                best_points = new_points
                best_r = r_new
    except Exception:
        pass
    return best_points, best_r


def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin before 150s kill

    rng = np.random.default_rng(42)

    # Initial candidates
    candidates = [_hex_initial(), _grid_initial()]
    best_pts = candidates[0].copy()
    best_s = _score(best_pts)
    for c in candidates[1:]:
        s = _score(c)
        if s > best_s:
            best_s = s
            best_pts = c.copy()

    # Multi-restart local search
    step0 = 0.04
    steps_per_restart = 3000
    cool = 0.995
    max_restarts = 15

    for r in range(max_restarts):
        if time.time() > deadline - 30.0:  # leave 30s for polish
            break
        if r == 0:
            cur = best_pts.copy()
        elif r % 3 == 1:
            cur = _hex_initial()
            cur += rng.normal(0.0, 0.02, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)
        elif r % 3 == 2:
            cur = _grid_initial()
            cur += rng.normal(0.0, 0.02, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)
        else:
            cur = best_pts.copy()
            cur += rng.normal(0.0, 0.03, size=(N, 2))
            cur = np.clip(cur, -0.4999999, 0.4999999)

        if not _valid(cur):
            cur = best_pts.copy()

        remaining = deadline - 30.0 - time.time()
        if remaining < 0.1:
            break
        cur, s = _perturb_and_refine(cur, rng, step0, steps_per_restart, cool,
                                     time_budget=max(0.1, remaining))
        if _valid(cur) and s > best_s:
            best_s = s
            best_pts = cur.copy()

    # Final polish with gradient-free optimization
    if time.time() < deadline - 5.0:
        best_pts, best_s = _polish(best_pts, deadline)

    # Final safety
    best_pts = np.clip(best_pts, -0.499999999999, 0.499999999999)
    if not _valid(best_pts):
        best_pts = _grid_initial()

    return best_pts


# EVOLVE-BLOCK-START
# construct() is the entry point
# EVOLVE-BLOCK-END
