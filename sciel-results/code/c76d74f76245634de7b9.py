"""Circle packing: physics relaxation + SLSQP polish for max min-distance."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 145.0  # seconds (safety margin under 150s wall-clock)
T_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - T_START)


def _score(pts):
    """Largest common radius for a point set inside [-0.5,0.5]^2."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    min_wall = (0.5 - np.abs(pts)).min()
    return min(0.5 * min_pair, min_wall)


def _lj_force(pts, eps=0.015, rep=2.0):
    """Repulsive force pushing points apart (soft LJ-like)."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Force magnitude ~ rep / dist^3, direction = d / dist
    mag = rep / (dist ** 3)
    F = d * mag[:, :, None]
    F = np.sum(F, axis=0)
    # Boundary: push inward near walls
    wall = 0.5 - np.abs(pts)
    F += eps * (np.sign(pts) * (wall < 0.02).astype(float)) * (0.02 - np.minimum(wall, 0.02))
    return F


def _relax(pts, steps=15000, dt_init=0.002, dt_end=0.00005):
    """Simulated-annealing-style relaxation of the repulsive dynamics."""
    pts = pts.astype(float).copy()
    rng = np.random.default_rng(0)
    for it in range(steps):
        frac = it / max(1, steps - 1)
        dt = dt_init * (dt_end / dt_init) ** frac
        F = _lj_force(pts)
        pts = pts + dt * F + dt * 0.001 * rng.standard_normal(pts.shape)
        # Clip to stay inside (soft)
        pts = np.clip(pts, -0.5, 0.5)
        if it % 2000 == 0 and _remaining() < 5.0:
            break
    return pts


def _polish(pts, budget=40.0):
    """SLSQP: maximize the minimum of (pair distances) and (wall distances)."""
    pts = np.asarray(pts, dtype=float).reshape(-1).copy()
    n = N

    def neg_score(x):
        p = x.reshape(n, 2)
        d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair = d.min()
        min_wall = (0.5 - np.abs(p)).min()
        return -min(min_pair, 2.0 * min_wall)

    # We want to maximize min(min_pair, 2*min_wall) because score = min(0.5*min_pair, min_wall)
    # = 0.5 * min(min_pair, 2*min_wall). So maximizing min(min_pair,2*min_wall) is equivalent.

    best = pts.copy()
    best_val = neg_score(pts)
    t0 = time.time()
    for it in range(300):
        if time.time() - t0 > budget or _remaining() < 8.0:
            break
        res = minimize(
            neg_score,
            pts,
            method='SLSQP',
            options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10},
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            v = neg_score(res.x)
            if v < best_val - 1e-15:
                best_val = v
                best = res.x.copy()
            pts = res.x.copy()
            # Nudge to escape local minima
            if it % 3 == 0:
                rng = np.random.default_rng(hash((it, best_val)) & 0xFFFFFFFF)
                pts = pts + 0.0005 * rng.standard_normal(pts.shape)
                pts = np.clip(pts, -0.5, 0.5)
    return best.reshape(n, 2)


def construct():
    if _remaining() < 3.0:
        # Not enough time: return a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # 1) Initial configuration: hexagonal-ish jittered grid
    rng = np.random.default_rng(1)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    # Jitter to break symmetry
    pts = grid + 0.003 * rng.standard_normal(grid.shape)
    pts = np.clip(pts, -0.49, 0.49)

    # 2) Physics relaxation
    if _remaining() > 25.0:
        pts = _relax(pts, steps=20000)

    # 3) SLSQP polish
    if _remaining() > 15.0:
        budget = max(5.0, min(40.0, _remaining() - 10.0))
        pts = _polish(pts, budget=budget)

    # Final clip for safety
    pts = np.clip(pts, -0.5, 0.5)
    return pts


# Quick self-test when run directly (not used by evaluator)
if __name__ == "__main__":
    c = construct()
    print("shape:", c.shape)
    print("score:", _score(c))
