# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 600.0
_MARGIN = 30.0  # safety margin before the 630 s kill


def _polygon_score_fast(pts):
    """Vectorized minimum-vertex-score (0 if any hard fail)."""
    n = pts.shape[0]
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # normalize
    c = pts.mean(axis=0)
    q = pts - c
    d = np.linalg.norm(q, axis=1)
    m = d.mean()
    if m <= 0.01:
        return 0.0
    q = q / m
    # convexity: cross products all same sign, no repeats
    v = np.roll(q, -1, axis=0) - q
    cross = q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
    # pairwise distances (n x n)
    diff = q[:, None, :] - q[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    ds = np.sort(dist, axis=1)  # n x 19
    if np.any(ds[:, 0] < 0.05):
        return 0.0
    gaps = ds[:, 1:] - ds[:, :-1]  # n x 18
    k = np.argmin(gaps, axis=1)
    D = 0.5 * (ds[:, k] + ds[:, k + 1])
    if np.any(D < 0.1):
        return 0.0
    # select 4 distances per vertex
    kk = k[:, None]
    idx_lo = np.where((k == 0), 0, np.where((k == 17), 15, k - 1))
    sel = np.stack([
        ds[np.arange(n), idx_lo],
        ds[np.arange(n), idx_lo + 1],
        ds[np.arange(n), idx_lo + 2],
        ds[np.arange(n), idx_lo + 3],
    ], axis=1)  # n x 4
    Dv = D[:, None]
    ratio = np.maximum(sel / Dv, Dv / sel)
    raw = ratio.mean(axis=1)
    score = np.minimum(1.0, 1.0 / raw)
    return float(score.min())


def _build_from_angles(thetas, L):
    """Close a 20-gon: equal edge lengths L, directions thetas (with closure correction)."""
    n = 20
    e = L * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    s = e.sum(axis=0)
    e = e - s / n
    pts = np.zeros((n, 2))
    for i in range(1, n):
        pts[i] = pts[i - 1] + e[i - 1]
    return pts


def _neg_score(theta):
    pts = _build_from_angles(theta, 1.0)
    return -_polygon_score_fast(pts)


def _optimize_from(theta0):
    """Bounded Nelder-Mead on the 20 edge directions (normalized to a loop)."""
    L = 1.0
    res = minimize(_neg_score, theta0, method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-7, fatol=1e-7))
    pts = _build_from_angles(res.x, L)
    return pts, _polygon_score_fast(pts)


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Baseline: regular polygon (valid, ~0.75).
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base = np.stack([np.cos(a), np.sin(a)], axis=1)
    best = base.copy()
    best_s = _polygon_score_fast(base)

    # Candidate edge-direction seeds. For a near-regular shape, the edge direction
    # of edge i is roughly a[i] + pi/2 + pi/n (turn angle pi/n).
    def regular_thetas():
        return a + np.pi / 2 + np.pi / n

    # Seed 1: regular (sanity), seed 2: regular + small perturbation.
    rng = np.random.default_rng(0)
    seeds = [regular_thetas()]
    for _ in range(3):
        seeds.append(regular_thetas() + rng.normal(0, 0.25, n))

    for t0 in seeds:
        if time.time() > _DEADLINE - _MARGIN:
            break
        try:
            pts, s = _optimize_from(t0)
        except Exception:
            continue
        if s > best_s:
            best = pts
            best_s = s

    # Polishing: a few more local runs from the best direction set if time remains.
    if best_s > 0 and time.time() < _DEADLINE - _MARGIN:
        # recover edge directions from best polygon (approx) for a re-run
        v = np.roll(best, -1, axis=0) - best
        th = np.arctan2(v[:, 1], v[:, 0])
        try:
            pts, s = _optimize_from(th)
            if s > best_s:
                best = pts
                best_s = s
        except Exception:
            pass

    return best


# EVOLVE-BLOCK-END
