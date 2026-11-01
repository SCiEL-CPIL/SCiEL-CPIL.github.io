"""Structural ring-init + repulsion relaxation for packing 50 circles in a unit disk."""
import numpy as np
import time

N = 50
DEADLINE = 150.0
MARGIN = 8.0  # safety margin seconds before wall-clock kill


def _score(pts):
    """Return the achievable common radius for a set of centres."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # container constraint
    dist = np.linalg.norm(pts, axis=1)
    if np.any(dist > 1.0):
        return 0.0
    r_cont = 1.0 - dist.min()
    # pairwise
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = d.min() / 2.0
    return min(r_pair, r_cont)


def _ring_init(rings):
    """Place circles on concentric rings with staggered angles.

    rings: list of (count, radius_fraction) where radius_fraction is fraction of (1 - r)
    Actually we use absolute-ish radii scaled; we'll refine. Returns (N,2).
    """
    pts = []
    # center circle
    pts.append([0.0, 0.0])
    for (count, frac) in rings:
        r = frac  # absolute radius for this ring
        for k in range(count):
            ang = 2.0 * np.pi * k / count + np.pi / count  # stagger
            pts.append([r * np.cos(ang), r * np.sin(ang)])
    pts = np.array(pts, dtype=float)
    if len(pts) < N:
        # pad with random near edge
        extra = np.random.randn(N - len(pts), 2) * 0.1 + np.array([0.5, 0.5])
        pts = np.vstack([pts, extra])
    elif len(pts) > N:
        pts = pts[:N]
    return pts


def _repulsion_relax(pts, steps=4000, dt_init=0.02, dt_end=0.0005,
                     repel=1.0, wall=4.0, deadline=None):
    """Iterative repulsion + wall constraint relaxation.

    Each point is pushed away from neighbours (inverse-square-ish) and pulled
    inward if it exceeds the unit disk. dt decays over time.
    """
    pts = pts.copy()
    n = pts.shape[0]
    for it in range(steps):
        if deadline is not None and (time.time() - deadline[0]) > (DEADLINE - MARGIN):
            break
        # compute pairwise vectors
        diff = pts[:, None, :] - pts[None, :, :]  # (n,n,2)
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, 1.0)
        dist = np.sqrt(dist2)
        # repulsion force: stronger when close
        # f = repel * diff / dist^3  (so magnitude ~ 1/dist^2)
        with np.errstate(divide='ignore', invalid='ignore'):
            fmag = repel / (dist ** 3)
        np.fill_diagonal(fmag, 0.0)
        force = fmag[:, :, None] * diff
        F = np.sum(force, axis=1)
        # wall force: if outside, pull toward origin proportional to excess
        r = np.linalg.norm(pts, axis=1)
        excess = np.clip(r - 1.0, 0.0, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            wall_dir = np.where(r > 1e-9, pts / r[:, None], 0.0)
        F -= wall * excess[:, None] * wall_dir
        # decay dt
        frac = it / max(steps - 1, 1)
        dt = dt_init * (dt_end / dt_init) ** frac
        pts += dt * F
        # hard project inside disk to keep finite & valid
        r = np.linalg.norm(pts, axis=1)
        too = r > 1.0
        if np.any(too):
            pts[too] *= (1.0 - 1e-6) / r[too, None]
    return pts


def construct():
    deadline = [time.time()]
    best = None
    best_score = -1.0

    # Candidate ring structures for N=50 (center + rings).
    # Known best has roughly: 1 center, then rings of increasing count.
    ring_configs = [
        [(12, 0.30), (13, 0.55), (14, 0.78)],
        [(11, 0.32), (12, 0.56), (14, 0.80)],
        [(10, 0.30), (12, 0.52), (13, 0.72), (13, 0.90)],
        [(12, 0.28), (14, 0.52), (14, 0.76)],
        [(9, 0.30), (12, 0.50), (13, 0.70), (14, 0.90)],
        [(13, 0.34), (13, 0.60), (13, 0.84)],
        [(12, 0.30), (13, 0.58), (13, 0.82)],
    ]

    rng = np.random.default_rng(12345)

    for cfg in ring_configs:
        if (time.time() - deadline[0]) > (DEADLINE - MARGIN - 2.0):
            break
        base = _ring_init(cfg)
        # add small random jitter to break symmetry
        jittered = base + rng.normal(0, 0.01, base.shape)
        relaxed = _repulsion_relax(jittered, steps=3000, dt_init=0.015,
                                   dt_end=0.0004, repel=1.0, wall=6.0,
                                   deadline=deadline)
        sc = _score(relaxed)
        if sc > best_score:
            best_score = sc
            best = relaxed

    # A few randomized restarts from ring-ish configs to escape local minima
    attempts = 0
    while (time.time() - deadline[0]) < (DEADLINE - MARGIN - 1.0) and attempts < 40:
        attempts += 1
        cfg = ring_configs[int(rng.integers(0, len(ring_configs)))]
        base = _ring_init(cfg)
        jitter = rng.normal(0, rng.uniform(0.02, 0.10), base.shape)
        cand = base + jitter
        cand = _repulsion_relax(cand, steps=2500, dt_init=0.012,
                                dt_end=0.0003, repel=1.0, wall=6.0,
                                deadline=deadline)
        sc = _score(cand)
        if sc > best_score:
            best_score = sc
            best = cand

    if best is None:
        # fallback: simple ring
        best = _ring_init([(12, 0.30), (13, 0.55), (14, 0.78)])

    # Final safety: ensure inside disk
    r = np.linalg.norm(best, axis=1)
    too = r > 1.0
    if np.any(too):
        best[too] *= (1.0 - 1e-6) / r[too, None]

    return best
