"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(pts):
    # pts: (N,2)
    n = len(pts)
    # pairwise distances
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    r_inside = 1.0 - np.sqrt((pts ** 2).sum(1)).max()
    return min(min_pair / 2.0, r_inside)


def _hex_init(n):
    # Generate a hexagonal lattice of points, keep those inside unit disk,
    # then pick n of them (closest to center first) to form a dense seed.
    # We want roughly n points inside the disk.
    # Try a few spacings until we have at least n points inside.
    for s in np.linspace(0.04, 0.20, 200):
        # build lattice
        h = s * np.sqrt(3.0) / 2.0
        # extent
        R = 1.0
        nx = int(2 * R / s) + 3
        ny = int(2 * R / h) + 3
        xs = (np.arange(nx) - nx // 2) * s
        ys = (np.arange(ny) - ny // 2) * h
        X, Y = np.meshgrid(xs, ys)
        # offset odd rows
        Yoff = Y + (np.arange(ny)[:, None] % 2) * (s / 2.0)
        R2 = X ** 2 + Yoff ** 2
        mask = R2 <= (1.0 - s / 2.0) ** 2
        pts = np.stack([X[mask], Yoff[mask]], axis=1)
        if len(pts) >= n:
            # pick n closest to origin
            r2 = (pts ** 2).sum(1)
            idx = np.argsort(r2, kind="stable")[:n]
            return pts[idx]
    # fallback: random
    ang = np.random.rand(n) * 2 * np.pi
    rad = np.sqrt(np.random.rand(n)) * 0.9
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)


def _relax(pts, iters, dt0, dt1, deadline):
    n = len(pts)
    pts = pts.astype(np.float64).copy()
    # adaptive repulsion radius
    # initial guess of spacing
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    spacing = dist.min()
    R = max(spacing * 1.5, 0.05)
    for it in range(iters):
        if time.time() > deadline:
            break
        dt = dt0 * (dt1 / dt0) ** (it / max(iters, 1))
        # pairwise
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1))
        # avoid divide by zero
        dist_safe = np.where(dist < 1e-12, 1e-12, dist)
        # repulsion: strong when close, decays with distance; only within cutoff R
        # force magnitude ~ (1 - dist/R) when dist < R else 0
        w = np.where(dist < R, (1.0 - dist / R) / dist_safe, 0.0)
        F = (d * w[:, :, None]).sum(0)
        # containment: push inward if outside
        r = np.sqrt((pts ** 2).sum(1))
        F -= pts * (r[:, None] / r_safe[:, None]) * 0.0  # placeholder
        r_safe = np.where(r < 1e-12, 1e-12, r)
        F -= pts * (r / r_safe)[:, None] * 0.0
        # simpler containment: if r > 0.98, pull toward center
        pull = np.where(r > 0.95, (r - 0.95)[:, None], 0.0) * pts / r_safe[:, None]
        F -= pull
        pts += dt * F
        # clamp inside
        r = np.sqrt((pts ** 2).sum(1))
        mask = r > 0.999
        if np.any(mask):
            pts[mask] *= 0.999 / r[mask][:, None]
    return pts


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 150s kill
    np.random.seed(0)

    best = None
    best_score = -1.0

    # Initial hex seed
    pts = _hex_init(N)
    s = _score(pts)
    if s > best_score:
        best_score = s
        best = pts.copy()

    # Relaxation schedule: several passes with decreasing step
    passes = [
        (400, 0.02, 0.002),
        (400, 0.01, 0.001),
        (600, 0.005, 0.0005),
        (600, 0.002, 0.0002),
        (800, 0.001, 0.0001),
    ]
    for iters, dt0, dt1 in passes:
        if time.time() > deadline:
            break
        pts = _relax(pts, iters, dt0, dt1, deadline)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best = pts.copy()

    # Local improvement: a few random restarts of the relaxation from perturbed best
    if time.time() < deadline - 20.0:
        for _ in range(5):
            if time.time() > deadline:
                break
            p = best.copy()
            p += np.random.randn(*p.shape) * 0.005
            # clamp
            r = np.sqrt((p ** 2).sum(1))
            mask = r > 0.999
            p[mask] *= 0.999 / r[mask][:, None]
            p = _relax(p, 600, 0.005, 0.0005, deadline)
            s = _score(p)
            if s > best_score:
                best_score = s
                best = p.copy()

    return best
# EVOLVE-BLOCK-END
