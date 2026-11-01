"""Improved packer: hex-lattice seed + vectorized simulated annealing with time guard."""
import numpy as np
import time

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 110.0  # safety margin before the 120 s wall-clock limit


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    # minimum pairwise half-distance
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _pairwise_min(points):
    """Vectorized minimum pairwise distance (full matrix)."""
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            # prefer inner points (smaller radius) to maximise margin
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _anneal(points, t_budget, rng):
    """Simulated annealing on the hex-lattice seed. Returns best points found."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r

    # initial temperature from typical pair distance
    d0 = _pairwise_min(cur)
    T = 0.02 * d0
    T_min = 1e-5 * d0
    step = 0.05 * d0

    while time.time() - start < t_budget and T > T_min:
        # propose a random move on a random point
        idx = rng.integers(N)
        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        # reject if it leaves the container (with margin)
        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            T *= 0.999
            continue

        cur[idx] = new_pt
        new_r = _score(cur)

        if new_r >= cur_r or rng.random() < np.exp((new_r - cur_r) / max(T, 1e-12)):
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
        else:
            cur[idx][0] -= dx
            cur[idx][1] -= dy

        T *= 0.999
        step *= 0.9999

    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # multi-start annealing within the time budget
    while time.time() - t0 < DEADLINE:
        # small perturbation of the seed for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 15.0)  # cap per-anneal budget
        cand, r = _anneal(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # final safety: ensure valid output
    final_r = _score(best)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
