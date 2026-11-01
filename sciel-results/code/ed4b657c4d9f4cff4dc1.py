"""Improved packer: hex-lattice seed + vectorized simulated annealing with time guard."""
import numpy as np
import time
from scipy.spatial import cKDTree

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
    # minimum pairwise half-distance using KD-tree for efficiency
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * np.min(dists[:, 1])
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _grad_polish(points, t_budget, rng):
    """Gradient-based polish: push overlapping pairs apart and boundary-violating points inward."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r

    d0 = _min_dist(cur)
    step = 0.005 * d0
    min_step = 1e-8 * d0

    while time.time() - start < t_budget:
        # Find the most violated constraint
        norms = np.linalg.norm(cur, axis=1)
        boundary_margin = 1.0 - norms

        tree = cKDTree(cur)
        dists, idxs = tree.query(cur, k=2)
        min_neighbor_dist = dists[:, 1]
        min_neighbor_idx = idxs[:, 1]

        # Identify worst pair (smallest distance)
        worst_pair_idx = np.argmin(min_neighbor_dist)
        i = worst_pair_idx
        j = min_neighbor_idx[i]
        d_ij = min_neighbor_dist[i]

        # Also check boundary violations
        worst_boundary_idx = np.argmin(boundary_margin)
        boundary_violation = -boundary_margin[worst_boundary_idx]

        # Decide which constraint to fix
        if d_ij < 2 * best_r and d_ij < 2 * (1.0 - norms[i]) and d_ij < 2 * (1.0 - norms[j]):
            # Fix pair overlap: push i and j apart
            diff = cur[i] - cur[j]
            n = np.linalg.norm(diff)
            if n < 1e-12:
                # Random direction
                ang = rng.random() * 2.0 * np.pi
                diff = np.array([np.cos(ang), np.sin(ang)])
                n = 1.0
            unit = diff / n
            push = 0.5 * step
            cur[i] += push * unit
            cur[j] -= push * unit
        elif boundary_margin[worst_boundary_idx] < 0.0:
            # Push point inward
            idx = worst_boundary_idx
            nrm = norms[idx]
            if nrm < 1e-12:
                continue
            cur[idx] -= (step / nrm) * cur[idx]
        else:
            # No clear violation; do a small random perturbation on tight points
            tight_mask = (boundary_margin < 0.02 * d0) | (min_neighbor_dist < 0.02 * d0)
            tight_idx = np.where(tight_mask)[0]
            if len(tight_idx) == 0:
                idx = rng.integers(N)
            else:
                idx = tight_idx[rng.integers(len(tight_idx))]
            ang = rng.random() * 2.0 * np.pi
            dx = step * np.cos(ang)
            dy = step * np.sin(ang)
            cur[idx][0] += dx
            cur[idx][1] += dy

        # Check if still inside container
        new_norms = np.linalg.norm(cur, axis=1)
        if np.any(new_norms > 1.0 - 1e-12):
            # Revert
            cur = best.copy()
            cur_r = best_r
            step *= 0.95
            continue

        new_r = _score(cur)
        if new_r >= cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            step *= 1.01
        else:
            cur = best.copy()
            cur_r = best_r
            step *= 0.97

        if step < min_step:
            step = min_step
        if step > 0.1 * d0:
            step = 0.1 * d0

    return best, best_r


def _min_dist(points):
    """Efficiently compute the minimum pairwise distance."""
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(np.min(dists[:, 1]))


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
    d0 = _min_dist(cur)
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

        old_pt = cur[idx].copy()
        cur[idx] = new_pt
        new_r = _score(cur)

        if new_r >= cur_r or rng.random() < np.exp((new_r - cur_r) / max(T, 1e-12)):
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
        else:
            cur[idx] = old_pt

        T *= 0.999
        step *= 0.9999

    return best, best_r


def _polish(points, t_budget, rng):
    """High-precision polish pass: tighter tolerance, more iterations, local solve."""
    start = time.time()
    best = points.copy()
    best_r = _score(best)
    cur = points.copy()
    cur_r = best_r

    # Smaller step size for polishing
    d0 = _min_dist(cur)
    step = 0.01 * d0
    min_step = 1e-6 * d0

    while time.time() - start < t_budget:
        # Focus on points that are "tight" (close to boundary or neighbors)
        # Compute local constraints for each point
        norms = np.linalg.norm(cur, axis=1)
        boundary_margin = 1.0 - norms
        tree = cKDTree(cur)
        dists, _ = tree.query(cur, k=2)
        min_neighbor = dists[:, 1]

        # Points with small margin are candidates for improvement
        tight_mask = (boundary_margin < 0.01 * d0) | (min_neighbor < 0.01 * d0)
        tight_idx = np.where(tight_mask)[0]

        if len(tight_idx) == 0:
            # No tight points, do a random move
            idx = rng.integers(N)
        else:
            idx = tight_idx[rng.integers(len(tight_idx))]

        ang = rng.random() * 2.0 * np.pi
        mag = rng.random() * step
        dx = mag * np.cos(ang)
        dy = mag * np.sin(ang)

        new_pt = cur[idx].copy()
        new_pt[0] += dx
        new_pt[1] += dy

        if np.hypot(new_pt[0], new_pt[1]) > 1.0 - 1e-9:
            continue

        old_pt = cur[idx].copy()
        cur[idx] = new_pt
        new_r = _score(cur)

        if new_r >= cur_r:
            cur_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = cur.copy()
            step *= 0.999
        else:
            cur[idx] = old_pt
            step *= 1.01

        if step < min_step:
            step = min_step
        if step > 0.1 * d0:
            step = 0.1 * d0

    return best, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # initial hex-lattice seed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    best = pts.copy()
    best_r = _score(best)

    # Phase 1: Multi-start annealing within the time budget
    while time.time() - t0 < DEADLINE - 30.0:  # reserve 30s for polishing
        # small perturbation of the seed for diversity
        jitter = 0.01 * rng.standard_normal((N, 2))
        cand = best + jitter
        # clip inside container
        rad = np.linalg.norm(cand, axis=1)
        mask = rad > 1.0
        if np.any(mask):
            cand[mask] = cand[mask] / rad[mask, None] * (1.0 - 1e-6)

        t_left = DEADLINE - 30.0 - (time.time() - t0)
        if t_left < 5.0:
            break
        t_alloc = min(t_left - 2.0, 15.0)  # cap per-anneal budget
        cand, r = _anneal(cand, t_alloc, rng)
        if r > best_r:
            best_r = r
            best = cand

    # Phase 2: High-precision polish pass
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 10.0:
        best, best_r = _polish(best, t_left - 5.0, rng)

    # Phase 3: Gradient-based final polish
    t_left = DEADLINE - (time.time() - t0)
    if t_left > 3.0:
        best, best_r = _grad_polish(best, t_left - 1.0, rng)

    # final safety: ensure valid output
    final_r = _score(best)
    if final_r <= 0.0:
        # fall back to a guaranteed-valid lattice
        best = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))

    return best
