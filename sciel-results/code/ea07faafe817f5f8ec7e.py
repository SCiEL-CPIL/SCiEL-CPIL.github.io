"""Vectorized force-directed packing for 300 circles in a unit disk."""

import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def _pack_score(pts):
    """Compute packing radius: min of boundary and pairwise distances."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    r_bound = 1.0 - np.linalg.norm(pts, axis=1)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return float(min(np.min(r_bound), min_pair))


def _init_points(n, seed=0):
    """Hexagonal lattice initialization with centering."""
    rng = np.random.RandomState(seed)
    spacing = 2.0 / (np.sqrt(n) * 0.98)
    pts = []
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                pts.append([x, y])
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < n:
        pts = rng.uniform(-0.95, 0.95, size=(n, 2))
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 0.95
        pts[mask] *= 0.95 / norms[mask, np.newaxis]
        return pts
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind='stable')[:n]
    pts = pts[idx].copy()
    pts += rng.normal(0, spacing * 0.01, size=pts.shape)
    return pts


def _force_directed_relax(pts, deadline):
    """Vectorized force-directed optimization."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_score = _pack_score(pts)
    lr = 0.0005
    it = 0
    last_improve = time.time()
    no_improve = 0

    while time.time() < deadline - 15:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        np.fill_diagonal(dist, np.inf)

        # Repulsion: 1/dist^2
        rep = np.zeros_like(dist)
        mask = dist < 1e-9
        dist_safe = np.where(mask, 1.0, dist)
        rep = np.where(mask, 0.0, 1.0 / (dist_safe ** 2))
        np.fill_diagonal(rep, 0.0)

        # Attraction: -0.5 * dist
        att = -0.5 * dist
        np.fill_diagonal(att, 0.0)

        weights = rep + att

        # Forces from pairwise interactions
        forces = np.einsum('ij,jk->ik', weights, diff)

        # Boundary: push inward if too close to edge
        norms = np.linalg.norm(pts, axis=1)
        bound_margin = 0.02
        too_close = norms > (1.0 - bound_margin)
        if np.any(too_close):
            inward = pts[norms > 1e-9] / norms[norms > 1e-9, np.newaxis]
            bforce = np.zeros_like(pts)
            bforce[norms > 1e-9] = inward
            strength = np.where(too_close, (norms - (1.0 - bound_margin)) / bound_margin, 0.0)
            bforce *= strength[:, np.newaxis] * 2.0
            forces -= bforce

        # Update with adaptive step
        pts_new = pts + lr * forces
        norms_new = np.linalg.norm(pts_new, axis=1)
        mask_out = norms_new > 0.999
        if np.any(mask_out):
            pts_new[mask_out] *= 0.999 / norms_new[mask_out, np.newaxis]

        score = _pack_score(pts_new)
        if score > best_score:
            best_score = score
            best_pts = pts_new.copy()
            no_improve = 0
            last_improve = time.time()
        else:
            no_improve += 1
            if no_improve > 200:
                lr *= 0.7
                if lr < 1e-7:
                    break
            elif no_improve > 50:
                lr *= 0.95

        pts = pts_new
        it += 1

    return best_pts, best_score


def _local_search(pts, deadline):
    """Refine with small random perturbations around best point."""
    rng = np.random.RandomState(123)
    n = pts.shape[0]
    best_pts = pts.copy()
    best_score = _pack_score(pts)
    step = 0.005

    while time.time() < deadline - 10:
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            direction = rng.normal(0, 1, size=(n, 2))
            direction /= (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9)
            pts_try = best_pts + step * direction
            norms = np.linalg.norm(pts_try, axis=1)
            mask = norms > 0.999
            if np.any(mask):
                pts_try[mask] *= 0.999 / norms[mask, np.newaxis]
            s = _pack_score(pts_try)
            if s > best_score:
                best_score = s
                best_pts = pts_try.copy()
        step *= 0.85
        if step < 1e-5:
            break

    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    best_pts = None
    best_score = -np.inf

    for seed in range(8):
        if time.time() > deadline - 40:
            break

        pts = _init_points(N, seed)
        pts, score = _force_directed_relax(pts, deadline)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    if best_pts is not None and time.time() < deadline - 20:
        best_pts, best_score = _local_search(best_pts, deadline)

    if best_pts is None:
        best_pts = _init_points(N, 0)

    return best_pts
# EVOLVE-BLOCK-END
