"""Optimized packing of 300 equal circles in a unit circle."""
import numpy as np
import time

N = 300
TIME_LIMIT = 118.0  # evaluator kills at 150s; keep a safety margin


def _hex_init(n=N):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    # Best known r ~ 0.05315 -> spacing ~ 0.1063. Use a slightly finer grid.
    spacing = 0.095
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # fallback: enlarge grid
        return _hex_init_fallback(n)
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _hex_init_fallback(n=N):
    spacing = 0.08
    k = int(1.2 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    # pairwise distances
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _optimize(pts, start_time, time_limit):
    """Force-directed repulsion + boundary + centering optimizer. Returns best pts."""
    n = len(pts)
    pts = pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)

    # target spacing: aim for slightly larger than current achieved radius * 2
    # We'll use a target based on the known best ~ 0.05315 * 2
    target = 2.0 * 0.05315

    # Adaptive step size
    step = 0.004
    it = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        # --- compute forces ---
        # pairwise
        diff = pts[:, None, :] - pts[None, :, :]          # (n,n,2)
        d = np.linalg.norm(diff, axis=2)                  # (n,n)
        np.fill_diagonal(d, np.inf)

        # repulsion: only for pairs closer than target
        mask = d < target
        # strength scaled by overlap
        overlap = np.where(mask, target - d, 0.0)
        # unit vectors
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = diff / d[:, :, None]
        unit = np.where(np.isfinite(unit), unit, 0.0)

        force = np.sum(unit * overlap[:, :, None], axis=1)  # repel apart

        # boundary: push inward if outside (1 - r)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        over = np.maximum(0.0, rad - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            inward = -pts / rad[:, None]
        inward = np.where(np.isfinite(inward), inward, 0.0)
        force += inward * over[:, None] * 5.0  # strong boundary push

        # mild centering to keep packing tight
        force += -pts * 0.02

        # --- update ---
        # normalize force magnitude to keep step stable
        fmag = np.linalg.norm(force, axis=1)
        max_fm = np.max(fmag) if len(fmag) else 1.0
        if max_fm > 0:
            force = force * (step / max_fm)

        pts = pts + force
        # clamp inside boundary (safety)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None]

        it += 1
        # recompute score periodically (it's O(n^2)); every few iters
        if it % 20 == 0:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

        # adapt step: if score stalled, reduce
        if it % 200 == 0 and step > 0.0002:
            step *= 0.95

    # final: ensure best_pts is valid (inside, distinct)
    return best_pts, best_score


def construct():
    start_time = time.time()
    pts = _hex_init(N)
    # ensure valid start: clip inside
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    base_score = _score(pts)
    best_pts = pts.copy()
    best_score = base_score

    try:
        opt_pts, opt_score = _optimize(pts, start_time, TIME_LIMIT)
        if opt_score > best_score:
            best_pts = opt_pts
            best_score = opt_score
    except Exception:
        pass

    # final safety: ensure finite, inside, shape (N,2)
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        # rebuild
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]

    return best_pts
