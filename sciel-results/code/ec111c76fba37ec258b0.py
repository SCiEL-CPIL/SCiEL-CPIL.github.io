"""Circle packing in a perimeter-4 rectangle.
Design: aspect-ratio sweep + grid seed + fast vectorized repulsive-force
relaxation (no collisions) to push the 21 circles apart and maximize sum(r).
"""
import time
import numpy as np

N = 21

# deterministic RNG seed
_SEED = 123456789


def _grid_positions(n, W, H):
    """Grid-like seed positions inside (0,W)x(0,H)."""
    cols = int(np.ceil(np.sqrt(n * W / max(H, 1e-9))))
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    xs = (np.arange(cols) + 0.5) * (W / cols)
    ys = (np.arange(rows) + 0.5) * (H / rows)
    X, Y = np.meshgrid(xs, ys)
    pos = np.column_stack([X.ravel(), Y.ravel()])[:n]
    return pos


def _relax(pos, W, H, iters, base, dt):
    """Push circles apart via repulsive forces; return final positions.

    Forces:
      - repulsion between close circles (drives them to touch)
      - soft wall pushback (keeps circles inside)
      - tiny centering (prevents drift)
    """
    pos = pos.astype(float).copy()
    n = pos.shape[0]
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]          # (n,n,2)
        dist = np.linalg.norm(d, axis=2)               # (n,n)
        np.fill_diagonal(dist, np.inf)
        close = dist < 2 * base
        # repulsive force magnitude ~ 1/(dist+eps) on close pairs
        fmag = np.zeros_like(dist)
        fmag[close] = 1.0 / (dist[close] + 1e-4)
        fmag[~np.isfinite(fmag)] = 0.0
        F = (d * fmag[:, :, None]).sum(axis=1)         # (n,2)

        # wall pushback
        px, py = pos[:, 0], pos[:, 1]
        F[:, 0] += np.where(px < base, (base - px) / base, 0.0) * 1.5
        F[:, 0] += np.where(px > W - base, (px - (W - base)) / base, 0.0) * 1.5
        F[:, 1] += np.where(py < base, (base - py) / base, 0.0) * 1.5
        F[:, 1] += np.where(py > H - base, (py - (H - base)) / base, 0.0) * 1.5

        # tiny centering
        F -= 1e-4 * pos

        pos += dt * F
        # clamp to stay in bounds (safety)
        pos[:, 0] = np.clip(pos[:, 0], 1e-6, W - 1e-6)
        pos[:, 1] = np.clip(pos[:, 1], 1e-6, H - 1e-6)
    return pos


def _value_for_aspect(W, H, deadline):
    """Try to maximize sum(r) for a fixed rectangle W x H (W+H=2)."""
    t0 = time.time()
    # initial grid seed
    pos = _grid_positions(N, W, H)
    # base radius estimate: area-based
    base = 0.5 * np.sqrt(W * H / N)

    best_pos = pos.copy()
    # staged relaxation: start with larger dt, shrink it
    stages = [(400, 0.5), (600, 0.25), (800, 0.1), (1000, 0.05), (1200, 0.02)]
    for iters, dt in stages:
        pos = _relax(pos, W, H, iters, base, dt)
        if time.time() > deadline:
            break
        best_pos = pos

    # compute radii from nearest neighbor and walls
    n = best_pos.shape[0]
    d = best_pos[:, None, :] - best_pos[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    wall = np.minimum(np.minimum(best_pos[:, 0], W - best_pos[:, 0]),
                      np.minimum(best_pos[:, 1], H - best_pos[:, 1]))
    r = 0.5 * np.minimum(nn, wall)
    r = np.maximum(r, 1e-6)
    return r.sum(), best_pos, r


def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # best over aspect ratios
    best_val = -1.0
    best_pos = None
    best_r = None
    best_W = best_H = None

    # candidate aspect ratios W/H (W+H=2 => W = 2*ratio/(1+ratio))
    ratios = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 0.65, 0.75,
              1.35, 1.6, 1.8, 2.2]
    # sort by "squareness" first (likely best), but keep variety
    ratios = sorted(ratios, key=lambda r: abs(np.log(r)))

    for ratio in ratios:
        W = 2.0 * ratio / (1.0 + ratio)
        H = 2.0 - W
        val, pos, r = _value_for_aspect(W, H, deadline)
        if val > best_val:
            best_val = val
            best_pos = pos
            best_r = r
            best_W, best_H = W, H
        if time.time() > deadline - 5.0:
            break

    if best_pos is None:
        # fallback: simple grid in a 1x2 rectangle
        pos = _grid_positions(N, 1.0, 1.0)
        return np.column_stack([pos, np.full(N, 0.04)])

    # final safety: clip radii so interiors are disjoint and inside
    # recompute radii conservatively
    n = best_pos.shape[0]
    d = best_pos[:, None, :] - best_pos[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1)
    wall = np.minimum(np.minimum(best_pos[:, 0], best_W - best_pos[:, 0]),
                      np.minimum(best_pos[:, 1], best_H - best_pos[:, 1]))
    r = 0.5 * np.minimum(nn, wall)
    r = np.clip(r, 1e-6, None)

    out = np.column_stack([best_pos, r])
    return out
