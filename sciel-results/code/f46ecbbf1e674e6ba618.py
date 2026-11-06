"""Rewritten program: force-directed circle packing with multi-restart local search."""
import numpy as np
import time as _time

N = 50
_T0 = _time.time()
_BUDGET = 150.0 * 0.90  # safety margin before the 150s kill


def _clearances(pts):
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    pair_min = d.min()
    b = np.minimum(0.5 - np.abs(pts), axis=1)
    bound_min = b.min()
    return pair_min, bound_min


def _score(pts):
    pair_min, bound_min = _clearances(pts)
    return min(pair_min / 2.0, bound_min)


def _valid(pts):
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return False
    if np.any(pts[:, 0] > 0.5) or np.any(pts[:, 1] > 0.5):
        return False
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 1] < -0.5):
        return False
    return True


def _init_grid(rng):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
    return g + rng.uniform(-0.004, 0.004, size=g.shape)


def _init_hex(rng, rows):
    m = rows
    cols = int(np.ceil(N / m))
    xs = (np.arange(cols) + 0.5) / cols - 0.5
    pts = []
    for r in range(m):
        if len(pts) >= N:
            break
        off = (0.5 / cols) if (r % 2) else 0.0
        for c in range(cols):
            if len(pts) >= N:
                break
            pts.append([xs[c] + off, (r + 0.5) / m - 0.5])
    pts = np.array(pts[:N], dtype=float)
    return pts + rng.uniform(-0.004, 0.004, size=pts.shape)


def _force_step(pts, rng, step, r_target):
    """One force-directed step: repel close pairs and walls, mild centering."""
    f = np.zeros_like(pts)
    # pairwise repulsion: only for pairs closer than 2*r_target (they overlap at that radius)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    mask = d < 2.0 * r_target
    # force magnitude ~ (2r - d)/d along direction, capped
    for i in range(N):
        for j in range(i + 1, N):
            if mask[i, j]:
                v = diff[i, j]
                dd = d[i, j]
                if dd < 1e-9:
                    v = rng.normal(size=2)
                    dd = np.linalg.norm(v)
                v = v / dd
                mag = (2.0 * r_target - dd) / (2.0 * r_target)
                f[i] += v * mag
                f[j] -= v * mag
    # wall repulsion
    for i in range(N):
        p = pts[i]
        for axis in range(2):
            if p[axis] > 0.5 - r_target:
                f[i, axis] -= (p[axis] - (0.5 - r_target)) / r_target
            elif p[axis] < -0.5 + r_target:
                f[i, axis] += ((-0.5 + r_target) - p[axis]) / r_target
    # mild centering to keep things balanced
    f -= 0.001 * pts
    # noise
    f += rng.normal(0.0, step * 0.02, size=f.shape)
    pts = pts + step * f
    pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)
    return pts


def _optimize(pts, rng, max_iter=6000):
    pts = pts.copy()
    best = _score(pts)
    best_pts = pts.copy()
    # start with a small target radius and grow it
    r_target = max(_score(pts), 0.05)
    step = 0.05
    for it in range(max_iter):
        if _time.time() - _T0 > _BUDGET:
            break
        pts = _force_step(pts, rng, step, r_target)
        cur = _score(pts)
        if cur > best:
            best = cur
            best_pts = pts.copy()
            # if we improved, try to raise the target radius to push further
            r_target = min(best + 0.002, 0.08)
        else:
            # no improvement; slowly cool
            step *= 0.999
            # occasionally nudge target up to escape
            if it % 200 == 0:
                r_target = min(r_target + 0.001, 0.08)
        if step < 1e-4:
            break
    return best_pts, best


def construct():
    rng = np.random.default_rng(20240501)
    best_pts = None
    best_val = -1.0
    inits = [_init_grid(rng)]
    for rows in [5, 6, 7, 8]:
        inits.append(_init_hex(rng, rows=rows))
    inits.append(rng.uniform(-0.45, 0.45, size=(N, 2)))

    for init in inits:
        if _time.time() - _T0 > _BUDGET:
            break
        for rep in range(2):
            if _time.time() - _T0 > _BUDGET:
                break
            r = np.random.default_rng(777 + rep * 13 + len(inits))
            pts, val = _optimize(init, r, max_iter=5000)
            if val > best_val:
                best_val = val
                best_pts = pts.copy()
    if best_pts is None:
        best_pts = _init_grid(rng)
    if not _valid(best_pts):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
    return best_pts
