"""Circle packing in unit square: force-directed relaxation + radius growth."""
import time
import numpy as np

N = 26


def _valid(circles):
    x = circles[:, 0]; y = circles[:, 1]; r = circles[:, 2]
    if np.any(r <= 0) or np.any(r > 0.5 + 1e-9):
        return False
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9) or np.any(y < -1e-9) or np.any(y > 1 + 1e-9):
        return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-9:
                return False
    return True


def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))


def _relax(pos, r, iters=6000):
    n = len(r)
    for it in range(iters):
        for i in range(n):
            fx = 0.0; fy = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.hypot(dx, dy)
                gap = r[i] + r[j]
                if d < 1e-9:
                    fx += 0.01 * (it % 2)
                    fy += 0.01 * ((it // 2) % 2)
                    continue
                if d < gap:
                    f = 0.5 * (gap - d)
                    fx += f * dx / d
                    fy += f * dy / d
            # wall repulsion
            if pos[i, 0] < r[i]:
                fx += 0.5 * (r[i] - pos[i, 0])
            if pos[i, 0] > 1 - r[i]:
                fx -= 0.5 * (pos[i, 0] - (1 - r[i]))
            if pos[i, 1] < r[i]:
                fy += 0.5 * (r[i] - pos[i, 1])
            if pos[i, 1] > 1 - r[i]:
                fy -= 0.5 * (pos[i, 1] - (1 - r[i]))
            # small damping toward center to keep things in view
            fx += 0.001 * (0.5 - pos[i, 0])
            fy += 0.001 * (0.5 - pos[i, 1])
            # clamp step
            step = 0.15
            pos[i, 0] += step * fx
            pos[i, 1] += step * fy
            # hard clamp to square (with margin)
            pos[i, 0] = min(1 - 1e-6, max(1e-6, pos[i, 0]))
            pos[i, 1] = min(1 - 1e-6, max(1e-6, pos[i, 1]))
    return pos


def _grow(pos, r, iters=20000):
    n = len(r)
    for it in range(iters):
        # try to grow each circle a little, then relax
        for i in range(n):
            # candidate growth
            dr = 0.002
            r[i] += dr
            # check feasibility roughly by relaxing a few steps
        # do a relaxation pass to resolve overlaps introduced
        for i in range(n):
            fx = 0.0; fy = 0.0
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.hypot(dx, dy)
                gap = r[i] + r[j]
                if d < 1e-9:
                    continue
                if d < gap:
                    f = 0.5 * (gap - d)
                    fx += f * dx / d
                    fy += f * dy / d
            if pos[i, 0] < r[i]:
                fx += 0.5 * (r[i] - pos[i, 0])
            if pos[i, 0] > 1 - r[i]:
                fx -= 0.5 * (pos[i, 0] - (1 - r[i]))
            if pos[i, 1] < r[i]:
                fy += 0.5 * (r[i] - pos[i, 1])
            if pos[i, 1] > 1 - r[i]:
                fy -= 0.5 * (pos[i, 1] - (1 - r[i]))
            pos[i, 0] += 0.1 * fx
            pos[i, 1] += 0.1 * fy
            pos[i, 0] = min(1 - 1e-6, max(1e-6, pos[i, 0]))
            pos[i, 1] = min(1 - 1e-6, max(1e-6, pos[i, 1]))
    return pos, r


def _layout_grid():
    # 5x5 grid, use 25, plus 1 extra
    side = 5
    pos = np.zeros((N, 2)); r = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            pos[idx, 0] = (i + 0.5) / side
            pos[idx, 1] = (j + 0.5) / side
            r[idx] = 0.5 / side - 0.02
            idx += 1
    # last circle in a gap
    pos[idx, 0] = 0.5
    pos[idx, 1] = 0.5
    r[idx] = 0.05
    return pos, r


def _layout_honeycomb():
    # staggered rows to pack more densely
    rows = 5
    pos = np.zeros((N, 2)); r = np.zeros(N)
    idx = 0
    for row in range(rows):
        y = (row + 0.5) / rows
        if row % 2 == 0:
            xs = [0.1, 0.3, 0.5, 0.7, 0.9]
        else:
            xs = [0.2, 0.4, 0.6, 0.8]
        for x in xs:
            if idx >= N:
                break
            pos[idx, 0] = x
            pos[idx, 1] = y
            r[idx] = 0.09
            idx += 1
    while idx < N:
        pos[idx, 0] = 0.5 + 0.1 * (idx % 3)
        pos[idx, 1] = 0.5 + 0.1 * ((idx // 3) % 3)
        r[idx] = 0.06
        idx += 1
    return pos, r


def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin before 150s kill
    best = None
    best_val = -1.0

    # candidate initial layouts
    layouts = [
        _layout_grid(),
        _layout_honeycomb(),
    ]
    # add a few randomized jittered grids
    rng = np.random.default_rng(12345)
    for k in range(4):
        p, rr = _layout_grid()
        p = p + rng.normal(0, 0.01, p.shape)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        layouts.append((p, rr.copy()))

    for pos, r in layouts:
        if time.time() > deadline:
            break
        pos2, r2 = _grow(pos.copy(), r.copy(), iters=15000)
        pos2, r2 = _relax(pos2, r2, iters=3000)
        # final grow
        pos2, r2 = _grow(pos2, r2, iters=10000)
        # enforce validity: shrink any overlapping pairs
        for _ in range(200):
            ok = True
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.hypot(pos2[i, 0] - pos2[j, 0], pos2[i, 1] - pos2[j, 1])
                    need = r2[i] + r2[j]
                    if d < need - 1e-9:
                        # shrink both a bit
                        shrink = (need - d) / 2 + 1e-5
                        r2[i] = max(1e-4, r2[i] - shrink)
                        r2[j] = max(1e-4, r2[j] - shrink)
                        ok = False
            if ok:
                break
        # wall fix
        for i in range(N):
            pos2[i, 0] = min(1 - r2[i], max(r2[i], pos2[i, 0]))
            pos2[i, 1] = min(1 - r2[i], max(r2[i], pos2[i, 1]))
        circles = np.column_stack([pos2, r2])
        if _valid(circles):
            val = _sum_radii(circles)
            if val > best_val:
                best_val = val
                best = circles

    if best is None:
        # fallback: trivial valid grid
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    return best
