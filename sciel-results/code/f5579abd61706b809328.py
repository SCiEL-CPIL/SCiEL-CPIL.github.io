"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time

N = 26

def _now():
    return time.time()


def _min_gap(x, y, r):
    """Minimum of (distance - r_i - r_j) over all pairs and (min(x,y) - r_i) to walls."""
    n = len(r)
    gap = np.full(n, np.inf)
    # wall gaps
    gap = np.minimum(gap, np.minimum(x, 1.0 - x) - r)
    gap = np.minimum(gap, np.minimum(y, 1.0 - y) - r)
    # pairwise
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    rr = r[:, None] + r[None, :]
    for i in range(n):
        for j in range(i + 1, n):
            g = d[i, j] - rr[i, j]
            if g < gap[i]:
                gap[i] = g
            if g < gap[j]:
                gap[j] = g
    return gap


def _soft_min_gap(x, y, r, lam):
    """Smooth surrogate of min gap via log-sum-exp on negative gaps."""
    gap = _min_gap(x, y, r)
    m = np.min(gap)
    # soft min: m - (1/lam) log(sum exp(lam*(gap - m))) ... use stable
    z = lam * (gap - m)
    s = np.sum(np.exp(z))
    return m - (1.0 / lam) * np.log(s)


def _gradient_soft_min(x, y, r, lam):
    """Gradient of soft-min gap w.r.t. x, y (r fixed)."""
    n = len(r)
    gap = _min_gap(x, y, r)
    m = np.min(gap)
    z = lam * (gap - m)
    w = np.exp(z)
    sw = np.sum(w)
    # d softmin / d gap_i = w_i / sw
    dsoft = w / sw
    gx = np.zeros(n)
    gy = np.zeros(n)
    # wall contributions: gap_i depends on min(x_i,1-x_i) etc.
    # For a given circle i, the wall gap is the min of 4 wall distances.
    # We need the subgradient direction of that min w.r.t. x_i, y_i.
    for i in range(n):
        wx = x[i]
        wxr = 1.0 - x[i]
        wy = y[i]
        wyr = 1.0 - y[i]
        # which wall is active for circle i (the min among its wall distances)
        # but gap[i] already includes pairwise; we need the wall subgrad of the min over walls
        # Determine active wall for the wall component:
        cands = [(wx, -1.0), (wxr, 1.0), (wy, -1.0), (wyr, 1.0)]
        # active = smallest distance; if tie, average
        mn = min(c[0] for c in cands)
        ax = 0.0
        ay = 0.0
        cnt = 0
        for dist, sgn in cands:
            if abs(dist - mn) < 1e-12:
                if dist == wx or dist == wxr:
                    ax += sgn
                else:
                    ay += sgn
                cnt += 1
        ax /= max(cnt, 1)
        ay /= max(cnt, 1)
        gx[i] += dsoft[i] * ax
        gy[i] += dsoft[i] * ay
    # pairwise contributions
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                continue
            # d softmin/d gap_i = dsoft[i]; gap_i = d - r_i - r_j
            # d gap_i / d x_i = dx/d ; / d x_j = -dx/d
            gx[i] += dsoft[i] * (dx / d)
            gx[j] += dsoft[j] * (-dx / d)
            gy[i] += dsoft[i] * (dy / d)
            gy[j] += dsoft[j] * (-dy / d)
    return gx, gy


def _local_search(x, y, r, deadline, lam0=8.0, lam1=60.0, iters=3000, lr0=0.02, lr1=0.002):
    """Maximize soft-min gap to push packing toward feasible (min gap >= 0)."""
    n = len(r)
    x = x.copy()
    y = y.copy()
    best_sum = np.sum(r)
    best_x, best_y = x.copy(), y.copy()
    for it in range(iters):
        if _now() > deadline - 3.0:
            break
        t = it / max(iters - 1, 1)
        lam = lam0 * (lam1 / lam0) ** t
        lr = lr0 * (lr1 / lr0) ** t
        gx, gy = _gradient_soft_min(x, y, r, lam)
        # normalize gradient
        gnorm = np.sqrt(np.sum(gx * gx) + np.sum(gy * gy))
        if gnorm > 1e-12:
            gx = gx / gnorm
            gy = gy / gnorm
        x += lr * gx
        y += lr * gy
        # clamp to box (keep centers within [r, 1-r] region loosely; allow slight)
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)
        gap = _min_gap(x, y, r)
        if np.min(gap) >= -1e-9:
            # feasible; try to record and continue to grow
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_x, best_y = x.copy(), y.copy()
    return best_x, best_y


def _is_valid(x, y, r, tol=1e-7):
    n = len(r)
    if np.min(r) < -tol:
        return False
    if np.min(x) < -tol or np.max(x) > 1.0 + tol:
        return False
    if np.min(y) < -tol or np.max(y) > 1.0 + tol:
        return False
    # wall clearance
    if np.min(np.minimum(x, 1.0 - x) - r) < -tol:
        return False
    if np.min(np.minimum(y, 1.0 - y) - r) < -tol:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _make_seed(kind, scale, rng):
    """Return (x, y, r) for a seed packing at given radius scale."""
    n = N
    if kind == 0:
        # hexagonal-ish grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        x = np.zeros(n); y = np.zeros(n); r = np.zeros(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) / cols + (0.0 if row % 2 == 0 else 0.5 / cols)
            y[i] = (row + 0.5) / rows
            r[i] = scale
        # keep within box
        x = np.clip(x, scale, 1.0 - scale)
        y = np.clip(y, scale, 1.0 - scale)
    elif kind == 1:
        # jittered square grid
        side = int(np.ceil(np.sqrt(n)))
        x = np.zeros(n); y = np.zeros(n); r = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side + (rng.random() - 0.5) * 0.3 / side
            y[i] = (row + 0.5) / side + (rng.random() - 0.5) * 0.3 / side
            r[i] = scale
        x = np.clip(x, scale, 1.0 - scale)
        y = np.clip(y, scale, 1.0 - scale)
    elif kind == 2:
        # corner + center bias
        x = np.zeros(n); y = np.zeros(n); r = np.zeros(n)
        for i in range(n):
            x[i] = rng.random()
            y[i] = rng.random()
            r[i] = scale
        x = np.clip(x, scale, 1.0 - scale)
        y = np.clip(y, scale, 1.0 - scale)
    else:
        # random
        x = rng.random(n); y = rng.random(n); r = np.full(n, scale)
        x = np.clip(x, scale, 1.0 - scale)
        y = np.clip(y, scale, 1.0 - scale)
    return x, y, r


def _try_pack(x0, y0, scale, deadline):
    """Grow from scale; do local search to reach feasible, then return sum of radii."""
    x, y = x0.copy(), y0.copy()
    r = np.full(N, scale)
    # local search to make feasible
    x, y = _local_search(x, y, r, deadline, lam0=6.0, lam1=40.0, iters=1500, lr0=0.03, lr1=0.004)
    if not _is_valid(x, y, r):
        return None
    return x, y, r, np.sum(r)


def construct():
    start = _now()
    deadline = start + 110.0  # safety margin before 120s
    rng = np.random.default_rng(12345)

    best = None
    best_sum = 0.0

    # Strategy: try a sequence of radius scales, from small to large,
    # with diverse seeds, keep the best feasible packing, and grow.
    # Start with a feasible base, then increase scale and re-pack.
    base_scale = 0.03
    # candidate scales to attempt (geometric growth)
    scales = []
    s = base_scale
    while s < 0.30:
        scales.append(s)
        s *= 1.15

    # First establish a feasible base at small scale
    for kind in range(3):
        if _now() > deadline - 20.0:
            break
        x0, y0, r0 = _make_seed(kind, base_scale, rng)
        res = _try_pack(x0, y0, base_scale, deadline)
        if res is not None:
            sx, sy, sr, ss = res
            if ss > best_sum:
                best_sum = ss
                best = (sx.copy(), sy.copy(), sr.copy())

    if best is None:
        # fallback: tiny circles
        x = np.linspace(0.05, 0.95, N)
        y = np.full(N, 0.5)
        r = np.full(N, 0.02)
        return np.column_stack([x, y, r])

    bx, by, br = best

    # Now grow: repeatedly increase scale, re-pack from current best positions
    cur_scale = base_scale
    for _ in range(40):
        if _now() > deadline - 25.0:
            break
        new_scale = cur_scale * 1.12
        if new_scale > 0.30:
            break
        improved = False
        for kind in range(4):
            if _now() > deadline - 25.0:
                break
            # seed from current best positions (slightly perturbed) or fresh
            if kind < 2:
                x0 = bx + (rng.random(N) - 0.5) * 0.02
                y0 = by + (rng.random(N) - 0.5) * 0.02
            else:
                x0, y0, _ = _make_seed(kind, new_scale, rng)
            r0 = np.full(N, new_scale)
            x0 = np.clip(x0, new_scale, 1.0 - new_scale)
            y0 = np.clip(y0, new_scale, 1.0 - new_scale)
            res = _try_pack(x0, y0, new_scale, deadline)
            if res is not None:
                sx, sy, sr, ss = res
                if ss > best_sum:
                    best_sum = ss
                    best = (sx.copy(), sy.copy(), sr.copy())
                    bx, by, br = sx, sy, sr
                    improved = True
                    cur_scale = new_scale
                    break
        if not improved:
            # can't grow further at this scale; try smaller increment
            new_scale = cur_scale * 1.05
            if new_scale <= cur_scale:
                break
            cur_scale = new_scale
            if cur_scale > 0.30:
                break

    # Final polish: tight local search on the best packing to maximize feasibility margin
    if best is not None and _now() < deadline - 8.0:
        bx, by, br = best
        x, y = _local_search(bx, by, br, deadline, lam0=20.0, lam1=120.0, iters=4000, lr0=0.01, lr1=0.001)
        if _is_valid(x, y, br):
            s = np.sum(br)
            if s > best_sum:
                best_sum = s
                best = (x.copy(), y.copy(), br.copy())

    bx, by, br = best
    # final validity check
    if not _is_valid(bx, by, br, tol=1e-6):
        # nudge inward slightly
        r2 = br * 0.999
        bx = np.clip(bx, r2, 1.0 - r2)
        by = np.clip(by, r2, 1.0 - r2)
        if not _is_valid(bx, by, r2, tol=1e-6):
            # last resort: shrink more
            r3 = br * 0.99
            bx = np.clip(bx, r3, 1.0 - r3)
            by = np.clip(by, r3, 1.0 - r3)
            return np.column_stack([bx, by, r3])
        return np.column_stack([bx, by, r2])

    return np.column_stack([bx, by, br])
