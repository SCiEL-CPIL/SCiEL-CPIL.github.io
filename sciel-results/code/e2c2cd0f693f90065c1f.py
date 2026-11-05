"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def _score(pts):
    """Return (value, valid). value = number of lines with exactly 4 points.
    valid = True iff all 16 points are distinct, in bounds, and no 5 collinear."""
    if len(pts) != 16:
        return 0, False
    if len(set(tuple(p) for p in pts)) != 16:
        return 0, False
    for p in pts:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return 0, False
    lines = defaultdict(list)
    for i in range(16):
        for j in range(i + 1, 16):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = abs(dx)
            h = abs(dy)
            while h:
                g, h = h, g % h
            if g:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            lines[(dx, dy)].append((i, j))
    val = 0
    for k, pairs in lines.items():
        cnt = defaultdict(int)
        for i, j in pairs:
            cnt[i] += 1
            cnt[j] += 1
        mx = max(cnt.values())
        if mx >= 5:
            return 0, False
        if mx == 4:
            val += 1
    return val, True


def _random_grid(rnd):
    """Random 4x4 grid: distinct x and y positions -> 8 guaranteed 4-point lines."""
    xs = rnd.sample(range(-1000, 1001), 4)
    ys = rnd.sample(range(-1000, 1001), 4)
    pts = [[x, y] for x in xs for y in ys]
    return pts


def _grid_score(pts):
    return _score(pts)[0]


def _refine_grid(pts, rnd, budget):
    """Local search: perturb individual points, keep if score improves and still valid."""
    cur = [list(p) for p in pts]
    curv, _ = _score(cur)
    best = [list(p) for p in cur]
    bestv = curv
    t0 = time.time()
    while time.time() - t0 < budget:
        p = rnd.randrange(16)
        old = list(cur[p])
        cur[p][0] = old[0] + rnd.randint(-40, 40)
        cur[p][1] = old[1] + rnd.randint(-40, 40)
        v, ok = _score(cur)
        if ok and v > curv:
            curv = v
            if v > bestv:
                bestv = v
                best = [list(q) for q in cur]
        else:
            cur[p] = old
    return best, bestv


def _random_walk(pts, rnd, budget):
    """Annealing-like walk allowing moves that keep the no-5-collinear constraint,
    accepting some score decreases to escape local optima."""
    cur = [list(p) for p in pts]
    curv, _ = _score(cur)
    best = [list(p) for p in cur]
    bestv = curv
    t0 = time.time()
    while time.time() - t0 < budget:
        p = rnd.randrange(16)
        old = list(cur[p])
        cur[p][0] = old[0] + rnd.randint(-60, 60)
        cur[p][1] = old[1] + rnd.randint(-60, 60)
        v, ok = _score(cur)
        if ok:
            # accept improving moves, and occasionally accept neutral/worse to escape
            if v >= curv or rnd.random() < 0.15:
                curv = v
                if v > bestv:
                    bestv = v
                    best = [list(q) for q in cur]
            else:
                cur[p] = old
        else:
            cur[p] = old
    return best, bestv


def construct():
    rnd = random.Random(12345)
    deadline = time.time() + 140.0
    best_pts = None
    best_val = 0

    # Phase 1: sample random grids, refine the good ones with local search.
    while time.time() < deadline - 20.0:
        pts = _random_grid(rnd)
        v, ok = _score(pts)
        if not ok:
            continue
        # cheap pre-filter: only refine promising grids
        if v >= best_val or v >= 8:
            pts2, v2 = _refine_grid(pts, rnd, 0.5)
            if v2 > best_val:
                best_val = v2
                best_pts = [list(p) for p in pts2]
        # occasional deeper walk from a good grid
        if best_pts is not None and v2 >= 10 and time.time() < deadline - 20.0:
            pts3, v3 = _random_walk(pts2, rnd, 1.0)
            if v3 > best_val:
                best_val = v3
                best_pts = [list(p) for p in pts3]

    if best_pts is None:
        # fallback: the 4x4 grid
        best_pts = [[x, y] for x in range(4) for y in range(4)]
        best_val, _ = _score(best_pts)

    # Phase 2: use remaining time to polish the best found configuration.
    if time.time() < deadline - 5.0:
        pts4, v4 = _random_walk(best_pts, rnd, max(0.0, deadline - 8.0))
        if v4 > best_val:
            best_val = v4
            best_pts = [list(p) for p in pts4]

    # Final safety: ensure a valid result is returned.
    v, ok = _score(best_pts)
    if not ok:
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    return best_pts
# EVOLVE-BLOCK-END
