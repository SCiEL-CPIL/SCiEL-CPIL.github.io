import itertools
import time
import random

# ---------- helpers ----------

def _lines_of_count(points, target=4):
    """Count lines containing exactly `target` of the given points.
    Points: list of (x,y) ints, distinct.
    """
    n = len(points)
    if n < target:
        return 0
    pts = list(points)
    count = 0
    # canonical line key: (slope, intercept) using reduced (dx,dy) and a reference point
    from collections import defaultdict
    # For each pair, compute line key; group.
    line_pts = defaultdict(list)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            # normalize direction
            if dx == 0:
                dx, dy = 0, 1
            elif dy == 0:
                dx, dy = 1, 0
            else:
                if dx < 0:
                    dx, dy = -dx, -dy
                # reduce by gcd
                g = abs(dx)
                # gcd
                a, b = abs(dx), abs(dy)
                while b:
                    a, b = b, a % b
                dx //= a
                dy //= a
            # line: dy*x - dx*y = c, where c = dy*xi - dx*yi
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            line_pts[key].append((i, j))
    # For each line key, we have pairs; collect unique points
    for key, pairs in line_pts.items():
        s = set()
        for i, j in pairs:
            s.add(i)
            s.add(j)
        if len(s) == target:
            count += 1
    return count


def _count4(points):
    return _lines_of_count(points, 4)


def _valid(points):
    """Check no five collinear and distinct."""
    pts = list(points)
    if len(set(pts)) != len(pts):
        return False
    n = len(pts)
    from collections import defaultdict
    line_pts = defaultdict(set)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0:
                dx, dy = 0, 1
            elif dy == 0:
                dx, dy = 1, 0
            else:
                if dx < 0:
                    dx, dy = -dx, -dy
                a, b = abs(dx), abs(dy)
                while b:
                    a, b = b, a % b
                dx //= a
                dy //= a
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            line_pts[key].add(i)
            line_pts[key].add(j)
    for key, s in line_pts.items():
        if len(s) >= 5:
            return False
    return True


def _in_bounds(points):
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    return True


# ---------- base configuration: 5x5 grid with two corners removed ----------
# 5x5 grid has 5 rows, 5 cols, 2 diagonals = 12 four+ lines, but 5-collinear exists.
# Removing two points can break 5-collinearity while keeping many 4-lines.
# We'll search for a good static seed.

def _base_seed():
    # 5x5 grid points
    grid = [(x, y) for x in range(5) for y in range(5)]
    # Try removing pairs to get 16 points, no 5-collinear, maximize 4-lines
    best = None
    best_score = -1
    for i in range(25):
        for j in range(i + 1, 25):
            pts = [p for k, p in enumerate(grid) if k != i and k != j]
            if not _valid(pts):
                continue
            s = _count4(pts)
            if s > best_score:
                best_score = s
                best = pts
    if best is None:
        best = [(x, y) for x in range(4) for y in range(4)]
    return best, best_score


# ---------- local search ----------

def _local_search(points, deadline, rng):
    """Local search: move a point to a nearby integer location or swap with a candidate.
    Returns best (points, score) found before deadline.
    """
    cur = [tuple(p) for p in points]
    cur_score = _count4(cur)
    best = list(cur)
    best_score = cur_score

    # candidate pool: integer points in a box around current bounding region
    minx = min(x for x, y in cur); maxx = max(x for x, y in cur)
    miny = min(y for x, y in cur); maxy = max(y for x, y in cur)
    pad = 3
    pool = set()
    for x in range(minx - pad, maxx + pad + 1):
        for y in range(miny - pad, maxy + pad + 1):
            if abs(x) <= 1000 and abs(y) <= 1000:
                pool.add((x, y))
    pool = list(pool)

    step = 0
    while time.time() < deadline:
        step += 1
        # choose a random index to move
        idx = rng.randrange(len(cur))
        ox, oy = cur[idx]
        # choose a random new location from pool (or a small random offset)
        if rng.random() < 0.5 and pool:
            nx, ny = rng.choice(pool)
        else:
            nx = ox + rng.randint(-2, 2)
            ny = oy + rng.randint(-2, 2)
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        if (nx, ny) in cur:
            continue
        trial = list(cur)
        trial[idx] = (nx, ny)
        if not _valid(trial):
            continue
        s = _count4(trial)
        if s > cur_score:
            cur = trial
            cur_score = s
            if s > best_score:
                best = list(cur)
                best_score = s
        # occasional swap
        elif rng.random() < 0.1:
            j = rng.randrange(len(cur))
            if j == idx:
                continue
            trial2 = list(cur)
            trial2[idx], trial2[j] = cur[j], cur[idx]
            if not _valid(trial2):
                continue
            s2 = _count4(trial2)
            if s2 > cur_score:
                cur = trial2
                cur_score = s2
                if s2 > best_score:
                    best = list(cur)
                    best_score = s2
    return best, best_score


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s
    rng = random.Random(12345)

    # Start from best static seed
    base, base_score = _base_seed()
    best = list(base)
    best_score = base_score

    # If base is already strong, do a quick local search to improve
    if time.time() < deadline - 5.0:
        cand, cs = _local_search(base, deadline, rng)
        if cs > best_score:
            best = cand
            best_score = cs

    # Final validation
    if not _valid(best) or not _in_bounds(best) or len(set(best)) != 16:
        # fallback to 4x4 grid
        best = [(x, y) for x in range(4) for y in range(4)]

    return [[x, y] for x, y in best]
