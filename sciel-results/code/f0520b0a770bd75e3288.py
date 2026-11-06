"""Search for 16 integer points maximizing lines with exactly 4 points."""
import time
import random
from itertools import combinations

DEADLINE = 140.0  # safety margin under 150s kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _count4(points):
    """Count lines containing exactly 4 of the given points (no 5 collinear assumed).
    Returns (num_exact4, max_collinear)."""
    n = len(points)
    # Map each line (normalized) to set of point indices
    line_map = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            # normalize direction
            if dx < 0:
                dx, dy = -dx, -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            # reduce by gcd
            g = _gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            # line key: normalized direction + a point on line (affine invariant)
            # Use (dx, dy, c) where c = dy*xi - dx*yi (line: dy*x - dx*y = c)
            c = dy * xi - dx * yi
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    num4 = 0
    maxc = 0
    for s in line_map.values():
        L = len(s)
        if L > maxc:
            maxc = L
        if L == 4:
            num4 += 1
    return num4, maxc


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _valid(points):
    # distinct, in bounds, no five collinear
    if len(points) != 16:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    _, maxc = _count4(points)
    return maxc <= 4


def _score(points):
    n4, maxc = _count4(points)
    if maxc > 4:
        return -1000 - maxc
    return n4


def _grid4x4():
    return [[x, y] for x in range(4) for y in range(4)]


def _random_start(rng):
    pts = []
    while len(pts) < 16:
        x = rng.randint(-60, 60)
        y = rng.randint(-60, 60)
        if (x, y) not in [(p[0], p[1]) for p in pts]:
            pts.append([x, y])
    return pts


def _local_search(points, rng, max_iters, time_limit):
    pts = [p[:] for p in points]
    cur = _score(pts)
    best = cur
    best_pts = [p[:] for p in pts]
    for it in range(max_iters):
        if time.time() > time_limit:
            break
        # pick a random point and a random move
        idx = rng.randrange(16)
        x, y = pts[idx]
        # candidate moves: small perturbations + line-aligned shifts
        moves = []
        # small perturbations
        for dx in (-2, -1, 1, 2):
            for dy in (-2, -1, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                moves.append((x + dx, y + dy))
        # line-aligned: try moving along directions of existing lines through idx
        # cheap: just add a few random larger steps
        for _ in range(6):
            dx = rng.randint(-8, 8)
            dy = rng.randint(-8, 8)
            if dx != 0 or dy != 0:
                moves.append((x + dx, y + dy))
        # dedupe
        seenm = set()
        uniq = []
        for mx, my in moves:
            if (mx, my) in seenm:
                continue
            seenm.add((mx, my))
            if abs(mx) > 1000 or abs(my) > 1000:
                continue
            if (mx, my) in [(p[0], p[1]) for i, p in enumerate(pts) if i != idx]:
                continue
            uniq.append((mx, my))
        if not uniq:
            continue
        nx, ny = rng.choice(uniq)
        old = pts[idx]
        pts[idx] = [nx, ny]
        ns = _score(pts)
        if ns > cur:
            cur = ns
            if cur > best:
                best = cur
                best_pts = [p[:] for p in pts]
        else:
            pts[idx] = old
    return best, best_pts


def construct():
    deadline = _START + DEADLINE
    rng = random.Random(12345)
    best_score = -10**9
    best_pts = None

    # Start from the 4x4 grid (known 10) and several random starts
    starts = [_grid4x4()]
    for _ in range(8):
        starts.append(_random_start(rng))

    # First pass: evaluate starts, keep best
    for s in starts:
        if _elapsed() > deadline - 2:
            break
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_pts = [p[:] for p in s]

    # Local search from best start and a few others, within budget
    remaining = deadline - _elapsed()
    per_iter_budget = max(0.5, remaining / 200.0)
    for s in starts:
        if _elapsed() > deadline - 1:
            break
        sc, sp = _local_search(s, rng, max_iters=300, time_limit=deadline - 1)
        if sc > best_score:
            best_score = sc
            best_pts = [p[:] for p in sp]

    # Final polish: more iterations from the current best
    if best_pts is not None and _elapsed() < deadline - 2:
        sc, sp = _local_search(best_pts, rng, max_iters=2000, time_limit=deadline - 1)
        if sc > best_score:
            best_score = sc
            best_pts = [p[:] for p in sp]

    if best_pts is None or not _valid(best_pts):
        return _grid4x4()
    return best_pts
