"""Explicit construction for the 12-point, few-ordinary-lines task.

Strategy:
  - Build a candidate 12-point integer set from structured seeds.
  - Local search: repeatedly replace one point with a better integer point
    (from a bounded candidate pool) to reduce the number of ordinary lines.
  - Return the best configuration found within the wall-clock budget.
"""
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 120 s kill


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _make_seed_grid():
    """3x4 grid seed."""
    return [[x, y] for x in range(4) for y in range(3)]


def _make_seed_lines():
    """Seed: 12 points arranged on 3 lines of 4, sharing intersection points.

    Line A: y=0, x in {0,1,2,3}
    Line B: y=3, x in {1,2,3,4}
    Line C: x+y=3 -> (0,3),(1,2),(2,1),(3,0)
    Union: (0,0),(1,0),(2,0),(3,0),(1,3),(2,3),(3,3),(4,3),(0,3),(1,2),(2,1)
    That's 11. Add one more: (0,1) on a new line? Let's just use grid seed.
    """
    # Just return a 3x4 grid for now; local search will improve it.
    return _make_seed_grid()


def _candidate_pool(points, radius=6):
    """Generate candidate integer points near the existing point cloud."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)
    pool = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            pool.append((cx + dx, cy + dy))
    # Also include points aligned with existing lines (extend each line by 1 step)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # direction
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            # extend in both directions
            pool.append((x1 - dx, y1 - dy))
            pool.append((x2 + dx, y2 + dy))
    # Deduplicate
    seen = set()
    unique = []
    for p in pool:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _local_search(points, pool, max_iters=200000):
    """Hill-climb: replace one point to reduce ordinary lines."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(42)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        # Pick a random index to replace
        idx = rng.randrange(12)
        old = cur[idx]
        # Pick a random candidate from pool (not equal to any current point)
        cur_set = set(tuple(p) for p in cur)
        # Try a few candidates
        improved = False
        for _ in range(20):
            cand = rng.choice(pool)
            if cand in cur_set:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        if not improved:
            # Occasionally refresh the pool
            if iters % 2000 == 0:
                pool = _candidate_pool(cur, radius=8)
    return cur, cur_score


def _local_search2(points, max_iters=300000):
    """More aggressive local search with dynamic candidate generation."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(123)
    iters = 0
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        idx = rng.randrange(12)
        old = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        # Generate candidates on-the-fly: perturb old point, or place on a random line
        improved = False
        # Strategy 1: small perturbation
        for _ in range(15):
            nx = old[0] + rng.randint(-5, 5)
            ny = old[1] + rng.randint(-5, 5)
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
        if improved:
            continue
        # Strategy 2: place on a line through two existing points (extend)
        for _ in range(10):
            i = rng.randrange(12)
            j = rng.randrange(12)
            if i == j:
                continue
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            # Place at x1 + k*dx, y1 + k*dy for random k
            k = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5])
            nx = x1 + k * dx
            ny = y1 + k * dy
            cand = (nx, ny)
            if cand in cur_set:
                continue
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                break
    return cur, cur_score


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    # Seed 1: 3x4 grid
    seed1 = _make_seed_grid()
    if _valid(seed1):
        s1 = _count_ordinary(seed1)
        if s1 < best_score:
            best_score = s1
            best_pts = [p[:] for p in seed1]

    # Local search from seed1
    if time.time() < _DEADLINE - 10:
        pool1 = _candidate_pool(seed1, radius=8)
        pts_ls, score_ls = _local_search(seed1, pool1, max_iters=100000)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # Local search 2 from best so far
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_ls2, score_ls2 = _local_search2(best_pts, max_iters=150000)
        if score_ls2 < best_score:
            best_score = score_ls2
            best_pts = pts_ls2

    # Seed 2: 4x3 grid (transpose)
    seed2 = [[x, y] for x in range(3) for y in range(4)]
    if _valid(seed2):
        s2 = _count_ordinary(seed2)
        if s2 < best_score:
            best_score = s2
            best_pts = [p[:] for p in seed2]
        if time.time() < _DEADLINE - 10:
            pool2 = _candidate_pool(seed2, radius=8)
            pts_ls3, score_ls3 = _local_search(seed2, pool2, max_iters=100000)
            if score_ls3 < best_score:
                best_score = score_ls3
                best_pts = pts_ls3

    # Seed 3: structured line-based seed
    # 3 lines of 4 points, arranged to share intersections
    # Line 1: (0,0),(1,0),(2,0),(3,0)
    # Line 2: (0,3),(1,3),(2,3),(3,3)
    # Line 3: (0,1),(1,2),(2,0) -- wait, (2,0) is on line 1
    # Let's use: 4 points on y=0, 4 on y=2, 4 on x=2 (vertical), sharing (2,0),(2,2)
    seed3 = [
        [0, 0], [1, 0], [2, 0], [3, 0],   # y=0
        [0, 2], [1, 2], [2, 2], [3, 2],   # y=2
        [2, 1], [2, 3],                    # x=2, but (2,0),(2,2) already there
        [4, 1], [4, 3]                     # extra
    ]
    # Dedup
    seen = set()
    seed3_clean = []
    for p in seed3:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            seed3_clean.append(p[:])
    if len(seed3_clean) == 12 and _valid(seed3_clean):
        s3 = _count_ordinary(seed3_clean)
        if s3 < best_score:
            best_score = s3
            best_pts = [p[:] for p in seed3_clean]
        if time.time() < _DEADLINE - 10:
            pool3 = _candidate_pool(seed3_clean, radius=8)
            pts_ls4, score_ls4 = _local_search(seed3_clean, pool3, max_iters=100000)
            if score_ls4 < best_score:
                best_score = score_ls4
                best_pts = pts_ls4

    # Final polish: one more local search pass on best
    if best_pts is not None and time.time() < _DEADLINE - 8:
        pts_final, score_final = _local_search2(best_pts, max_iters=200000)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = _make_seed_grid()

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
