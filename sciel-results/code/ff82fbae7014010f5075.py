"""Program for 12-point few ordinary lines task."""
import time
import random
from math import gcd

# Deadline guard: start the clock immediately
_START = time.time()
_DEADLINE = 150.0

def _time_left():
    return _DEADLINE - 15.0 - (time.time() - _START)  # 15s safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the points. O(n^3) but n=12."""
    n = len(points)
    if n < 3:
        return 0
    # Map line -> set of points on it
    # Use canonical (dx, dy, cross) with gcd-normalized dx,dy
    line_map = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            g = gcd(abs(dx), abs(dy))
            if g:
                dx //= g
                dy //= g
            # canonical sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            cross = dx * yi - dy * xi
            key = (dx, dy, cross)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    ordinary = 0
    for key, ptset in line_map.items():
        if len(ptset) == 2:
            ordinary += 1
    return ordinary

def _valid(points):
    """Check: 12 distinct integer points, |x|,|y|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not (isinstance(p, list) and len(p) == 2):
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # not all collinear
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    if (x1 - x0) * (y2 - y0) == (x2 - x0) * (y1 - y0):
        # check all others
        for p in points[3:]:
            x, y = p
            if (x1 - x0) * (y - y0) != (x2 - x0) * (y1 - y0):
                return True
        return False
    return True

def _known_configs():
    """Generate a list of known or heuristic configurations."""
    configs = []

    # 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])

    # 4x3 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])

    # 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])

    # 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])

    # 3x3 grid + 3 extra points
    base = [[x, y] for x in range(3) for y in range(3)]
    # add points that might lie on existing lines
    extras = [[3, 0], [0, 3], [1, 3]]
    configs.append(base + extras)
    extras2 = [[3, 1], [1, 3], [2, 4]]
    configs.append(base + extras2)
    extras3 = [[4, 0], [0, 4], [2, 2]]  # (2,2) already in base, skip
    # fix: use distinct
    extras3 = [[4, 0], [0, 4], [3, 2]]
    configs.append(base + extras3)

    # Hessian-like: 9 inflection points of cubic + 3 more
    # y^2 = x^3 - 1 has 9 inflection points at (1,0), (omega,0)... not integer
    # Try Pappus configuration (9 points) + 3
    # Pappus: two lines of 3 points each, plus 3 intersection points
    # (0,0),(1,0),(2,0) and (0,1),(1,1),(2,1) -> lines y=0, y=1
    # cross intersections: (0,0)-(1,1),(1,0)-(2,1),(2,0)-(0,1) etc
    pappus = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]]
    # add 3 more
    configs.append(pappus + [[3,0],[0,2],[2,2]])
    configs.append(pappus + [[3,1],[1,2],[2,3]])

    # Symmetric: regular 12-gon approximations
    # Just try a few small random seeds
    return configs

def _local_search(points, max_iters=200000):
    """Local search: change one coordinate of one point. Return best."""
    import time as _t
    best = [p[:] for p in points]
    best_score = _count_ordinary(best)
    cur = [p[:] for p in points]
    cur_score = best_score
    step = 1
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        # pick random point and coordinate
        idx = random.randrange(12)
        coord = random.randrange(2)
        old_val = cur[idx][coord]
        # try a few candidate values
        candidates = [old_val - 1, old_val + 1, old_val - 2, old_val + 2,
                      old_val - 3, old_val + 3, random.randint(-5, 5) + old_val]
        best_cand = None
        best_cand_score = cur_score
        for cv in candidates:
            if abs(cv) > 10**9:
                continue
            test = [p[:] for p in cur]
            test[idx][coord] = cv
            # check distinctness
            seen = set()
            valid = True
            for p in test:
                key = (p[0], p[1])
                if key in seen:
                    valid = False
                    break
                seen.add(key)
            if not valid:
                continue
            s = _count_ordinary(test)
            if s < best_cand_score:
                best_cand_score = s
                best_cand = test
        if best_cand is not None:
            cur = best_cand
            cur_score = best_cand_score
            if cur_score < best_score:
                best_score = cur_score
                best = [p[:] for p in cur]
        else:
            # occasional random restart to escape local minima
            if it % 1000 == 0:
                cur = [p[:] for p in random.sample(range(-10, 11), 12)]
                # make distinct
                for i in range(12):
                    while (cur[i][0], cur[i][1]) in [(cur[j][0], cur[j][1]) for j in range(i)]:
                        cur[i][0] = random.randint(-10, 10)
                        cur[i][1] = random.randint(-10, 10)
                cur_score = _count_ordinary(cur)
                if cur_score < best_score:
                    best_score = cur_score
                    best = [p[:] for p in cur]
    return best, best_score

def construct():
    import time as _t
    best_points = None
    best_score = float('inf')

    # Try known configurations
    for cfg in _known_configs():
        if _valid(cfg):
            s = _count_ordinary(cfg)
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in cfg]
            if best_score <= 6:
                return best_points

    # Local search from best known
    if best_points is not None:
        pts, score = _local_search(best_points, max_iters=500000)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            return best_points

    # Random local searches from scratch
    for _ in range(50):
        if _time_left() < 5.0:
            break
        pts = [[random.randint(-8, 8), random.randint(-8, 8)] for _ in range(12)]
        # ensure distinct
        for i in range(12):
            tries = 0
            while (pts[i][0], pts[i][1]) in [(pts[j][0], pts[j][1]) for j in range(i)] and tries < 100:
                pts[i][0] = random.randint(-8, 8)
                pts[i][1] = random.randint(-8, 8)
                tries += 1
        if not _valid(pts):
            continue
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
        if best_score <= 6:
            return best_points
        # local search
        pts2, score2 = _local_search(pts, max_iters=100000)
        if score2 < best_score:
            best_score = score2
            best_points = [p[:] for p in pts2]
        if best_score <= 6:
            return best_points

    # Final fallback
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]

    return best_points
