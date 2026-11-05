"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

_DEADLINE = 140.0  # safety margin before 150s kill
_start_time = None


def _elapsed():
    return time.time() - _start_time


def _count_ordinary_lines(pts):
    """pts: list of 12 [x,y] integer points. Returns number of lines containing exactly 2 points."""
    n = len(pts)
    # map from normalized line to count
    line_count = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            # normalize direction (dx, dy) and offset to canonical form
            # line: A x + B y = C where A=dy, B=-dx, C=dy*x1 - dx*y1
            A = dy
            B = -dx
            C = dy * x1 - dx * y1
            g = math.gcd(A, B, C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            # canonical sign: first nonzero of (A,B) positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            line_count[key] = line_count.get(key, 0) + 1
    # ordinary lines are those with exactly 2 points
    ordinary = sum(1 for c in line_count.values() if c == 2)
    return ordinary


def _all_collinear(pts):
    if len(pts) < 3:
        return True
    (x1, y1) = pts[0]
    (x2, y2) = pts[1]
    for (x, y) in pts[2:]:
        if (x2 - x1) * (y - y1) != (y2 - y1) * (x - x1):
            return False
    return True


def _distinct(pts):
    return len(set(tuple(p) for p in pts)) == len(pts)


def _valid(pts):
    if len(pts) != 12:
        return False
    if not _distinct(pts):
        return False
    for (x, y) in pts:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
    if _all_collinear(pts):
        return False
    return True


def _grid_4x3():
    return [[x, y] for x in range(4) for y in range(3)]


def _boroczky_like():
    """A configuration with many collinearities.
    Use points: 6 on a line, and arrange others to form multiple 3+ lines.
    """
    # 6 points on x-axis, 3 on y-axis, 3 on line x+y=const
    # Let's do: 6 on y=0, 4 on x=0, 2 on x+y=5 (but avoid overlap)
    # Actually simpler: use a known good pattern
    pts = []
    # 6 points on line y=0
    for i in range(6):
        pts.append([i, 0])
    # 3 points on line x=0 (one overlaps at (0,0))
    pts.append([0, 1])
    pts.append([0, 2])
    pts.append([0, 3])
    # 3 more points to make additional collinearities
    # Line through (1,0) and (0,1): x+y=1 -> add (2,-1)? no, keep positive
    # Let's add points on line x+y=3: (3,0) already, (2,1), (1,2), (0,3) already
    pts.append([2, 1])
    pts.append([1, 2])
    # Need one more. Line through (4,0) and (2,1): slope -1/2, x+2y=4 -> (0,2) already, (6,-1)
    # Let's just add (3, 1) and check
    pts.append([3, 1])
    return pts


def _random_good_start(rng):
    """Generate a random configuration biased toward collinearities."""
    pts = []
    # Choose 3-4 lines, put multiple points on each
    num_lines = rng.choice([3, 4])
    lines = []
    for _ in range(num_lines):
        # random line through 2 random points in a small range
        x1, y1 = rng.randint(-5, 5), rng.randint(-5, 5)
        x2, y2 = rng.randint(-5, 5), rng.randint(-5, 5)
        if x1 == x2 and y1 == y2:
            continue
        lines.append((x1, y1, x2, y2))
    # Assign 12 points to lines (with overlaps)
    for i in range(12):
        if lines:
            line = lines[i % len(lines)]
            x1, y1, x2, y2 = line
            t = rng.randint(0, 3)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            pts.append([x, y])
        else:
            pts.append([rng.randint(-5, 5), rng.randint(-5, 5)])
    # Remove duplicates by jittering
    seen = set()
    for i, p in enumerate(pts):
        while tuple(p) in seen:
            p = [p[0] + rng.randint(-2, 2), p[1] + rng.randint(-2, 2)]
        seen.add(tuple(p))
        pts[i] = p
    return pts


def _candidate_positions(rng, pts):
    """Generate candidate positions for a point to move to."""
    cands = set()
    # Small integer grid
    for x in range(-10, 11):
        for y in range(-10, 11):
            cands.add((x, y))
    # Positions that create collinearities with existing points
    for i, (x1, y1) in enumerate(pts):
        for j, (x2, y2) in enumerate(pts):
            if i == j:
                continue
            # Line through i and j, extend
            dx = x2 - x1
            dy = y2 - y1
            for t in range(-3, 4):
                if t == 0 or t == 1:
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    cands.add((nx, ny))
    # Random
    for _ in range(20):
        cands.add((rng.randint(-20, 20), rng.randint(-20, 20)))
    return list(cands)


def _local_search(pts, rng, time_limit=5.0):
    """Local search: move one point at a time to reduce ordinary lines."""
    best = pts[:]
    best_score = _count_ordinary_lines(best)
    deadline = time.time() + time_limit
    current = best_score
    temp = 1.0
    for _ in range(10000):
        if time.time() > deadline or _elapsed() > _DEADLINE - 10:
            break
        # Pick a random point
        i = rng.randrange(12)
        old = best[i]
        # Pick a candidate position
        cands = _candidate_positions(rng, best)
        new_pos = rng.choice(cands)
        if new_pos == tuple(old):
            continue
        trial = best[:]
        trial[i] = [new_pos[0], new_pos[1]]
        if not _valid(trial):
            continue
        trial_score = _count_ordinary_lines(trial)
        delta = trial_score - current
        if delta < 0 or (delta == 0 and rng.random() < 0.1) or (rng.random() < math.exp(-delta / max(temp, 0.01))):
            current = trial_score
            best = trial
            if current < best_score:
                best_score = current
            temp *= 0.999
    return best, best_score


def construct():
    global _start_time
    _start_time = time.time()
    rng = random.Random(42)

    # Start with several initial configurations
    inits = [
        _grid_4x3(),
        _boroczky_like(),
        _random_good_start(rng),
        _random_good_start(rng),
        _random_good_start(rng),
    ]

    best_pts = None
    best_score = float('inf')

    for init in inits:
        if _elapsed() > _DEADLINE - 15:
            break
        if not _valid(init):
            continue
        init_score = _count_ordinary_lines(init)
        if init_score < best_score:
            best_score = init_score
            best_pts = init[:]
        # Local search from this init
        deadline_left = _DEADLINE - 15 - _elapsed()
        if deadline_left < 1:
            break
        ls_pts, ls_score = _local_search(init, rng, time_limit=max(1.0, min(deadline_left * 0.3, 10.0)))
        if ls_score < best_score:
            best_score = ls_score
            best_pts = ls_pts[:]

    # If we still have time, do more local search from the best
    if best_pts is not None and _elapsed() < _DEADLINE - 20:
        deadline_left = _DEADLINE - 20 - _elapsed()
        for _ in range(5):
            if _elapsed() > _DEADLINE - 15:
                break
            ls_pts, ls_score = _local_search(best_pts, rng, time_limit=max(1.0, min(deadline_left * 0.2, 15.0)))
            if ls_score < best_score:
                best_score = ls_score
                best_pts = ls_pts[:]
            deadline_left = _DEADLINE - 20 - _elapsed()
            if deadline_left < 1:
                break

    if best_pts is None:
        best_pts = _grid_4x3()

    # Final validity check
    if not _valid(best_pts):
        best_pts = _grid_4x3()

    return [[int(p[0]), int(p[1])] for p in best_pts]
# EVOLVE-BLOCK-END
