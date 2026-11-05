# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_START = time.time()
_DEADLINE = _START + 138.0  # safety margin before 150 s kill


def _line_key(p1, p2):
    """Normalized integer line coefficients (A, B, C) for line through p1, p2."""
    x1, y1 = p1
    x2, y2 = p2
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
    return (A, B, C)


def _all_lines(pts):
    """Multiset (key -> count) of lines determined by all pairs of pts."""
    counts = {}
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            counts[k] = counts.get(k, 0) + 1
    return counts


def _ordinary_from_counts(counts):
    # a line is 'ordinary' iff exactly one pair lies on it (i.e. exactly 2 points)
    return sum(1 for c in counts.values() if c == 1)


def _count_ordinary(pts):
    return _ordinary_from_counts(_all_lines(pts))


def _valid(points):
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
    # not all collinear
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return True
    return False


def _delta_move(counts, idx, old, new, pts):
    """Return new ordinary count after replacing pts[idx]=old with new.
    Uses incremental update of the line multiset."""
    # lines through old with each other point
    for j in range(12):
        if j == idx:
            continue
        k = _line_key(old, pts[j])
        counts[k] -= 1
        if counts[k] == 0:
            del counts[k]
    # lines through new with each other point
    for j in range(12):
        if j == idx:
            continue
        k = _line_key(new, pts[j])
        counts[k] = counts.get(k, 0) + 1
    return _ordinary_from_counts(counts)


def _get_pool(pts):
    """Candidate replacement positions: near-lattice + line extensions."""
    pool = set()
    for (x, y) in pts:
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx == 0 and dy == 0:
                    continue
                pool.add((x + dx, y + dy))
    # extend every pair-line a few integer steps
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(1, 4):
                pool.add((x1 + k * dx, y1 + k * dy))
                pool.add((x1 - k * dx, y1 - k * dy))
                pool.add((x2 + k * dx, y2 + k * dy))
                pool.add((x2 - k * dx, y2 - k * dy))
    # keep within bounds
    pool = {p for p in pool if abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9}
    return list(pool)


def _local_search_sa(pts0, budget):
    """Simulated annealing with incremental counting. Returns best (pts, score)."""
    cur = [list(p) for p in pts0]
    counts = _all_lines(cur)
    cur_score = _ordinary_from_counts(counts)
    best = [list(p) for p in cur]
    best_score = cur_score
    rng = random.Random(hash(tuple(tuple(p) for p in cur)) & 0xffffffff)
    pool = _get_pool(cur)
    iters = 0
    start_temp = 8.0
    end_temp = 0.05
    max_iters = 300000
    while iters < max_iters and time.time() < budget:
        iters += 1
        temp = start_temp * (end_temp / start_temp) ** (iters / max_iters)
        idx = rng.randint(0, 11)
        old = tuple(cur[idx])
        cur_set = set(tuple(p) for p in cur)
        # try a batch of random pool candidates
        n_try = min(8, len(pool))
        improved = False
        for cand in rng.sample(pool, n_try):
            if cand in cur_set:
                continue
            new_score = _delta_move(counts, idx, old, cand, cur)
            delta = new_score - cur_score
            if delta < 0 or (temp > 0 and rng.random() < min(1.0, 1.0 / (1.0 + max(delta, 0) / temp))):
                cur[idx] = list(cand)
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best = [list(p) for p in cur]
                break
        if not improved and iters % 400 == 0:
            pool = _get_pool(cur)
    return best, best_score


def _configs():
    cfgs = []
    cfgs.append([[x, y] for x in range(4) for y in range(3)])
    cfgs.append([[x, y] for x in range(3) for y in range(4)])
    cfgs.append([[x, y] for x in range(2) for y in range(6)])
    cfgs.append([[x, y] for x in range(6) for y in range(2)])
    cfgs.append([[t, t * t] for t in range(-5, 5)])
    cfgs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    cfgs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                 [0, 1], [0, 2], [0, 3],
                 [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    cfgs.append([(x, y) for x in range(3) for y in range(3)] + [[3, 0], [3, 1], [3, 2]])
    cfgs.append([(x, y) for x in range(3) for y in range(3)] + [[1, 3], [3, 1], [-1, 1]])
    return cfgs


def construct():
    best_pts = None
    best_score = float('inf')

    # Evaluate diverse seeds
    for cfg in _configs():
        if not _valid(cfg):
            continue
        s = _count_ordinary(cfg)
        if s < best_score:
            best_score = s
            best_pts = [list(p) for p in cfg]

    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)

    # Aggressive local search from the best seed and a few others.
    # Incremental counting makes each move cheap -> many more steps fit.
    seeds = [best_pts]
    for cfg in _configs():
        if not _valid(cfg):
            continue
        if time.time() > _DEADLINE - 12:
            break
        seeds.append([list(p) for p in cfg])
        if len(seeds) >= 5:
            break

    for seed in seeds:
        if time.time() > _DEADLINE - 8:
            break
        pts, score = _local_search_sa(seed, _DEADLINE - 8)
        if score < best_score:
            best_score = score
            best_pts = pts
        if best_score <= 6:
            break

    if not _valid(best_pts):
        best_pts = [[x, y] for x in range(4) for y in range(3)]
    return best_pts
# EVOLVE-BLOCK-END
