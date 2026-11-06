import time
import random
from itertools import combinations

_DEADLINE = 110.0  # seconds wall-clock budget, safety margin under 120s

def _elapsed(start):
    return time.time() - start

def _is_collinear(p, q, r):
    (x1, y1), (x2, y2), (x3, y3) = p, q, r
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _valid_no_four(points):
    """Return True if no four points are collinear and all distinct within bounds."""
    n = len(points)
    # distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
    # no four collinear: for every pair, count collinear
    # efficient: for each i, group by slope to j>i
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            # reduce
            if dx == 0 and dy == 0:
                return False
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes[(dx, dy)] = slopes.get((dx, dy), 0) + 1
            if slopes[(dx, dy)] >= 3:
                return False  # i plus 3 others on same line = 4 collinear
    return True

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _build_triples(points):
    """Return list of frozensets of size-3 collinear index sets."""
    n = len(points)
    triples = set()
    for i in range(n):
        slopes = {}
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            if dx == 0 and dy == 0:
                continue
            g = _gcd(dx, dy)
            dx //= g
            dy //= g
            slopes.setdefault((dx, dy), []).append(j)
        for k in slopes.values():
            for a, b in combinations(k, 2):
                triples.add(frozenset((i, a, b)))
    return triples

def _max_3collinear_free(points, deadline_start, budget_frac=0.5):
    """Branch and bound for largest subset with no three collinear.
    Returns best size found (may be optimistic lower bound if cut off)."""
    n = len(points)
    triples = _build_triples(points)
    # For each vertex, which triples contain it
    vertex_triples = [set() for _ in range(n)]
    for t in triples:
        for v in t:
            vertex_triples[v].add(t)
    # adjacency: two vertices conflict if they are in a common triple? No.
    # Constraint: no triple fully contained in selected set.
    # This is a 3-uniform hypergraph independent set.
    # B&B: branch include/exclude; when including v, must exclude enough to break all triples containing v.
    # Use recursive with bitmasks.
    allmask = (1 << n) - 1
    # Precompute for each vertex the list of triples (as bitmask) containing it
    vtrip = [[] for _ in range(n)]
    for t in triples:
        m = 0
        for v in t:
            m |= (1 << v)
        for v in t:
            vtrip[v].append(m)

    best = [0]
    # order vertices by degree (number of triples) descending for pruning
    order = sorted(range(n), key=lambda v: -len(vtrip[v]))
    # We'll do a DFS over vertices in `order`, deciding include/exclude,
    # maintaining a mask of "forbidden" vertices (that must be excluded)
    # because including current would complete a triple.
    # A vertex v can be included iff none of its triples are already "completed"
    # by two other included vertices. We track included set.

    def dfs(idx, included_mask, count):
        if time.time() - deadline_start > 110.0:
            return
        if count + (n - idx) <= best[0]:
            return
        if idx == n:
            if count > best[0]:
                best[0] = count
            return
        v = order[idx]
        # Option 1: exclude v
        dfs(idx + 1, included_mask, count)
        # Option 2: include v if it doesn't complete any triple
        ok = True
        for m in vtrip[v]:
            # triple m contains v; the other two bits
            others = m & ~(1 << v)
            if (included_mask & others) == others:
                ok = False
                break
        if ok:
            dfs(idx + 1, included_mask | (1 << v), count + 1)

    dfs(0, 0, 0)
    return best[0]

def _score(points):
    return _max_3collinear_free(points, time.time())

def construct():
    start = time.time()
    rng = random.Random(12345)

    # Candidate generator: start from a base structured set, then perturb.
    # Base idea: a 5x4 grid gives many 3-collinear but also 4-collinear.
    # We'll generate random-ish sets biased toward low coordinates and test.

    best_pts = None
    best_score = 21  # worse than any valid (max 20)

    def consider(pts):
        nonlocal best_pts, best_score
        if not _valid_no_four(pts):
            return
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts

    # Strategy A: take a base grid and randomly delete/replace to break 4-collinear
    # while preserving 3-collinear structure.
    base = [(x, y) for x in range(0, 5) for y in range(0, 4)]  # 20 points, 5 cols x 4 rows
    # This base has rows of 5 (4-collinear) and cols of 4 (4-collinear) -> invalid.
    # Perturb: random shifts.

    for attempt in range(20000):
        if time.time() - start > 105.0:
            break
        pts = list(base)
        # random perturbation: shift some points by small amounts
        for _ in range(rng.randint(3, 8)):
            idx = rng.randrange(20)
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            x, y = pts[idx]
            nx, ny = x + dx, y + dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                pts[idx] = (nx, ny)
        consider(pts)
        if best_score <= 12:
            break

    # Strategy B: pure random sampling in a small box, many trials
    for attempt in range(30000):
        if time.time() - start > 108.0:
            break
        pts = []
        for _ in range(20):
            pts.append((rng.randint(-6, 6), rng.randint(-6, 6)))
        consider(pts)
        if best_score <= 12:
            break

    # Strategy C: structured "3-per-line" via modular arithmetic on a torus-like grid
    # Use points (x, (x^2 mod p)-ish) but integer small; combine two conics.
    # Try a few hand-crafted dense configurations.
    for c in _candidate_configs():
        if time.time() - start > 109.0:
            break
        consider(c)
        if best_score <= 12:
            break

    if best_pts is None:
        # Fallback: a valid set (parabola) score 20
        return [[x, x * x] for x in range(-10, 10)]

    return [list(p) for p in best_pts]

def _candidate_configs():
    cfgs = []
    # A: 20 points from a 6x6 grid with a hole pattern that breaks 4-collinear
    # Use a "no four in a line" known pattern but dense.
    # Take a 7x7 grid and pick 20 via a modular rule: y = (a*x^2 + b*x + c) mod 7 style won't be integer small enough? It is small.
    # Conic over integers: y = x^2 gives no 3 collinear. We WANT 3-collinear.
    # Use union of a few 3-point lines.
    # Build lines with 3 points each, reuse points.
    # Simple: take the 20 points as a 5x4 grid but replace one point per long line to break 4-collinear.
    base = [(x, y) for x in range(5) for y in range(4)]
    # rows (y fixed) have 5 points -> 4-collinear. Break each row by moving one point.
    for y in range(4):
        # move the point at x=4 in row y down by 1 (out of row)
        pts = list(base)
        pts[4 + y * 5] = (4, y + 4)  # move to y=4,5,6,7 -> but that may create new 4-collinear
        cfgs.append(pts)
    # columns (x fixed) have 4 points -> 4-collinear. Break each column.
    for x in range(5):
        pts = list(base)
        pts[x * 4 + 3] = (x + 5, 3)  # move bottom of column out
        cfgs.append(pts)
    # break both rows and columns
    pts = list(base)
    for y in range(4):
        pts[4 + y * 5] = (4, y + 4)
    for x in range(5):
        # recompute index after row moves; just rebuild
        pass
    # rebuild cleanly
    pts = list(base)
    moved = set()
    for y in range(4):
        i = 4 + y * 5
        pts[i] = (4, y + 4)
        moved.add(i)
    for x in range(5):
        i = x * 4 + 3
        if i not in moved:
            pts[i] = (x + 5, 3)
    cfgs.append(pts)
    return cfgs
