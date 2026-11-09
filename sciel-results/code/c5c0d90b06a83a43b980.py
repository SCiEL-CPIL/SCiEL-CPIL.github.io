"""Rewritten program: search for 20 integer points with no four collinear and a small no-three-collinear subset."""
import time
import random
import itertools
from math import gcd

_DEADLINE_SECONDS = 110.0  # safety margin before the 120s kill
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    # integer cross product test
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    # returns True if any 4 of the points are collinear
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            # count how many other points are on line ij
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            # normalize direction to avoid double counting and to use a canonical key
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dxn, dyn = dx // g, dy // g
            else:
                dxn, dyn = dx, dy
            # canonical sign
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            cnt = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0] - points[i][0]) * dyn == (points[k][1] - points[i][1]) * dxn:
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False


def _max_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # precompute for each pair, the set of other points collinear with them
    # we need for any triple whether they are collinear
    triple_bad = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triple_bad[i][j] = True
                    triple_bad[i][k] = True
                    triple_bad[j][k] = True

    # We want max size subset S with no bad triple.
    # Use branch and bound on inclusion/exclusion.
    # Represent state by (start_index, current_set_mask or list, current_size)
    # To speed up, precompute for each point the list of points that form a bad triple with two others in the current set?
    # Simpler: recursive with pruning.

    best = [0]

    # Precompute for each point i, the list of j>i such that triple_bad[i][j]
    # We'll do a standard B&B: try to include point i if possible.

    # Use bitmask for n<=20
    # We'll use a recursive function that goes over indices from `start` to n-1,
    # maintaining the set of chosen points as a bitmask, and a list of chosen indices.

    chosen = [0] * n  # 1 if chosen
    chosen_count = [0]

    def can_add(i):
        # check if adding i to current chosen set creates a bad triple
        # i is collinear with some pair (a,b) already chosen
        for a in range(n):
            if not chosen[a]:
                continue
            for b in range(a + 1, n):
                if not chosen[b]:
                    continue
                if triple_bad[a][b] and (a == i or b == i):
                    # actually we need triple (a,b,i) to be bad
                    pass
            # better: for each pair (a,b) in chosen, check if (a,b,i) is collinear
        # redo properly
        for a in range(n):
            if not chosen[a]:
                continue
            for b in range(a + 1, n):
                if not chosen[b]:
                    continue
                # check triple (a, b, i)
                if _collinear(points[a], points[b], points[i]):
                    return False
        return True

    def branch(start, size):
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        # upper bound: size + (n - start)
        if size + (n - start) <= best[0]:
            return
        # try to include start
        if can_add(start):
            chosen[start] = 1
            branch(start + 1, size + 1)
            chosen[start] = 0
        # try to exclude start
        branch(start + 1, size)

    branch(0, 0)
    return best[0]


def _valid(points):
    # distinct, in range, no four collinear
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    if _has_four_collinear(points):
        return False
    return True


def _construct_random(seed):
    rng = random.Random(seed)
    # generate random points and check
    for _ in range(200):
        pts = []
        used = set()
        while len(pts) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if not _has_four_collinear(pts):
            return pts
    return None


def _construct_parabola():
    # parabola y = x^2 gives no three collinear, but we need no four collinear (satisfied)
    # score is 20, which is high. We want lower score.
    # Instead, let's try to create a configuration with many collinear triples but no four.
    # Use a grid-like structure with controlled lines.
    pass


def _construct_from_lines():
    """
    Try to build a configuration with a moderate score.
    Strategy: use a set of lines, each with exactly 3 points, and ensure no 4 are collinear.
    This is like a 3-uniform hypergraph embedded in the plane.
    """
    # We'll try a specific known construction: take points from a 5x4 grid with some perturbations?
    # Or use the "no four in line" problem solutions.
    # For now, let's try a simple heuristic: start with a grid and perturb.
    # Actually, let's try to use a random search with local improvement.
    pass


def construct():
    global _START
    _START = time.time()
    
    # First, try a quick deterministic construction that might have a lower score.
    # Idea: use a 5x4 grid but shift some points to avoid 4-in-line.
    # A 5x4 grid has many 4-in-line (rows of 5, columns of 4).
    # Let's try a "bent" grid.
    
    # Try random constructions first, keep the best score found.
    best_points = None
    best_score = 999
    
    # Try a few deterministic patterns
    # Pattern 1: parabola (score 20, but valid)
    pts_parab = [[x, x * x] for x in range(-10, 10)]
    if _valid(pts_parab):
        s = _max_no_three_collinear(pts_parab)
        if s < best_score:
            best_score = s
            best_points = pts_parab
    
    # Try random constructions
    for seed in range(500):
        if _remaining() < 5.0:
            break
        pts = _construct_random(seed)
        if pts is None:
            continue
        # quick score estimate: just check if it's valid (already checked in construct_random)
        # compute score
        s = _max_no_three_collinear(pts)
        if s < best_score:
            best_score = s
            best_points = pts
            if best_score <= 12:  # good enough, stop early
                break
    
    # If we haven't found anything, fall back to parabola
    if best_points is None:
        best_points = pts_parab
    
    return best_points
