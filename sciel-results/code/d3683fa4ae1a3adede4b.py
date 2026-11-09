import time
import itertools
from collections import defaultdict

_DEADLINE = 115.0  # safety margin under 120s (and 150s kill)

def _line_key(p, q):
    # canonical integer line representation (A, B, C) for Ax + By + C = 0
    x1, y1 = p
    x2, y2 = q
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    g = _gcd3(A, B, C)
    A //= g; B //= g; C //= g
    # canonical sign
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _gcd3(a, b, c):
    return _gcd(_gcd(a, b), c)

def _count_ordinary(points):
    # points: list of (x, y)
    n = len(points)
    if n < 3:
        return n
    line_pts = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            line_pts[k].append((i, j))
    # For each line key, collect the set of point indices
    line_set = defaultdict(set)
    for k, pairs in line_pts.items():
        for i, j in pairs:
            line_set[k].add(i)
            line_set[k].add(j)
    ordinary = 0
    for k, s in line_set.items():
        if len(s) == 2:
            ordinary += 1
    return ordinary

def construct():
    t0 = time.time()
    best_points = None
    best_score = None

    # Build catalog on a small grid: lines with >= 3 integer points within range
    # Use a grid that's large enough to have many 3+ collinear lines but small
    # enough to enumerate pairs quickly.
    # Grid: symmetric around origin, side length ~ 7 (coords -3..3) gives 49 points.
    # That's 49*48/2 = 1176 pairs, manageable.
    # But we want lines with many points to be useful. Let's use coords -4..4 (9x9=81 pts).
    # 81*80/2 = 3240 pairs. Still fine.
    # Actually, let's go a bit larger to get lines with 4, 5 points: -5..5 (11x11=121 pts).
    # 121*120/2 = 7260 pairs. Fine.

    half = 5
    pts = [(x, y) for x in range(-half, half + 1) for y in range(-half, half + 1)]
    n_pts = len(pts)

    # Build line -> set of point indices
    line_to_indices = defaultdict(set)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            k = _line_key(pts[i], pts[j])
            line_to_indices[k].add(i)
            line_to_indices[k].add(j)

    # Filter to lines with >= 3 points
    good_lines = {k: s for k, s in line_to_indices.items() if len(s) >= 3}

    # Precompute for each point, the list of good lines it's on
    pt_lines = defaultdict(list)
    for k, s in good_lines.items():
        for idx in s:
            pt_lines[idx].append(k)

    # Strategy: greedy + local search.
    # Start with a random or structured initial set, then try to swap points
    # to reduce ordinary line count.

    import random
    rng = random.Random(42)

    # Initial: pick points that are on many good lines (high "line degree")
    # Sort points by number of good lines they participate in
    point_degree = [len(pt_lines[i]) for i in range(n_pts)]
    # Indices sorted by degree descending
    sorted_indices = sorted(range(n_pts), key=lambda i: -point_degree[i])

    # Greedy initial: pick top points, but ensure they form good lines together
    # Start with the first point, then greedily add points that share lines with existing
    initial = [sorted_indices[0]]
    for idx in sorted_indices[1:]:
        if len(initial) >= 12:
            break
        # Check if this point shares a good line with at least one existing point
        shares = False
        for line_k in pt_lines[idx]:
            for existing in initial:
                if existing in good_lines[line_k]:
                    shares = True
                    break
            if shares:
                break
        if shares or len(initial) < 3:
            initial.append(idx)
        # If we can't reach 12 with sharing, just take high-degree points
        if len(initial) < 12 and not shares:
            initial.append(idx)

    # Ensure we have exactly 12 distinct points
    initial = initial[:12]
    if len(initial) < 12:
        for idx in sorted_indices:
            if idx not in initial:
                initial.append(idx)
            if len(initial) == 12:
                break

    # Convert to point tuples
    current = [pts[i] for i in initial]
    current_score = _count_ordinary(current)

    # Local search: try swapping each point with another candidate to reduce score
    improved = True
    while improved and (time.time() - t0) < _DEADLINE:
        improved = False
        for i in range(12):
            if (time.time() - t0) > _DEADLINE:
                break
            # Try replacing current[i] with each other point in catalog
            for j in range(n_pts):
                if pts[j] in current:
                    continue
                new_current = current[:i] + [pts[j]] + current[i+1:]
                # Quick check: not all collinear
                # Compute score
                new_score = _count_ordinary(new_current)
                if new_score < current_score:
                    current = new_current
                    current_score = new_score
                    improved = True
                    break  # restart inner loop with new current
            if improved:
                break  # restart outer loop

    # Convert to list of [x, y]
    result = [[x, y] for x, y in current]

    # Safety: verify distinctness and bounds
    assert len(set(map(tuple, result))) == 12
    assert all(abs(x) <= 10**9 and abs(y) <= 10**9 for x, y in result)
    # Not all collinear
    assert _count_ordinary(result) != 12  # if all collinear, every line has >=3, ordinary=0? No.
    # Actually if all 12 collinear, there's 1 line with 12 points, ordinary = 0.
    # But task says "not all collinear". Let's verify.
    # Check if all on one line
    if len(result) >= 3:
        k0 = _line_key(result[0], result[1])
        if all(_line_key(result[0], p) == k0 for p in result[1:]):
            # All collinear, need to fix
            # Perturb one point
            result[11] = [result[11][0] + 1, result[11][1]]
            # Re-check
            # This might not be in our catalog, but it's a valid integer point

    return result
