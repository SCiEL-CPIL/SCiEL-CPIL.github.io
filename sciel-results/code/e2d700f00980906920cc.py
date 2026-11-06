"""Structural change: constraint-satisfaction greedy construction of 6 disjoint collinear triples
plus 2 singleton points, then local-search refinement to minimize the largest 3-collinear-free subset.
This replaces the previous no-collinear configuration (score 20) with one that has many forced
collinearities, aiming for score ~14."""
import random
import time
from math import gcd

DEADLINE = time.time() + 130  # safety margin before 150s kill

def _collinear(p1, p2, p3):
    """Check if three points are collinear using cross product."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) == (p2[1] - p1[1]) * (p3[0] - p1[0])

def _line_key(p1, p2):
    """Get a normalized key for the line through p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx, dy = dx // g, dy // g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    # Line equation: dx*(y - p1[1]) = dy*(x - p1[0])
    # => dx*y - dy*x = dx*p1[1] - dy*p1[0]
    c = dx * p1[1] - dy * p1[0]
    g2 = gcd(abs(dx), abs(dy))
    # Already normalized by g, but c might have common factor with dx,dy? No, c is derived.
    # Actually we need to normalize (dx, dy, c) by gcd(dx, dy, c) for uniqueness.
    gc = gcd(gcd(abs(dx), abs(dy)), abs(c))
    if gc > 1:
        dx, dy, c = dx // gc, dy // gc, c // gc
    return (dx, dy, c)

def _no_four_collinear(points):
    """Check that no line contains 4 or more of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # Each pair contributes to one line; a line with k points has C(k,2) pairs
    for key, count in line_counts.items():
        # count = C(k,2) => k*(k-1)/2 = count => k = (1 + sqrt(1+8*count))/2
        k_sq = 1 + 8 * count
        import math
        k_int = math.isqrt(k_sq)
        if k_int * k_int != k_sq:
            # Not a perfect square, shouldn't happen for valid combinatorics
            continue
        k = (1 + k_int) // 2
        if k >= 4:
            return False
    return True

def _max_independent_set_3collinear(points):
    """Find the largest subset with no three collinear via branch and bound.
    Points: list of (x,y) tuples. Returns size of largest 3-collinear-free subset."""
    n = len(points)
    # Precompute which triples are collinear
    triple_collinear = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    triple_collinear[i][j] = True
                    triple_collinear[i][k] = True
                    triple_collinear[j][k] = True

    # Branch and bound to find max subset with no 3 collinear
    # A subset is valid if no 3 of its points are collinear.
    # We want the maximum size.
    
    # Use recursive search with pruning
    best = [0]
    
    # Precompute for each point, which other points are "dangerous" (collinear with a third in some context)
    # Actually, simpler: state is the set of chosen points. When adding a new point, 
    # check if it creates a collinear triple with any two already chosen.
    
    # Sort points to get deterministic ordering
    # Use a heuristic: process points in order of how many collinear triples they participate in (descending)
    collinear_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    collinear_count[i] += 1
                    collinear_count[j] += 1
                    collinear_count[k] += 1
    
    # Order: points with more collinear triples first (harder to place)
    order = sorted(range(n), key=lambda i: -collinear_count[i])
    order_set = set(order)
    pos_in_order = {order[i]: i for i in range(n)}
    
    # Recursive B&B
    import sys
    sys.setrecursionlimit(10000)
    
    def bnb(idx, chosen, size):
        if time.time() > DEADLINE:
            return
        if size + (n - idx) <= best[0]:
            return
        if idx == n:
            if size > best[0]:
                best[0] = size
            return
        
        # Option 1: don't choose point order[idx]
        bnb(idx + 1, chosen, size)
        
        # Option 2: choose point order[idx] if valid
        pt = order[idx]
        # Check if adding pt to chosen creates a collinear triple
        # Need to check all pairs in chosen: for any pair (a,b) in chosen, is pt collinear with a,b?
        # Also need to check pairs where pt is one of the three.
        valid = True
        chosen_list = list(chosen)
        for a in chosen_list:
            for b in chosen_list:
                if a >= b:
                    continue
                if _collinear(points[a], points[b], points[pt]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            chosen.add(pt)
            bnb(idx + 1, chosen, size + 1)
            chosen.discard(pt)
    
    bnb(0, set(), 0)
    return best[0]

def _build_initial_config(seed):
    """Build a configuration of 20 points with 6 disjoint collinear triples + 2 singletons.
    Returns None if failed."""
    rng = random.Random(seed)
    
    # Candidate pool: all integer points with |x|,|y| <= 100
    # To make it manageable, use a subset
    pool = []
    for x in range(-50, 51):
        for y in range(-50, 51):
            pool.append((x, y))
    
    used = set()
    triples = []
    
    for _ in range(6):
        # Pick a random line by choosing two random points
        found = False
        for attempt in range(200):
            p1 = rng.choice(pool)
            p2 = rng.choice(pool)
            if p1 == p2:
                continue
            # Find all points on this line within the pool
            line_pts = []
            for p in pool:
                if _collinear(p1, p2, p):
                    line_pts.append(p)
            if len(line_pts) < 3:
                continue
            # Shuffle and pick 3 disjoint from used
            rng.shuffle(line_pts)
            triple = []
            for p in line_pts:
                if p not in used:
                    triple.append(p)
                    if len(triple) == 3:
                        break
            if len(triple) == 3:
                # Check no-four-collinear constraint
                candidate = list(used) + triple
                if _no_four_collinear(candidate):
                    triples.append(triple)
                    used.update(triple)
                    found = True
                    break
        if not found:
            return None
    
    # Add 2 more points
    remaining = [p for p in pool if p not in used]
    rng.shuffle(remaining)
    points = []
    for t in triples:
        points.extend(t)
    
    for p in remaining:
        candidate = points + [p]
        if _no_four_collinear(candidate):
            points.append(p)
            if len(points) == 20:
                break
    
    if len(points) != 20:
        return None
    
    return points

def _local_search(points, rng, time_limit):
    """Try to improve the score by swapping points."""
    current_points = list(points)
    current_score = _max_independent_set_3collinear(current_points)
    
    while time.time() < time_limit:
        improved = False
        # Try random swaps
        for _ in range(500):
            if time.time() > time_limit:
                break
            i = rng.randrange(20)
            # Pick a replacement point from a broader range
            new_pt = (rng.randint(-100, 100), rng.randint(-100, 100))
            if new_pt in current_points:
                continue
            candidate = list(current_points)
            candidate[i] = new_pt
            if not _no_four_collinear(candidate):
                continue
            # Quick check: does this reduce the score?
            # Full score computation is expensive, so only do it if we have time
            new_score = _max_independent_set_3collinear(candidate)
            if new_score < current_score:
                current_points = candidate
                current_score = new_score
                improved = True
                break
        if not improved:
            # Try 2-swaps
            for _ in range(100):
                if time.time() > time_limit:
                    break
                i = rng.randrange(20)
                j = rng.randrange(20)
                if i == j:
                    continue
                new_pt1 = (rng.randint(-100, 100), rng.randint(-100, 100))
                new_pt2 = (rng.randint(-100, 100), rng.randint(-100, 100))
                if new_pt1 in current_points or new_pt2 in current_points:
                    continue
                if new_pt1 == new_pt2:
                    continue
                candidate = list(current_points)
                candidate[i] = new_pt1
                candidate[j] = new_pt2
                if not _no_four_collinear(candidate):
                    continue
                new_score = _max_independent_set_3collinear(candidate)
                if new_score < current_score:
                    current_points = candidate
                    current_score = new_score
                    improved = True
                    break
        if not improved:
            # Reset and try a new initial config if time permits
            break
    
    return current_points, current_score

def construct():
    start_time = time.time()
    best_points = None
    best_score = 21  # higher than max possible
    
    # Try multiple seeds
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        
        points = _build_initial_config(seed)
        if points is None:
            continue
        
        # Check validity
        if len(points) != 20:
            continue
        if len(set(points)) != 20:
            continue
        if not all(abs(x) <= 100 and abs(y) <= 100 for x, y in points):
            continue
        if not _no_four_collinear(points):
            continue
        
        score = _max_independent_set_3collinear(points)
        if score < best_score:
            best_score = score
            best_points = list(points)
        
        if best_score <= 13:
            break
    
    if best_points is None:
        # Fallback: try a fixed known configuration
        # 6 lines with 3 points each
        # Line 1: y = 0, x = -10, -5, 0
        # Line 2: y = 10, x = 5, 10, 15
        # Line 3: y = -10, x = -15, -20, -25
        # Line 4: x = 20, y = -5, 0, 5
        # Line 5: x = -30, y = -25, -20, -15
        # Line 6: y = 2x, points (0,0) is already used... 
        # Let me just return a simple valid config
        fallback = [
            (-10, 0), (-5, 0), (0, 0),
            (5, 10), (10, 10), (15, 10),
            (-15, -10), (-20, -10), (-25, -10),
            (20, -5), (20, 0), (20, 5),
            (-30, -25), (-30, -20), (-30, -15),
            (30, 20), (30, 25),
            (40, 30), (40, 35),
        ]
        if _no_four_collinear(fallback):
            return fallback
        # Last resort
        return [[x, x * x] for x in range(-10, 10)]
    
    # Local search if time permits
    remaining = DEADLINE - time.time()
    if remaining > 20:
        rng = random.Random(42)
        improved_points, improved_score = _local_search(best_points, rng, time.time() + remaining - 5)
        if improved_score < best_score:
            best_points = improved_points
            best_score = improved_score
    
    return [[x, y] for x, y in best_points]
