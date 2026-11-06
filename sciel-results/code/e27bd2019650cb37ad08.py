"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START

def _score(points):
    """Return (num_exact_4_lines, valid) for 16 points."""
    pts = [(p[0], p[1]) for p in points]
    if len(set(pts)) != 16:
        return 0, False
    for x, y in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return 0, False
    # Check no five collinear and count exact-4 lines
    # Group by line: for each pair, compute line key, but that's O(n^2) lines.
    # Better: for each pair, get the line, count points on it.
    # Use a dict from line-representation to set of points on that line.
    line_points = defaultdict(set)
    for i in range(16):
        for j in range(i + 1, 16):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            # Compute line in reduced form: ax + by + c = 0 with gcd normalization
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = _gcd3(a, b, c)
            if g != 0:
                a //= g
                b //= g
                c //= g
            # Normalize sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            line_points[(a, b, c)].add(i)
    
    num_exact_4 = 0
    for line_key, idx_set in line_points.items():
        n = len(idx_set)
        if n == 5:
            return 0, False  # five collinear
        if n == 4:
            num_exact_4 += 1
    return num_exact_4, True


def _gcd3(a, b, c):
    from math import gcd
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g


def _neighbors(points):
    """Generate random neighbor configurations by moving one or two points."""
    pts = [(p[0], p[1]) for p in points]
    n = len(pts)
    # Move one point to a random nearby location
    idx = random.randint(0, n - 1)
    x, y = pts[idx]
    dx = random.randint(-30, 30)
    dy = random.randint(-30, 30)
    if dx == 0 and dy == 0:
        dx = random.randint(1, 10)
    new_pts = list(pts)
    new_pts[idx] = (x + dx, y + dy)
    return new_pts


def _try_improve(points, max_iters=50000, deadline=None):
    """Simulated annealing / hill climbing to improve a configuration."""
    best_score, best_valid = _score(points)
    best_pts = [list(p) for p in points]
    cur_pts = [list(p) for p in points]
    cur_score, cur_valid = best_score, best_valid
    T = 50.0
    cooling = 0.9999
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        new_pts = _neighbors(cur_pts)
        new_score, new_valid = _score(new_pts)
        if not new_valid:
            new_score = -100
        
        if new_score > cur_score:
            cur_pts = new_pts
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_pts = [list(p) for p in cur_pts]
        else:
            # Accept with probability based on temperature
            delta = new_score - cur_score
            if T > 0.01:
                import math
                if random.random() < math.exp(delta / T):
                    cur_pts = new_pts
                    cur_score = new_score
        T *= cooling
    
    return best_pts, best_score


def _initial_grid():
    """4x4 grid as starting point."""
    return [[x, y] for x in range(4) for y in range(4)]


def _random_initial():
    """Generate a random valid-ish starting configuration."""
    pts = []
    while len(pts) < 16:
        x = random.randint(-50, 50)
        y = random.randint(-50, 50)
        if (x, y) not in pts:
            pts.append([x, y])
    return pts


def _refine_with_local_search(points, deadline):
    """Multiple restarts of local search."""
    best_pts = [list(p) for p in points]
    best_score, _ = _score(points)
    
    for restart in range(20):
        if deadline and time.time() > deadline - 5:
            break
        if restart == 0:
            cur = [list(p) for p in points]
        else:
            cur = _random_initial()
            # Try to fix validity
            for _ in range(100):
                s, v = _score(cur)
                if v:
                    break
                cur = _random_initial()
            if not v:
                continue
        
        cur_pts, cur_score = _try_improve(cur, max_iters=3000, deadline=deadline)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = cur_pts
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s
    
    # Start with 4x4 grid
    initial = _initial_grid()
    best_pts, best_score = _refine_with_local_search(initial, deadline)
    
    # Try a few more random starts if time permits
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
        rand_pts = _random_initial()
        # Quick check
        s, v = _score(rand_pts)
        if not v:
            continue
        cur_pts, cur_score = _try_improve(rand_pts, max_iters=2000, deadline=deadline)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = cur_pts
    
    # Final local polish
    if time.time() < deadline - 5:
        final_pts, final_score = _try_improve(best_pts, max_iters=10000, deadline=deadline)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    return [list(p) for p in best_pts]

# EVOLVE-BLOCK-END
