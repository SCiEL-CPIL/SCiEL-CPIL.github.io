import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin

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
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

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
    return not _collinear_check(points)

def _line_through_two(p1, p2):
    """Get normalized line coefficients (A, B, C) for line through p1, p2."""
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

def _score_from_line_counts(line_counts):
    """Compute ordinary line count from precomputed line counts."""
    return sum(1 for c in line_counts.values() if c == 1)

def _build_line_counts(points):
    """Build line count dictionary for a set of points."""
    n = len(points)
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
    return line_counts

def _try_known_configs():
    """Try known good configurations and their variants."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 12 in a line with perturbations - not valid, skip
    
    # Cubic curve points: (t, t^2) for t in 0..11
    configs.append([[t, t*t] for t in range(12)])
    
    # Parabola: (t, t^2) for t in -5..4
    configs.append([[t, t*t] for t in range(-5, 5)])
    
    # 3x3 grid + 3 extra points
    base = [[x, y] for x in range(3) for y in range(3)]
    # Add points that create more collinearities
    configs.append(base + [[3, 0], [3, 1], [3, 2]])  # 4x3 grid
    configs.append(base + [[0, 3], [1, 3], [2, 3]])  # 3x4 grid
    
    # 2x3x2 pattern
    configs.append([[x, y] for x in range(2) for y in range(3)] + [[x+3, y] for x in range(2) for y in range(3)])
    
    # Try to find a configuration with 6 ordinary lines
    # Known: 12 points can have as few as 6 ordinary lines over reals
    # Let's try a configuration based on a regular polygon or other symmetric shapes
    
    # Points on two lines: 6 on each, but not all collinear
    # Line 1: (0,0) to (5,0), Line 2: (0,1) to (5,1)
    configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # 3 lines with 4 points each
    # Line 1: horizontal, Line 2: vertical, Line 3: diagonal
    configs.append([[0, 0], [1, 0], [2, 0], [3, 0],
                    [0, 1], [0, 2], [0, 3],
                    [1, 1], [2, 2], [3, 3],
                    [4, 4], [5, 5]])
    
    # Pappus-like configuration: 3 lines with 4 points, arranged to create more collinearities
    # Points at intersections of 4 lines in general position (6 intersection points) + 6 more
    # Actually, let's try: 3 concurrent lines with 4 points each (12 points, 3 shared at center)
    # That gives 12 - 3 + 3 = 12 distinct points if we count properly
    # Line 1: (0,0), (1,0), (2,0), (3,0)
    # Line 2: (0,0), (0,1), (0,2), (0,3) 
    # Line 3: (0,0), (1,1), (2,2), (3,3)
    # But that's only 1 + 3 + 3 + 3 = 10 distinct points (origin shared)
    # Need 12, so add 2 more
    
    # Better: 4 lines, each with 3 points, arranged so points are shared
    # Configuration: complete quadrilateral + extra points
    # Let's try: 3 lines forming a triangle, with 4 points on each line
    # Line AB: (0,0), (1,0), (2,0), (3,0)
    # Line BC: (3,0), (3,1), (3,2), (3,3)
    # Line CA: (0,3), (1,3), (2,3), (0,0) -- wait, need to be careful
    # Actually: triangle vertices (0,0), (3,0), (0,3)
    # Side 1: (0,0), (1,0), (2,0), (3,0)
    # Side 2: (3,0), (3,1), (3,2), (3,3)
    # Side 3: (0,3), (1,3), (2,3), (0,0) -- but (0,0) already counted
    # Distinct: (0,0),(1,0),(2,0),(3,0),(3,1),(3,2),(3,3),(0,3),(1,3),(2,3) = 10 points
    # Need 2 more: add (1,1) and (2,2) which are on diagonal
    configs.append([[0,0],[1,0],[2,0],[3,0],[3,1],[3,2],[3,3],[0,3],[1,3],[2,3],[1,1],[2,2]])
    
    # Another: 2x2 grid of 3-point lines
    # Lines: y=0: (0,0),(1,0),(2,0); y=1: (0,1),(1,1),(2,1); y=2: (0,2),(1,2),(2,2)
    # x=0: (0,0),(0,1),(0,2); x=1: (1,0),(1,1),(1,2); x=2: (2,0),(2,1),(2,2)
    # That's 9 points. Add 3 more: (3,0),(3,1),(3,2)
    configs.append([[x,y] for x in range(3) for y in range(3)] + [[3,0],[3,1],[3,2]])
    
    # 3x3 grid + diagonal points
    configs.append([[x,y] for x in range(3) for y in range(3)] + [[3,3],[4,4],[5,5]])
    
    # Hexagonal-like arrangement
    configs.append([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[3,1],[3,2],[3,3]])
    
    return configs

def _local_search_sa(points, max_iters=200000, start_temp=10.0, end_temp=0.1):
    """Simulated annealing local search to minimize ordinary lines."""
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_score = cur_score
    
    # Build a fixed candidate pool once: neighborhood + line extensions
    pool = set()
    for p in cur:
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if dx == 0 and dy == 0:
                    continue
                pool.add((p[0] + dx, p[1] + dy))
    # Extend lines through pairs
    for i in range(12):
        for j in range(i + 1, 12):
            x1, y1 = cur[i]
            x2, y2 = cur[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dx //= g
                dy //= g
            for k in range(-4, 5):
                if k == 0:
                    continue
                pool.add((x1 + k*dx, y1 + k*dy))
                pool.add((x2 + k*dx, y2 + k*dy))
    pool_list = list(pool)
    
    iters = 0
    rng = random.Random(42)
    cur_set = set(tuple(p) for p in cur)
    
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        temp = start_temp * (end_temp / start_temp) ** (iters / max_iters)
        
        idx = rng.randint(0, 11)
        old = cur[idx]
        
        # Try several random candidates
        n_try = min(10, len(pool_list))
        candidates = rng.sample(pool_list, n_try)
        improved = False
        
        for cand in candidates:
            if cand in cur_set:
                continue
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            
            delta = new_score - cur_score
            if delta < 0 or (temp > 0 and rng.random() < min(1.0, 1.0 / (1.0 + delta / temp))):
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                if cur_score < best_score:
                    best = [p[:] for p in cur]
                    best_score = cur_score
                break
    
    return best, best_score

def _construct_from_lines():
    """Try to construct configurations from line arrangements."""
    configs = []
    
    # Try: 4 lines, each containing 3 points, with points shared between lines
    # This is like a finite projective plane configuration
    
    # Configuration: 3 lines through a common point (star) + 3 more lines
    # Center: (0,0)
    # Line 1 (horizontal): (0,0), (1,0), (2,0)
    # Line 2 (vertical): (0,0), (0,1), (0,2)
    # Line 3 (diagonal): (0,0), (1,1), (2,2)
    # That's 1 + 2 + 2 + 2 = 7 points. Need 5 more.
    # Add: (3,0), (0,3), (3,3), (1,2), (2,1)
    # Check if (1,2) and (2,1) create new collinearities
    configs.append([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[2,2],[3,0],[0,3],[3,3],[1,2],[2,1]])
    
    # Configuration: complete quadrilateral
    # 4 lines in general position create 6 intersection points
    # Add 6 more points to get 12
    # Lines: x=0, x=1, y=0, y=1 -> intersections: (0,0),(0,1),(1,0),(1,1)
    # That's only 4 points. Need more lines.
    
    # Try: 3 parallel lines with 4 points each, plus 3 more
    # Line y=0: (0,0),(1,0),(2,0),(3,0)
    # Line y=1: (0,1),(1,1),(2,1),(3,1)
    # Line y=2: (0,2),(1,2),(2,2),(3,2)
    # That's 12 points - it's the 4x3 grid
    configs.append([[x,y] for x in range(4) for y in range(3)])
    
    # Try: 2 lines with 6 points each (parallel)
    configs.append([[x,0] for x in range(6)] + [[x,1] for x in range(6)])
    
    # Try: configuration based on 12th roots of unity projected
    # Not integer, so skip
    
    # Try: 3x4 grid with shifted columns
    configs.append([[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]])
    
    # Try: non-uniform grid
    configs.append([[0,0],[1,0],[3,0],[6,0],[0,1],[1,1],[3,1],[6,1],[0,2],[1,2],[3,2],[6,2]])
    
    return configs

def construct():
    best_pts = None
    best_score = float('inf')
    
    # Try known configurations
    all_configs = _try_known_configs() + _construct_from_lines()
    for cfg in all_configs:
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_pts = [p[:] for p in cfg]
    
    # If we haven't found a good config, use 4x3 grid as fallback
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Run simulated annealing from best config
    if time.time() < _DEADLINE - 10:
        pts, score = _local_search_sa(best_pts, max_iters=300000)
        if score < best_score:
            best_pts = pts
            best_score = score
    
    # Run SA from other good starting points if time allows
    if time.time() < _DEADLINE - 15:
        for cfg in all_configs[:12]:
            if not _valid(cfg):
                continue
            if time.time() >= _DEADLINE - 10:
                break
            pts, score = _local_search_sa(cfg, max_iters=80000)
            if score < best_score:
                best_pts = pts
                best_score = score
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
