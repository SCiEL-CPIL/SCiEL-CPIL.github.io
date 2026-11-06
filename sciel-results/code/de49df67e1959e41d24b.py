"""Construct 12 integer points minimizing ordinary lines (lines with exactly 2 points)."""

import time
import itertools
import random

# Deadline guard: evaluator kills at 150s, use 120s limit with safety margin
_DEADLINE = 120.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed() - 2.0  # 2s safety margin

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Build a set of lines (as normalized tuples) and count points per line
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction vector
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                dx, dy = 0, 1
            elif dy == 0:
                dx, dy = 1, 0
            else:
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
            # Line key: (dx, dy, intercept) where line is dx*y - dy*x = c
            # c = dx*y1 - dy*x1
            c = dx * y1 - dy * x1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    # A line with exactly 2 points contributes 1 to the ordinary line count
    # But we need to be careful: line_counts counts pairs, not lines
    # Actually, we need to count distinct lines that have exactly 2 points
    # Let's redo: for each line, count how many points are on it
    # Better approach: use a dict mapping line -> set of point indices
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                dx, dy = 0, 1
            elif dy == 0:
                dx, dy = 1, 0
            else:
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0 or (dx == 0 and dy < 0):
                    dx, dy = -dx, -dy
            c = dx * y1 - dy * x1
            key = (dx, dy, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _evaluate(points):
    """Return (is_valid, ordinary_count)."""
    n = len(points)
    if n != 12:
        return False, 999
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False, 999
        seen.add(tuple(p))
    # Check bounds
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False, 999
    # Check not all collinear
    if _all_collinear(points):
        return False, 999
    return True, _count_ordinary_lines(points)

def _all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y0) != dy * (x - x0):
            return False
    return True

def _local_search(points, max_iters=5000):
    """Local search: try moving each point by small integer offsets."""
    best = list(points)
    best_score = _count_ordinary_lines(best)
    offsets = [(-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),
               (-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),
               (0,-2),(0,-1),(0,1),(0,2),
               (1,-2),(1,-1),(1,0),(1,1),(1,2),
               (2,-2),(2,-1),(2,0),(2,1),(2,2)]
    # Also try larger jumps
    for _ in range(5):
        offsets.append((random.randint(-5,5), random.randint(-5,5)))
    
    iters = 0
    while iters < max_iters and _remaining() > 1.0:
        iters += 1
        improved = False
        for i in range(12):
            for dx, dy in offsets:
                new_pts = list(best)
                new_pts[i] = [best[i][0] + dx, best[i][1] + dy]
                # Quick check: distinct and in bounds
                valid = True
                seen = set()
                for j, p in enumerate(new_pts):
                    if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                        valid = False
                        break
                    t = tuple(p)
                    if t in seen:
                        valid = False
                        break
                    seen.add(t)
                if not valid:
                    continue
                if _all_collinear(new_pts):
                    continue
                score = _count_ordinary_lines(new_pts)
                if score < best_score:
                    best = new_pts
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try random restarts within local search
            for _ in range(10):
                i = random.randint(0, 11)
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                new_pts = list(best)
                new_pts[i] = [best[i][0] + dx, best[i][1] + dy]
                seen = set()
                valid = True
                for p in new_pts:
                    if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                        valid = False
                        break
                    t = tuple(p)
                    if t in seen:
                        valid = False
                        break
                    seen.add(t)
                if not valid or _all_collinear(new_pts):
                    continue
                score = _count_ordinary_lines(new_pts)
                if score < best_score:
                    best = new_pts
                    best_score = score
                    improved = True
                    break
            if not improved:
                break
    return best, best_score

def _generate_initial_points():
    """Generate a diverse initial set of 12 points."""
    # Start with a 3x4 grid and perturb
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x + random.randint(-3, 3), y + random.randint(-3, 3)])
    # Ensure distinct
    while len(set(tuple(p) for p in pts)) < 12:
        i = random.randint(0, 11)
        pts[i] = [random.randint(-10, 10), random.randint(-10, 10)]
    return pts

def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    # Try multiple random initializations
    random.seed(42)
    num_restarts = 20
    for restart in range(num_restarts):
        if _remaining() < 5.0:
            break
        pts = _generate_initial_points()
        # Local search
        result, score = _local_search(pts, max_iters=3000)
        if score < best_score:
            best_score = score
            best_points = result
        if best_score <= 6:
            break
    
    if best_points is None:
        # Fallback: 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
