"""Rewritten program: combinatorial design + greedy local search for 12 rational points with few ordinary lines."""
import time
import random
from math import gcd

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Map each line (represented by a canonical (a,b,c) of ax+by+c=0) to the set of point indices on it.
    line_points = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            # Line through i and j: (y - yi)*(xj - xi) = (x - xi)*(yj - yi)
            # => (yj - yi)*x - (xj - xi)*y + (xj - xi)*yi - (yj - yi)*xi = 0
            a = yj - yi
            b = -(xj - xi)
            c = (xj - xi) * yi - (yj - yi) * xi
            # Canonicalize the line equation (a, b, c) so that the same line has the same key.
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
            if g:
                a //= g
                b //= g
                c //= g
            # Make (a, b) have a consistent sign
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    ordinary = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    n = len(points)
    if n < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for i in range(2, n):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _evaluate(points):
    """Return the number of ordinary lines, or None if invalid."""
    if len(points) != 12:
        return None
    # Check distinctness
    seen = set()
    for p in points:
        t = (p[0], p[1])
        if t in seen:
            return None
        seen.add(t)
    if not _not_all_collinear(points):
        return None
    return _count_ordinary_lines(points)

def _make_hexagon_grid():
    """Generate a set of 12 integer points based on a regular hexagon-like structure.
    We use a hexagonal lattice: points of the form (i, j) in axial coordinates,
    converted to Cartesian: x = i + j/2, y = j*sqrt(3)/2.
    To keep integer coordinates, we scale by 2: x = 2i + j, y = j*sqrt(3).
    But sqrt(3) is irrational, so we need a different approach.
    
    Alternative: use a rectangular grid with some perturbations to create collinearities.
    Let's use a 3x4 grid as a base and then apply local search.
    """
    # Start with a 3x4 grid
    return [[x, y] for x in range(4) for y in range(3)]

def _greedy_local_search(points, max_iter=50000):
    """Greedy local search: try swapping or perturbing points to reduce ordinary lines."""
    current = [list(p) for p in points]
    current_score = _evaluate(current)
    if current_score is None:
        return current, current_score
    
    best = [list(p) for p in current]
    best_score = current_score
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        # Try a random move: pick a random point and perturb it
        idx = random.randint(0, 11)
        x, y = current[idx]
        # Random perturbation
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        if dx == 0 and dy == 0:
            continue
        new_x = x + dx
        new_y = y + dy
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        
        old_point = current[idx]
        current[idx] = [new_x, new_y]
        new_score = _evaluate(current)
        
        if new_score is not None and new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            # Revert
            current[idx] = old_point
    
    return best, best_score

def _try_multiple_starts():
    """Try multiple random starting configurations and local search from each."""
    best_points = None
    best_score = float('inf')
    
    # Start with the 3x4 grid
    grid = _make_hexagon_grid()
    pts, score = _greedy_local_search(grid, max_iter=30000)
    if score is not None and score < best_score:
        best_score = score
        best_points = pts
    
    if _elapsed() > _DEADLINE:
        return best_points, best_score
    
    # Try a few random perturbations of the grid
    for restart in range(5):
        if _elapsed() > _DEADLINE:
            break
        random_pts = []
        for x in range(4):
            for y in range(3):
                rx = x + random.randint(-2, 2)
                ry = y + random.randint(-2, 2)
                random_pts.append([rx, ry])
        pts, score = _greedy_local_search(random_pts, max_iter=20000)
        if score is not None and score < best_score:
            best_score = score
            best_points = pts
    
    return best_points, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_points, best_score = _try_multiple_starts()
    
    if best_points is None:
        # Fallback: return the 3x4 grid
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points
