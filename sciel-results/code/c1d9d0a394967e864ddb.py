import time
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (1, 0, -x1)
    if dy == 0:
        return (0, 1, -y1)
    g = _gcd(dx, dy)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    g2 = _gcd(dy, dx)
    if g2 > 1:
        dy //= g2
        dx //= g2
        c //= g2
    return (dy, -dx, c)

def _count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return sum(1 for pts in lines.values() if len(pts) == 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = float('inf')
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
    # Strategy: Exhaustive search over 12-point subsets of the 4x4 grid
    # The 4x4 grid is known to contain highly structured configurations.
    # If the optimal configuration (6 ordinary lines) is a subset of the 4x4 grid,
    # this will find it.
    
    grid_4x4 = [[x, y] for x in range(4) for y in range(4)]
    
    # Enumerate all 12-point subsets (equivalent to choosing 4 points to remove)
    for removed in combinations(range(16), 4):
        if time.time() > deadline:
            break
        
        removed_set = set(removed)
        subset = [grid_4x4[i] for i in range(16) if i not in removed_set]
        
        # Check distinctness (guaranteed by construction)
        # Check not all collinear (4x4 grid subsets are rarely collinear)
        # Quick check: if all points are on one line, skip
        # But 12 points from 4x4 cannot be collinear (max 4 per line)
        
        score = _count_ordinary(subset)
        update_best(subset, score)
        
        if best_score == 6:
            # Theoretical minimum reached
            break
            
    # If we didn't find 6, try other grids or configurations
    # But 4x4 is the most promising for high collinearity.
    
    # Fallback: 3x4 grid and its shears (from original code)
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    update_best(grid_3x4, _count_ordinary(grid_3x4))
    
    for k in range(-5, 6):
        if time.time() > deadline: break
        pts = [[x + k*y, y] for x in range(4) for y in range(3)]
        if len(set(map(tuple, pts))) != 12: continue
        s = _count_ordinary(pts)
        update_best(pts, s)
        
    return best_points
