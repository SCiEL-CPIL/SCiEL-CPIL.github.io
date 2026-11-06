"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """Check if points are valid: 12 distinct, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    for i in range(2, len(points)):
        x, y = points[i]
        if A * x + B * y + C != 0:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_points = None
    best_count = float('inf')
    
    # Search all subsets of 12 points from a 4x4 grid
    # This covers the known optimal configurations for 12 points
    grid = [(i, j) for i in range(4) for j in range(4)]
    
    # There are C(16, 4) = 1820 ways to remove 4 points
    for removed in combinations(range(16), 4):
        if time.time() > deadline:
            break
        pts = [grid[i] for i in range(16) if i not in removed]
        if _is_valid(pts):
            cnt = _count_ordinary(pts)
            if cnt < best_count:
                best_count = cnt
                best_points = pts
            if best_count <= 6:
                return best_points
    
    # If not found in 4x4, try 3x4 grid (full) and its variants
    # 3x4 grid is just one specific case of 4x4 with 4 points removed (the 4th column)
    # But we already covered all 4x4 subsets.
    
    # Try other grid sizes if time permits
    # 2x6 grid
    grid2 = [(i, j) for i in range(6) for j in range(2)]
    # C(12, 12) = 1, just the full grid
    pts = grid2
    if _is_valid(pts):
        cnt = _count_ordinary(pts)
        if cnt < best_count:
            best_count = cnt
            best_points = pts
            
    # 6x2 grid
    grid3 = [(i, j) for i in range(2) for j in range(6)]
    pts = grid3
    if _is_valid(pts):
        cnt = _count_ordinary(pts)
        if cnt < best_count:
            best_count = cnt
            best_points = pts
            
    # Fallback to a known good configuration if search fails (should not happen)
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
