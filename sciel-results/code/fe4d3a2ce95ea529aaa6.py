"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _count_ordinary(pts):
    """Count lines containing exactly 2 of the given points."""
    n = len(pts)
    if n < 2:
        return 0
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c))
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Canonical form: first nonzero of (a,b) positive
            if a < 0 or (a == 0 and b < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for v in line_map.values() if len(v) == 2)

def _is_valid(pts):
    """Check 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(pts) != 12:
        return False
    seen = set()
    for p in pts:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        key = (p[0], p[1])
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    if len(pts) < 3:
        return True
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    for i in range(2, len(pts)):
        xi, yi = pts[i]
        if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
            return True
    return False

def _generate_triangle_config():
    """
    Generate a configuration of 12 points on 3 lines forming a triangle.
    5 points on each line, sharing 3 vertices.
    Vertices: (0,0), (10,0), (0,10)
    Line 1 (y=0): (0,0), (2,0), (4,0), (6,0), (8,0), (10,0) -> 6 points?
    We need 5 points per line including vertices.
    Vertices: A=(0,0), B=(10,0), C=(0,10)
    Line AB: A, P1, P2, P3, B -> 5 points.
    Line BC: B, Q1, Q2, Q3, C -> 5 points.
    Line CA: C, R1, R2, R3, A -> 5 points.
    Total points = 5+5+5 - 3 (shared) = 12.
    """
    A = (0, 0)
    B = (10, 0)
    C = (0, 10)
    
    pts = set()
    pts.add(A)
    pts.add(B)
    pts.add(C)
    
    # Add 2 points on AB (excluding A, B)
    # Use integer coordinates
    pts.add((2, 0))
    pts.add((4, 0))
    # Wait, I need 3 internal points per side to make 5 total per side?
    # Side AB: A, P1, P2, P3, B.
    # My current set: A, B. I added (2,0), (4,0). That's 4 points.
    # I need one more on AB.
    pts.add((6, 0))
    
    # Add 3 internal points on BC
    # B=(10,0), C=(0,10). Line x+y=10.
    pts.add((8, 2))
    pts.add((6, 4))
    pts.add((4, 6))
    
    # Add 3 internal points on CA
    # C=(0,10), A=(0,0). Line x=0.
    pts.add((0, 2))
    pts.add((0, 4))
    pts.add((0, 6))
    
    # Check count
    if len(pts) != 12:
        return []
        
    return [list(p) for p in pts]

def _generate_conic_config():
    """
    Approximate Böröczky configuration.
    6 points on y=0, 6 points on y=x^2.
    """
    pts = set()
    # 6 points on x-axis
    for x in range(1, 7):
        pts.add((x, 0))
    # 6 points on parabola y=x^2
    # Avoid x=0 to keep distinct from origin if used, but origin is not in set above.
    # Use negative x for parabola to separate visually?
    for x in range(-6, 0):
        pts.add((x, x*x))
        
    if len(pts) != 12:
        return []
        
    return [list(p) for p in pts]

def _generate_grid_config():
    """
    4x3 grid.
    """
    pts = []
    for x in range(4):
        for y in range(3):
            pts.append([x, y])
    return pts

def construct():
    deadline = time.time() + 140
    
    candidates = []
    
    # Add deterministic candidates
    tri = _generate_triangle_config()
    if tri:
        candidates.append(tri)
        
    conic = _generate_conic_config()
    if conic:
        candidates.append(conic)
        
    grid = _generate_grid_config()
    if grid:
        candidates.append(grid)
        
    # Evaluate candidates
    best_pts = None
    best_score = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        score = _count_ordinary(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
            
    # If we haven't found a good score, do a brief targeted search
    # but primarily rely on the deterministic structures.
    # The triangle config might have more than 6 ordinary lines.
    # Let's check if we can improve the triangle config by shifting points.
    # For now, return the best deterministic candidate.
    
    if best_pts is None:
        # Fallback to grid
        best_pts = grid
        best_score = _count_ordinary(grid)
        
    return [list(p) for p in best_pts]
# EVOLVE-BLOCK-END
