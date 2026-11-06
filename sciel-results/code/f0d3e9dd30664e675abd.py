"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    # Use a set of normalized line representations
    # A line through (x1,y1) and (x2,y2) can be represented by (a,b,c) where
    # a*x + b*y + c = 0, normalized so gcd(a,b,c)=1 and a>0 or (a==0 and b>0)
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - y2*x1
            # Normalize
            if a == 0 and b == 0:
                continue
            g = gcd(abs(a), abs(b))
            if g != 0:
                a //= g
                b //= g
                c //= g
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    # Count lines with exactly 2 points (i.e., the line was formed by exactly one pair)
    ordinary = 0
    for key, count in line_counts.items():
        if count == 1:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """Check if points are 12 distinct integer points with |x|,|y| <= 1e9, not all collinear."""
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
    # Check not all collinear
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return True
    return False

def _try_configurations(deadline):
    """Try known configurations and local search."""
    best_points = None
    best_count = float('inf')
    
    def update(points):
        nonlocal best_points, best_count
        if not _is_valid(points):
            return
        cnt = _count_ordinary(points)
        if cnt < best_count:
            best_count = cnt
            best_points = [list(p) for p in points]
    
    # Try 2x6 grid
    update([(x, y) for x in range(2) for y in range(6)])
    
    # Try 3x4 grid
    update([(x, y) for x in range(3) for y in range(4)])
    
    # Try 4x3 grid
    update([(x, y) for x in range(4) for y in range(3)])
    
    # Try 6x2 grid
    update([(x, y) for x in range(6) for y in range(2)])
    
    # Try points on a circle (integer points on x^2+y^2=r^2)
    # x^2 + y^2 = 25: (0,5),(5,0),(3,4),(4,3),(-3,4),(-4,3),(-3,-4),(-4,-3),(3,-4),(4,-3),(0,-5),(-5,0)
    circle_pts = [(0,5),(5,0),(3,4),(4,3),(-3,4),(-4,3),(-3,-4),(-4,-3),(3,-4),(4,-3),(0,-5),(-5,0)]
    update(circle_pts)
    
    # Try points on a parabola y = x^2 (but need 12 distinct integer points)
    # x in {-3,-2,-1,0,1,2,3,4,5,6,7,8} -> y = x^2
    parab_pts = [(x, x*x) for x in range(-3, 9)]
    update(parab_pts)
    
    # Try a 4-point line + 8 other points arranged to maximize collinearities
    # Start with 4 collinear, then add points to create more 3+ collinear sets
    
    # Local search from best found
    if best_points is not None:
        random.seed(42)
        current = [list(p) for p in best_points]
        current_count = best_count
        
        while time.time() < deadline - 2:  # safety margin
            if current_count == 6:
                break
            # Try random perturbation
            new_pts = [list(p) for p in current]
            idx = random.randint(0, 11)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_pts[idx][0] += dx
            new_pts[idx][1] += dy
            
            if not _is_valid(new_pts):
                continue
            
            new_count = _count_ordinary(new_pts)
            if new_count < current_count:
                current = new_pts
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_points = [list(p) for p in current]
    
    return best_points

def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    result = _try_configurations(deadline)
    if result is None:
        # Fallback: 3x4 grid
        return [[x, y] for x in range(4) for y in range(3)]
    return result

# EVOLVE-BLOCK-END
