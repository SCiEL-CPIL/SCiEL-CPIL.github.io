"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from fractions import Fraction

_start_time = time.time()
TIME_LIMIT = 140  # Leave 10s margin before 150s kill

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Use a set of line signatures: each line identified by normalized (a,b,c) for ax+by+c=0
    # For two points (x1,y1),(x2,y2): line is (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
    # Normalize by gcd and sign
    from math import gcd
    lines = {}  # line_key -> count of points on it
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            # Normalize
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g > 0:
                a //= g
                b //= g
                c //= g
            # Ensure consistent sign: make first non-zero positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = 2
            else:
                lines[key] += 1
    # Count lines with exactly 2 points
    ordinary = sum(1 for cnt in lines.values() if cnt == 2)
    return ordinary

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if not isinstance(p[0], int) or not isinstance(p[1], int):
            return False
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
        if p in seen:
            return False
        seen.add(p)
    # Check not all collinear
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    cross = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
    if cross == 0:
        # Check if all are collinear
        for i in range(3, 12):
            xi, yi = points[i]
            cross_i = (x1-x0)*(yi-y0) - (y1-y0)*(xi-x0)
            if cross_i != 0:
                return True
        return False
    return True

def _try_grid():
    """3x4 grid."""
    points = [[x, y] for x in range(4) for y in range(3)]
    return points

def _try_pappus_like():
    """Try a Pappus-inspired configuration for 12 points."""
    # Use points on two lines plus some structure
    # Actually, let's try a 2x6 arrangement or other
    pass

def _try_configurations():
    """Try several hand-crafted configurations and return the best."""
    candidates = []
    
    # 1. 4x3 grid
    pts1 = [[x, y] for x in range(4) for y in range(3)]
    candidates.append(pts1)
    
    # 2. 3x4 grid (transposed)
    pts2 = [[x, y] for x in range(3) for y in range(4)]
    candidates.append(pts2)
    
    # 3. Try points on a parabola: (i, i^2) for i in some range, plus extras
    #    But parabola points: any 3 determine a parabola, so lines through 2 parabola points
    #    may not contain a 3rd. Let's try (i, i^2) for i=0..5 and add 6 more points strategically
    
    # 4. Try a hexagonal/symmetric arrangement
    #    Regular hexagon vertices + center + ... but that's 7. 
    #    Two concentric hexagons?
    #    Hex 1: (1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1) - not quite regular
    #    Let's use actual regular-ish hexagon: (2,0),(1,sqrt3)... no, need integers.
    #    Integer hexagon: (2,0),(1,1),(-1,1),(-2,0),(-1,-1),(1,-1)
    hex1 = [(2,0),(1,1),(-1,1),(-2,0),(-1,-1),(1,-1)]
    hex2 = [(4,0),(2,2),(-2,2),(-4,0),(-2,-2),(2,-2)]
    pts3 = [list(p) for p in hex1 + hex2]
    candidates.append(pts3)
    
    # 5. Try a 2x6 grid
    pts4 = [[x, y] for x in range(6) for y in range(2)]
    candidates.append(pts4)
    
    # 6. Try points forming a "star" or other pattern
    #    Center + 11 around it
    center = [0, 0]
    # 11 points on a circle-ish integer pattern
    ring1 = [(3,0),(0,3),(-3,0),(0,-3),(2,2),(-2,2),(-2,-2),(2,-2),(1,3),(-1,3),(-1,-3)]
    # That's 11, total 12
    pts5 = [center] + [list(p) for p in ring1]
    candidates.append(pts5)
    
    # 7. Try a more structured approach: 3 groups of 4 on 3 lines
    #    Line y=0: (0,0),(1,0),(2,0),(3,0)
    #    Line y=1: (0,1),(1,1),(2,1),(3,1)  
    #    Line y=2: (0,2),(1,2),(2,2),(3,2)
    #    This is the 4x3 grid again.
    
    # 8. Try: 4 points on each of 3 concurrent lines (like a star)
    #    Lines through origin with slopes 0, 1, -1, and infinity? No, 3 lines.
    #    Line 1 (y=0): (-2,0),(-1,0),(1,0),(2,0)
    #    Line 2 (x=0): (0,-2),(0,-1),(0,1),(0,2)
    #    Line 3 (y=x): (-2,-2),(-1,-1),(1,1),(2,2)
    #    That's 12 points, 3 lines with 4 points each (plus origin not included to avoid duplicates)
    #    Wait, origin would be on all 3. Let's exclude origin.
    pts6 = [
        [-2,0],[-1,0],[1,0],[2,0],
        [0,-2],[0,-1],[0,1],[0,2],
        [-2,-2],[-1,-1],[1,1],[2,2]
    ]
    candidates.append(pts6)
    
    # 9. Try 3 lines with 4 points each, not concurrent
    #    Line y=0: (0,0),(1,0),(2,0),(3,0)
    #    Line y=1: (0,1),(1,1),(2,1),(3,1)
    #    Line x=0: (0,2),(0,3),(0,4),(0,5) -- but (0,0) and (0,1) already used
    #    Need distinct points. Let me use:
    #    Line y=0: (4,0),(5,0),(6,0),(7,0)
    #    Line y=1: (4,1),(5,1),(6,1),(7,1)
    #    Line x=0: (0,0),(0,1),(0,2),(0,3)
    pts7 = [
        [4,0],[5,0],[6,0],[7,0],
        [4,1],[5,1],[6,1],[7,1],
        [0,0],[0,1],[0,2],[0,3]
    ]
    candidates.append(pts7)
    
    # 10. Try a "Pascal" or other projective configuration
    #     For now, let's also try random local search
    
    best_pts = None
    best_count = float('inf')
    
    for pts in candidates:
        if not _is_valid(pts):
            continue
        cnt = _count_ordinary(pts)
        if cnt < best_count:
            best_count = cnt
            best_pts = [p[:] for p in pts]
    
    return best_pts, best_count

def _local_search(points, max_iters=5000):
    """Try to improve by moving individual points."""
    if not _is_valid(points):
        return points, _count_ordinary(points)
    
    current_pts = [p[:] for p in points]
    current_count = _count_ordinary(current_pts)
    
    # Search space for new points: small integer grid
    # Try moving one point at a time to a nearby location
    search_range = 15
    base_x = sum(p[0] for p in current_pts) // 12
    base_y = sum(p[1] for p in current_pts) // 12
    
    for iteration in range(max_iters):
        if time.time() - _start_time > TIME_LIMIT:
            break
        improved = False
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_pts[idx]
        
        # Try several candidate positions
        candidates = []
        for dx in range(-search_range, search_range+1):
            for dy in range(-search_range, search_range+1):
                nx = old_x + dx
                ny = old_y + dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    candidates.append((nx, ny))
        random.shuffle(candidates)
        
        for nx, ny in candidates[:200]:  # Try up to 200 candidates per point
            if time.time() - _start_time > TIME_LIMIT:
                break
            # Check if this position is already used
            if (nx, ny) in [(p[0], p[1]) for p in current_pts]:
                continue
            new_pts = [p[:] for p in current_pts]
            new_pts[idx] = [nx, ny]
            if not _is_valid(new_pts):
                continue
            cnt = _count_ordinary(new_pts)
            if cnt < current_count:
                current_pts = new_pts
                current_count = cnt
                improved = True
                break
        if not improved:
            # Try a different point
            pass
    
    return current_pts, current_count

def construct():
    global _start_time
    _start_time = time.time()
    
    random.seed(42)
    
    # First try hand-crafted configurations
    best_pts, best_count = _try_configurations()
    
    if best_pts is None:
        # Fallback: 4x3 grid
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_count = _count_ordinary(best_pts)
    
    # Local search to improve
    if time.time() - _start_time < TIME_LIMIT - 5:
        best_pts, best_count = _local_search(best_pts, max_iters=10000)
    
    # Final validation
    if not _is_valid(best_pts):
        best_pts = [[x, y] for x in range(4) for y in range(3)]
    
    return best_pts
# EVOLVE-BLOCK-END

