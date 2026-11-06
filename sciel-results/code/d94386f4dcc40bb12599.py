import math
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Leave 15s margin for the 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _deadline_exceeded():
    return _elapsed() > _TIME_LIMIT

def _sq(n):
    return n * n

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return _sq(dx) + _sq(dy)

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) == 0

def _are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Determinant of the 4x4 matrix:
    # x^2+y^2, x, y, 1
    # This is faster than solving for circle center
    d1 = x1*x1 + y1*y1
    d2 = x2*x2 + y2*y2
    d3 = x3*x3 + y3*y3
    d4 = x4*x4 + y4*y4
    
    # Using Sarrus rule or expansion for 4x4 determinant
    # | d1 x1 y1 1 |
    # | d2 x2 y2 1 |
    # | d3 x3 y3 1 |
    # | d4 x4 y4 1 |
    
    # Expand along last column:
    # 1 * det(d2,x2,y2; d3,x3,y3; d4,x4,y4) - 1 * det(d1,x1,y1; d3,x3,y3; d4,x4,y4)
    # + 1 * det(d1,x1,y1; d2,x2,y2; d4,x4,y4) - 1 * det(d1,x1,y1; d2,x2,y2; d3,x3,y3)
    
    def det3(a,b,c, d,e,f, g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        
    return (det3(d2,x2,y2, d3,x3,y3, d4,x4,y4) - 
            det3(d1,x1,y1, d3,x3,y3, d4,x4,y4) + 
            det3(d1,x1,y1, d2,x2,y2, d4,x4,y4) - 
            det3(d1,x1,y1, d2,x2,y2, d3,x3,y3)) == 0

def _valid_set(pts):
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    for i in range(n):
        for j in range(i+1, n):
            dsq = _dist_sq(pts[i], pts[j])
            if not _is_square(dsq):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    return True

def _format_output(pts):
    return {'D': 1, 'points': [[[p[0], 1], [p[1], 1]] for p in pts]}

def _is_valid_addition(base, p):
    """Check if p can be added to base set."""
    # Distances
    for b in base:
        if not _is_square(_dist_sq(b, p)):
            return False
    # Collinearity
    n = len(base)
    for i in range(n):
        for j in range(i+1, n):
            if _are_collinear(base[i], base[j], p):
                return False
    # Concyclicity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_concyclic(base[i], base[j], base[k], p):
                    return False
    return True

def _generate_pythagorean_points(max_coord):
    """Generate (x, y) such that x^2+y^2 is a square, |x|,|y| <= max_coord."""
    pts = set()
    # Iterate m, n for Euclid's formula
    # x = k(m^2-n^2), y = k(2mn)
    # We want x, y <= max_coord
    # m^2-n^2 <= max_coord/k and 2mn <= max_coord/k
    # Simple bound: m < sqrt(max_coord) is too small if k=1.
    # Actually x^2+y^2=r^2.
    # Let's just iterate r up to max_coord*sqrt(2) and find integer solutions?
    # No, that's the slow way.
    # Better: iterate m, n.
    # Max m is approx sqrt(max_coord) if k=1? No.
    # If m^2-n^2 <= max_coord, then m <= sqrt(max_coord + n^2).
    # Let's just iterate m up to sqrt(max_coord*2) + 10?
    # Actually, for x,y <= M, r <= M*sqrt(2).
    # m^2+n^2 = r^2 is not right, x= m^2-n^2, y=2mn.
    # r = m^2+n^2.
    # So m^2+n^2 <= M*sqrt(2).
    # Max m is approx sqrt(M*sqrt(2)).
    
    max_m = int(math.sqrt(max_coord * math.sqrt(2))) + 10
    
    for m in range(1, max_m + 1):
        for n in range(1, m):
            if (m - n) % 2 == 0: # Skip non-primitive if we want all? No, we want all.
                pass
            x = m*m - n*n
            y = 2*m*n
            r = m*m + n*n
            # We can scale by k
            # k*x <= max_coord and k*y <= max_coord
            max_k = min(max_coord // x, max_coord // y)
            for k in range(1, max_k + 1):
                px, py = x*k, y*k
                pts.add((px, py))
                pts.add((px, -py))
                pts.add((-px, py))
                pts.add((-px, -py))
                if x != y:
                    pts.add((py, px))
                    pts.add((py, -px))
                    pts.add((-py, px))
                    pts.add((-py, -px))
    return list(pts)

def _find_6th_point_grid(base5, max_r):
    """Find a 6th point at integer distance from all 5 base points using grid search."""
    ox, oy = base5[0]
    translated = [(x - ox, y - oy) for x, y in base5]
    base_set = set(translated)
    
    candidates = _generate_pythagorean_points(max_r)
    
    # Pre-check: point must be at integer distance from ALL base points.
    # We can filter candidates by checking against base[1], base[2] first (fastest).
    # Then check against the rest.
    
    for cx, cy in candidates:
        if _deadline_exceeded():
            return None
        if (cx, cy) in base_set:
            continue
        
        # Quick check: distance to origin (0,0) is sqrt(cx^2+cy^2) which is integer by construction.
        # Check against other base points.
        ok = True
        for bx, by in translated[1:]: # Skip origin
            dsq = (cx - bx)**2 + (cy - by)**2
            if not _is_square(dsq):
                ok = False
                break
        if not ok:
            continue
            
        px, py = cx + ox, cy + oy
        # Full validation
        if _is_valid_addition(base5, (px, py)):
            return base5 + [(px, py)]
            
    return None

def _find_7th_point(base6, max_r):
    """Find a 7th point at integer distance from all 6 base points."""
    ox, oy = base6[0]
    translated = [(x - ox, y - oy) for x, y in base6]
    base_set = set(translated)
    
    candidates = _generate_pythagorean_points(max_r)
    
    for cx, cy in candidates:
        if _deadline_exceeded():
            return None
        if (cx, cy) in base_set:
            continue
            
        ok = True
        for bx, by in translated[1:]:
            dsq = (cx - bx)**2 + (cy - by)**2
            if not _is_square(dsq):
                ok = False
                break
        if not ok:
            continue
            
        px, py = cx + ox, cy + oy
        if _is_valid_addition(base6, (px, py)):
            return base6 + [(px, py)]
            
    return None

def construct():
    # Known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _valid_set(best_points):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    # Strategy 1: Grid search for 6th point
    # The previous code was too slow. This one is optimized.
    # Try increasing radii.
    for max_r in [100, 200, 400, 800, 1600, 3200]:
        if _deadline_exceeded():
            break
        result = _find_6th_point_grid(best_points, max_r)
        if result is not None:
            best_points = result
            break
            
    # Strategy 2: Try to find 7th point if we have 6
    if len(best_points) == 6 and not _deadline_exceeded():
        for max_r in [100, 200, 400, 800]:
            if _deadline_exceeded():
                break
            result = _find_7th_point(best_points, max_r)
            if result is not None:
                best_points = result
                break
                
    # If we still have 5, try a different seed?
    # The seed is fixed. If we can't find 6, we return 5.
    # But maybe we can try a randomized search?
    # Given the time limit, the grid search is the best bet.
    
    return _format_output(best_points)
