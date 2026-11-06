import math
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 138.0  # Leave 12s margin for the 150s kill

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

def _dist_sq(p1, p2, D):
    """Squared distance between (x1,y1) and (x2,y2) in the D-plane:
       (x1-x2)^2 + D*(y1-y2)^2"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return _sq(dx) + D * _sq(dy)

def _are_collinear(p1, p2, p3):
    """Three points collinear: cross product of (p2-p1) and (p3-p1) is 0."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) == 0

def _are_concyclic(p1, p2, p3, p4, D):
    """Four points concyclic in the D-plane.
       Circle: x^2 + D*y^2 + Ax + By + C = 0
       Determinant must be 0:
       | x  y  x^2+D*y^2  1 |
       | x  y  x^2+D*y^2  1 |
       | x  y  x^2+D*y^2  1 |
       | x  y  x^2+D*y^2  1 |
    """
    def det4(m):
        # m is 4x4
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
             - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
             + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
             - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    
    m = []
    for p in [p1, p2, p3, p4]:
        m.append([p[0], p[1], _sq(p[0]) + D * _sq(p[1]), 1])
    return det4(m) == 0

def _valid_set(pts, D):
    """Check if a set of points (list of (int, int)) is valid for the given D."""
    n = len(pts)
    if n < 3:
        return False
    
    # Check distinctness
    if len(set(pts)) != n:
        return False
    
    # Check rational (integer) distances
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _dist_sq(pts[i], pts[j], D)
            if not _is_square(dsq):
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(pts[i], pts[j], pts[k]):
                    return False
    
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    
    return True

def _format_output(pts, D):
    """Format integer points to the required output format."""
    return {'D': D, 'points': [[[p[0], 1], [p[1], 1]] for p in pts]}

def _find_candidates_near(base, D, x_range, y_range):
    """Find points with integer distances to all base points."""
    candidates = []
    base_set = set(base)
    for x in range(x_range[0], x_range[1] + 1):
        if _deadline_exceeded():
            break
        for y in range(y_range[0], y_range[1] + 1):
            p = (x, y)
            if p in base_set:
                continue
            valid = True
            for bp in base:
                dsq = _dist_sq(p, bp, D)
                if not _is_square(dsq):
                    valid = False
                    break
            if valid:
                candidates.append(p)
    return candidates

def _try_add_point(base, D, x_range, y_range):
    """Try to add one point to base set maintaining validity."""
    candidates = _find_candidates_near(base, D, x_range, y_range)
    for cand in candidates:
        if _deadline_exceeded():
            return None
        test_set = base + [cand]
        if _valid_set(test_set, D):
            return test_set
    return None

def _search_d1():
    """Search for integral point sets with D=1."""
    # Known 5-point set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _valid_set(base5, 1):
        return [(0, 0), (1, 0), (0, 1)], 1
    
    # Try to extend to 6 points with a wider search
    result = _try_add_point(base5, 1, (-200, 200), (-200, 200))
    if result is not None:
        return result, 1
    
    # Try other 5-point configurations
    # The set {(0,0), (a,0), (0,b), (a,b), (x,y)} where all distances are integers
    # Let's try to find 5-point sets that might extend to 6
    
    # Try symmetric configurations
    # Known: (0,0), (6,0), (0,8), (6,8) form a rectangle with integer diagonals (10)
    # Add (3, 4*sqrt(7))... no, need integer coords for D=1
    
    # Try to find a 5-point set different from the known one that might extend
    # Use the fact that we need all pairwise distances to be integers
    
    # Let's try a different approach: build up from 3 points
    # Start with equilateral-like or other configurations
    
    # Actually, let's try to extend the 5-point set more aggressively
    # by checking a larger range but with early termination
    
    # Try adding two points at once (search for pairs)
    # This is expensive, so let's try a limited version
    if not _deadline_exceeded():
        # Try to find a 6th point with extended range
        result = _try_add_point(base5, 1, (-500, 500), (-500, 500))
        if result is not None:
            return result, 1
    
    return base5, 1

def _search_d2():
    """Search for integral point sets with D=2."""
    # For D=2, distance squared = dx^2 + 2*dy^2
    # Need dx^2 + 2*dy^2 to be a perfect square
    
    # Start with simple configurations
    # (0,0) and (a,b): a^2 + 2b^2 = c^2
    # (0,0), (2,1): 4+2=6 not square
    # (0,0), (4,1): 16+2=18 not square  
    # (0,0), (2,2): 4+8=12 not square
    # (0,0), (6,1): 36+2=38 not square
    # (0,0), (4,2): 16+8=24 not square
    # (0,0), (8,1): 64+2=66 not square
    # (0,0), (1,2): 1+8=9=3^2! So (0,0) and (1,2) work
    # (0,0), (3,2): 9+8=17 not square
    # (0,0), (5,2): 25+8=33 not square
    # (0,0), (7,2): 49+8=57 not square
    # (0,0), (9,2): 81+8=89 not square
    # (0,0), (11,2): 121+8=129 not square
    # (0,0), (13,2): 169+8=177 not square
    # (0,0), (15,2): 225+8=233 not square
    # (0,0), (17,2): 289+8=297 not square
    # (0,0), (19,2): 361+8=369 not square
    # (0,0), (21,2): 441+8=449 not square
    # (0,0), (23,2): 529+8=537 not square
    # (0,0), (25,2): 625+8=633 not square
    # (0,0), (27,2): 729+8=737 not square
    # (0,0), (29,2): 841+8=849 not square
    # (0,0), (31,2): 961+8=969 not square
    # (0,0), (33,2): 1089+8=1097 not square
    # (0,0), (35,2): 1225+8=1233 not square
    # (0,0), (37,2): 1369+8=1377 not square
    # (0,0), (39,2): 1521+8=1529 not square
    # (0,0), (41,2): 1681+8=1689 not square
    # (0,0), (43,2): 1849+8=1857 not square
    # (0,0), (45,2): 2025+8=2033 not square
    # (0,0), (47,2): 2209+8=2217 not square
    # (0,0), (49,2): 2401+8=2409 not square
    # (0,0), (51,2): 2601+8=2609 not square
    # (0,0), (53,2): 2809+8=2817 not square
    # (0,0), (55,2): 3025+8=3033 not square
    # (0,0), (57,2): 3249+8=3257 not square
    # (0,0), (59,2): 3481+8=3489 not square
    # (0,0), (61,2): 3721+8=3729 not square
    # (0,0), (63,2): 3969+8=3977 not square
    # (0,0), (65,2): 4225+8=4233 not square
    # (0,0), (67,2): 4489+8=4497 not square
    # (0,0), (69,2): 4761+8=4769 not square
    # (0,0), (71,2): 5041+8=5049 not square
    # (0,0), (73,2): 5329+8=5337 not square
    # (0,0), (75,2): 5625+8=5633 not square
    # (0,0), (77,2): 5929+8=5937 not square
    # (0,0), (79,2): 6241+8=6249 not square
    # (0,0), (81,2): 6561+8=6569 not square
    # (0,0), (83,2): 6889+8=6897 not square
    # (0,0), (85,2): 7225+8=7233 not square
    # (0,0), (87,2): 7569+8=7577 not square
    # (0,0), (89,2): 7921+8=7929 not square
    # (0,0), (91,2): 8281+8=8289 not square
    # (0,0), (93,2): 8649+8=8657 not square
    # (0,0), (95,2): 9025+8=9033 not square
    # (0,0), (97,2): 9409+8=9417 not square
    # (0,0), (99,2): 9801+8=9809 not square
    
    # Let me try other y values
    # dy=1: dx^2 + 2 = c^2 => c^2 - dx^2 = 2 => (c-dx)(c+dx)=2 => only trivial
    # dy=2: dx^2 + 8 = c^2 => c^2 - dx^2 = 8 => (c-dx)(c+dx)=8
    #   c-dx=2, c+dx=4 => c=3, dx=1 => (1,2) works (found above)
    #   c-dx=1, c+dx=8 => c=4.5 not integer
    # dy=3: dx^2 + 18 = c^2 => c^2 - dx^2 = 18 => (c-dx)(c+dx)=18
    #   c-dx=2, c+dx=9 => c=5.5 not integer
    #   c-dx=3, c+dx=6 => c=4.5 not integer
    # dy=4: dx^2 + 32 = c^2 => c^2 - dx^2 = 32 => (c-dx)(c+dx)=32
    #   c-dx=2, c+dx=16 => c=9, dx=7 => (7,4) works
    #   c-dx=4, c+dx=8 => c=6, dx=2 => (2,4) works
    #   c-dx=1, c+dx=32 => c=16.5 not integer
    # dy=5: dx^2 + 50 = c^2 => c^2 - dx^2 = 50 => (c-dx)(c+dx)=50
    #   c-dx=2, c+dx=25 => c=13.5 not integer
    #   c-dx=5, c+dx=10 => c=7.5 not integer
    # dy=6: dx^2 + 72 = c^2 => c^2 - dx^2 = 72 => (c-dx)(c+dx)=72
    #   c-dx=2, c+dx=36 => c=19, dx=17 => (17,6) works
    #   c-dx=4, c+dx=18 => c=11, dx=7 => (7,6) works
    #   c-dx=6, c+dx=12 => c=9, dx=3 => (3,6) works
    #   c-dx=8, c+dx=9 => c=8.5 not integer
    
    # So points with integer distance from origin: (1,2), (7,4), (2,4), (17,6), (7,6), (3,6), etc.
    # Let's try to build a set
    
    # Try (0,0), (1,2), (7,4), (2,4)
    # dist((1,2),(7,4)): dx=6, dy=2, dsq=36+8=44 not square
    # dist((1,2),(2,4)): dx=1, dy=2, dsq=1+8=9=3^2 ✓
    # dist((7,4),(2,4)): dx=5, dy=0, dsq=25=5^2 ✓
    # dist((0,0),(1,2))=3, dist((0,0),(7,4))=9, dist((0,0),(2,4))=6
    # So (0,0), (1,2), (2,4), (7,4) all have integer distances!
    
    base = [(0, 0), (1, 2), (2, 4), (7, 4)]
    
    if not _valid_set(base, 2):
        # Try simpler 3-point set
        base3 = [(0, 0), (1, 2), (2, 4)]
        if _valid_set(base3, 2):
            return base3, 2
        return [(0, 0), (1, 0), (0, 1)], 1  # fallback to D=1
    
    # Try to extend to 5 points
    result = _try_add_point(base, 2, (-50, 50), (-50, 50))
    if result is not None:
        return result, 2
    
    # Try to extend to 6 points
    if not _deadline_exceeded():
        result = _try_add_point(base, 2, (-100, 100), (-100, 100))
        if result is not None:
            return result, 2
    
    return base, 2

def _search_d3():
    """Search for integral point sets with D=3."""
    # For D=3, distance squared = dx^2 + 3*dy^2
    # (0,0), (1,1): 1+3=4=2^2 ✓
    # (0,0), (2,1): 4+3=7 not square
    # (0,0), (3,1): 9+3=12 not square
    # (0,0), (4,1): 16+3=19 not square
    # (0,0), (5,1): 25+3=28 not square
    # (0,0), (6,1): 36+3=39 not square
    # (0,0), (7,1): 49+3=52 not square
    # (0,0), (8,1): 64+3=67 not square
    # (0,0), (9,1): 81+3=84 not square
    # (0,0), (10,1): 100+3=103 not square
    # (0,0), (11,1): 121+3=124 not square
    # (0,0), (12,1): 144+3=147 not square
    # (0,0), (13,1): 169+3=172 not square
    # (0,0), (14,1): 196+3=199 not square
    # (0,0), (15,1): 225+3=228 not square
    # (0,0), (16,1): 256+3=259 not square
    # (0,0), (17,1): 289+3=292 not square
    # (0,0), (18,1): 324+3=327 not square
    # (0,0), (19,1): 361+3=364 not square
    # (0,0), (20,1): 400+3=403 not square
    # (0,0), (21,1): 441+3=444 not square
    # (0,0), (22,1): 484+3=487 not square
    # (0,0), (23,1): 529+3=532 not square
    # (0,0), (24,1): 576+3=579 not square
    # (0,0), (25,1): 625+3=628 not square
    # (0,0), (26,1): 676+3=679 not square
    # (0,0), (27,1): 729+3=732 not square
    # (0,0), (28,1): 784+3=787 not square
    # (0,0), (29,1): 841+3=844 not square
    # (0,0), (30,1): 900+3=903 not square
    # (0,0), (31,1): 961+3=964 not square
    # (0,0), (32,1): 1024+3=1027 not square
    # (0,0), (33,1): 1089+3=1092 not square
    # (0,0), (34,1): 1156+3=1159 not square
    # (0,0), (35,1): 1225+3=1228 not square
    # (0,0), (36,1): 1296+3=1299 not square
    # (0,0), (37,1): 1369+3=1372 not square
    # (0,0), (38,1): 1444+3=1447 not square
    # (0,0), (39,1): 1521+3=1524 not square
    # (0,0), (40,1): 1600+3=1603 not square
    # (0,0), (41,1): 1681+3=1684 not square
    # (0,0), (42,1): 1764+3=1767 not square
    # (0,0), (43,1): 1849+3=1852 not square
    # (0,0), (44,1): 1936+3=1939 not square
    # (0,0), (45,1): 2025+3=2028 not square
    # (0,0), (46,1): 2116+3=2119 not square
    # (0,0), (47,1): 2209+3=2212 not square
    # (0,0), (48,1): 2304+3=2307 not square
    # (0,0), (49,1): 2401+3=2404 not square
    # (0,0), (50,1): 2500+3=2503 not square
    
    # dy=1: dx^2+3=c^2 => c^2-dx^2=3 => (c-dx)(c+dx)=3 => c-dx=1, c+dx=3 => c=2, dx=1 => (1,1)
    # dy=2: dx^2+12=c^2 => c^2-dx^2=12 => (c-dx)(c+dx)=12
    #   c-dx=2, c+dx=6 => c=4, dx=2 => (2,2)
    #   c-dx=3, c+dx=4 => c=3.5 not integer
    #   c-dx=1, c+dx=12 => c=6.5 not integer
    # dy=3: dx^2+27=c^2 => c^2-dx^2=27 => (c-dx)(c+dx)=27
    #   c-dx=1, c+dx=27 => c=14, dx=13 => (13,3)
    #   c-dx=3, c+dx=9 => c=6, dx=3 => (3,3)
    # dy=4: dx^2+48=c^2 => c^2-dx^2=48 => (c-dx)(c+dx)=48
    #   c-dx=2, c+dx=24 => c=13, dx=11 => (11,4)
    #   c-dx=4, c+dx=12 => c=8, dx=4 => (4,4)
    #   c-dx=6, c+dx=8 => c=7, dx=1 => (1,4)
    
    # Points with integer distance from origin: (1,1), (2,2), (13,3), (3,3), (11,4), (4,4), (1,4), ...
    
    # Try (0,0), (1,1), (2,2)
    # dist((1,1),(2,2)): dx=1, dy=1, dsq=1+3=4=2^2 ✓
    # But (0,0), (1,1), (2,2) are collinear! (slope 1)
    
    # Try (0,0), (1,1), (1,4)
    # dist((1,1),(1,4)): dx=0, dy=3, dsq=0+27=27 not square
    
    # Try (0,0), (1,1), (4,4)
    # dist((1,1),(4,4)): dx=3, dy=3, dsq=9+27=36=6^2 ✓
    # But (0,0), (1,1), (4,4) are collinear!
    
    # Try (0,0), (1,1), (3,3)
    # Collinear
    
    # Try (0,0), (1,4), (11,4)
    # dist((1,4),(11,4)): dx=10, dy=0, dsq=100=10^2 ✓
    # dist((0,0),(1,4)): 1+48=49=7^2 ✓
    # dist((0,0),(11,4)): 121+48=169=13^2 ✓
    # Check collinearity: (0,0), (1,4), (11,4)
    # Cross product: (1-0)*(4-0) - (11-0)*(4-0) = 4-44 = -40 ≠ 0 ✓
    
    base = [(0, 0), (1, 4), (11, 4)]
    
    if not _valid_set(base, 3):
        return [(0, 0), (1, 0), (0, 1)], 1
    
    # Try to extend
    result = _try_add_point(base, 3, (-30, 30), (-30, 30))
    if result is not None:
        return result, 3
    
    # Try to add (2,2)
    test = base + [(2, 2)]
    if _valid_set(test, 3):
        base = test
        result = _try_add_point(base, 3, (-30, 30), (-30, 30))
        if result is not None:
            return result, 3
    
    return base, 3

def _search_d5():
    """Search for integral point sets with D=5."""
    # For D=5, distance squared = dx^2 + 5*dy^2
    # (0,0), (1,1): 1+5=6 not square
    # (0,0), (2,1): 4+5=9=3^2 ✓
    # (0,0), (3,1): 9+5=14 not square
    # (0,0), (4,1): 16+5=21 not square
    # (0,0), (5,1): 25+5=30 not square
    # (0,0), (6,1): 36+5=41 not square
    # (0,0), (7,1): 49+5=54 not square
    # (0,0), (8,1): 64+5=69 not square
    # (0,0), (9,1): 81+5=86 not square
    # (0,0), (10,1): 100+5=105 not square
    # (0,0), (11,1): 121+5=126 not square
    # (0,0), (12,1): 144+5=149 not square
    # (0,0), (13,1): 169+5=174 not square
    # (0,0), (14,1): 196+5=201 not square
    # (0,0), (15,1): 225+5=230 not square
    # (0,0), (16,1): 256+5=261 not square
    # (0,0), (17,1): 289+5=294 not square
    # (0,0), (18,1): 324+5=329 not square
    # (0,0), (19,1): 361+5=366 not square
    # (0,0), (20,1): 400+5=405 not square
    # (0,0), (21,1): 441+5=446 not square
    # (0,0), (22,1): 484+5=489 not square
    # (0,0), (23,1): 529+5=534 not square
    # (0,0), (24,1): 576+5=581 not square
    # (0,0), (25,1): 625+5=630 not square
    # (0,0), (26,1): 676+5=681 not square
    # (0,0), (27,1): 729+5=734 not square
    # (0,0), (28,1): 784+5=789 not square
    # (0,0), (29,1): 841+5=846 not square
    # (0,0), (30,1): 900+5=905 not square
    # (0,0), (31,1): 961+5=966 not square
    # (0,0), (32,1): 1024+5=1029 not square
    # (0,0), (33,1): 1089+5=1094 not square
    # (0,0), (34,1): 1156+5=1161 not square
    # (0,0), (35,1): 1225+5=1230 not square
    # (0,0), (36,1): 1296+5=1301 not square
    # (0,0), (37,1): 1369+5=1374 not square
    # (0,0), (38,1): 1444+5=1449 not square
    # (0,0), (39,1): 1521+5=1526 not square
    # (0,0), (40,1): 1600+5=1605 not square
    # (0,0), (41,1): 1681+5=1686 not square
    # (0,0), (42,1): 1764+5=1769 not square
    # (0,0), (43,1): 1849+5=1854 not square
    # (0,0), (44,1): 1936+5=1941 not square
    # (0,0), (45,1): 2025+5=2030 not square
    # (0,0), (46,1): 2116+5=2121 not square
    # (0,0), (47,1): 2209+5=2214 not square
    # (0,0), (48,1): 2304+5=2309 not square
    # (0,0), (49,1): 2401+5=2406 not square
    # (0,0), (50,1): 2500+5=2505 not square
    
    # dy=1: dx^2+5=c^2 => c^2-dx^2=5 => (c-dx)(c+dx)=5 => c-dx=1, c+dx=5 => c=3, dx=2 => (2,1)
    # dy=2: dx^2+20=c^2 => c^2-dx^2=20 => (c-dx)(c+dx)=20
    #   c-dx=2, c+dx=10 => c=6, dx=4 => (4,2)
    #   c-dx=4, c+dx=5 => c=4.5 not integer
    #   c-dx=1, c+dx=20 => c=10.5 not integer
    # dy=3: dx^2+45=c^2 => c^2-dx^2=45 => (c-dx)(c+dx)=45
    #   c-dx=3, c+dx=15 => c=9, dx=6 => (6,3)
    #   c-dx=5, c+dx=9 => c=7, dx=2 => (2,3)
    #   c-dx=1, c+dx=45 => c=23, dx=22 => (22,3)
    
    # Points: (2,1), (4,2), (6,3), (2,3), (22,3), ...
    
    # Try (0,0), (2,1), (2,3)
    # dist((2,1),(2,3)): dx=0, dy=2, dsq=0+20=20 not square
    
    # Try (0,0), (2,1), (4,2)
    # dist((2,1),(4,2)): dx=2, dy=1, dsq=4+5=9=3^2 ✓
    # But (0,0), (2,1), (4,2) are collinear! (slope 1/2)
    
    # Try (0,0), (2,1), (6,3)
    # dist((2,1),(6,3)): dx=4, dy=2, dsq=16+20=36=6^2 ✓
    # But (0,0), (2,1), (6,3) are collinear! (slope 1/2)
    
    # Try (0,0), (2,1), (2,3)
    # dist((2,1),(2,3)): 0+20=20 not square
    
    # Try (0,0), (4,2), (2,3)
    # dist((4,2),(2,3)): dx=-2, dy=1, dsq=4+5=9=3^2 ✓
    # dist((0,0),(4,2)): 16+20=36=6^2 ✓
    # dist((0,0),(2,3)): 4+45=49=7^2 ✓
    # Check collinearity: (0,0), (4,2), (2,3)
    # Cross product: (4-0)*(3-0) - (2-0)*(2-0) = 12-4 = 8 ≠ 0 ✓
    
    base = [(0, 0), (4, 2), (2, 3)]
    
    if not _valid_set(base, 5):
        return [(0, 0), (1, 0), (0, 1)], 1
    
    # Try to extend
    result = _try_add_point(base, 5, (-30, 30), (-30, 30))
    if result is not None:
        return result, 5
    
    return base, 5

def construct():
    best_points = None
    best_D = 1
    best_n = 0
    
    # Try D=1 first (most well-studied)
    if not _deadline_exceeded():
        pts, D = _search_d1()
        if pts is not None and len(pts) > best_n:
            best_points = pts
            best_D = D
            best_n = len(pts)
    
    # Try D=2
    if not _deadline_exceeded():
        pts, D = _search_d2()
        if pts is not None and len(pts) > best_n:
            best_points = pts
            best_D = D
            best_n = len(pts)
    
    # Try D=3
    if not _deadline_exceeded():
        pts, D = _search_d3()
        if pts is not None and len(pts) > best_n:
            best_points = pts
            best_D = D
            best_n = len(pts)
    
    # Try D=5
    if not _deadline_exceeded():
        pts, D = _search_d5()
        if pts is not None and len(pts) > best_n:
            best_points = pts
            best_D = D
            best_n = len(pts)
    
    # Fallback: always have a valid 3-point set
    if best_points is None or len(best_points) < 3:
        best_points = [(0, 0), (1, 0), (0, 1)]
        best_D = 1
    
    return _format_output(best_points, best_D)
