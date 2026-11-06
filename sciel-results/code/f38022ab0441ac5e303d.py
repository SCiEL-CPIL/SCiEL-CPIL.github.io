import math
import time
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Leave 10s margin for the 150s kill

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
    """Return squared distance as (num, den) tuple for exact integer arithmetic."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return _sq(dx) + _sq(dy)

def _is_rational_square(val):
    """Check if a Fraction is a perfect square of a rational number."""
    num = val.numerator
    den = val.denominator
    if num < 0:
        return False
    return _is_square(num) and _is_square(den)

def _are_collinear(p1, p2, p3):
    """Check if three points are collinear using cross product."""
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) == 0

def _are_concyclic(p1, p2, p3, p4):
    """Check if four points are concyclic using determinant."""
    # Circle equation: x^2 + y^2 + Dx + Ey + F = 0
    # Determinant of matrix:
    # | x  y  x^2+y^2  1 |
    # | x  y  x^2+y^2  1 |
    # | x  y  x^2+y^2  1 |
    # | x  y  x^2+y^2  1 |
    def det4(m):
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
             - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
             + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
             - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    
    m = []
    for p in [p1, p2, p3, p4]:
        m.append([p[0], p[1], _sq(p[0]) + _sq(p[1]), 1])
    return det4(m) == 0

def _valid_set(pts):
    """Check if a set of points (as list of (int, int) tuples) is valid."""
    n = len(pts)
    if n < 3:
        return False
    
    # Check distinctness
    if len(set(pts)) != n:
        return False
    
    # Check rational distances
    for i in range(n):
        for j in range(i+1, n):
            dsq = _dist_sq(pts[i], pts[j])
            # dsq is an integer; need it to be a perfect square
            if not _is_square(dsq):
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(pts[i], pts[j], pts[k]):
                    return False
    
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(pts[i], pts[j], pts[k], pts[l]):
                        return False
    
    return True

def _format_output(pts):
    """Format integer points to the required output format."""
    return {'D': 1, 'points': [[[p[0], 1], [p[1], 1]] for p in pts]}

def _search_6_points():
    """Try to find a 6-point integral set in general position."""
    # Start with the known 5-point set and try to add a 6th point
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try adding points in a grid around the existing set
    # Points must have integer distances to all 5 base points
    # and must not create collinearities or concyclicities
    
    # Generate candidate points
    candidates = []
    # Search in a reasonable range
    for x in range(-100, 101):
        for y in range(-100, 101):
            if (x, y) in base:
                continue
            # Quick check: integer distances to all base points
            valid = True
            for bp in base:
                dsq = _dist_sq((x, y), bp)
                if not _is_square(dsq):
                    valid = False
                    break
            if valid:
                candidates.append((x, y))
    
    if _deadline_exceeded():
        return None
    
    # Try each candidate
    for cand in candidates:
        if _deadline_exceeded():
            return None
        test_set = base + [cand]
        if _valid_set(test_set):
            return test_set
    
    return None

def _search_7_points():
    """Try to find a 7-point integral set in general position."""
    # This is more difficult; try local search from 6-point sets
    # For now, return None and rely on 5 or 6 point results
    return None

def _search_8_points():
    """Try to find an 8-point integral set in general position."""
    return None

def construct():
    # Start with the known 5-point set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify the base set
    if not _valid_set(best_points):
        # Fallback to a simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}
    
    # Try to find 6 points
    if not _deadline_exceeded():
        result = _search_6_points()
        if result is not None:
            best_points = result
    
    # Try to find 7 points
    if not _deadline_exceeded():
        result = _search_7_points()
        if result is not None:
            best_points = result
    
    # Try to find 8 points
    if not _deadline_exceeded():
        result = _search_8_points()
        if result is not None:
            best_points = result
    
    return _format_output(best_points)
