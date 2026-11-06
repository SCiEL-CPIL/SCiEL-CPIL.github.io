"""Rewritten program to search for integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a if a != 0 else 1

def _dist_sq_rational(p1, p2):
    """Check if squared distance between p1 and p2 is a perfect square (rational distance)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    d2 = dx * dx + dy * dy
    return _is_square(d2)

def _no_three_collinear(points):
    """Check if no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Determinant
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    return True

def _no_four_concyclic(points):
    """Check if no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    
                    # General circle equation: x^2 + y^2 + Dx + Ey + F = 0
                    # We solve for D, E, F using 3 points and check the 4th.
                    # Or use determinant.
                    # | x^2+y^2  x  y  1 |
                    # | ...      ... ... ...|
                    
                    # Using fractions to avoid floating point errors
                    # System of linear equations for D, E, F:
                    # x_i^2 + y_i^2 + D*x_i + E*y_i + F = 0
                    # Let A = x^2+y^2, b=x, c=y
                    # We have:
                    # D*x1 + E*y1 + F = -A1
                    # D*x2 + E*y2 + F = -A2
                    # D*x3 + E*y3 + F = -A3
                    
                    A1 = x1*x1 + y1*y1
                    A2 = x2*x2 + y2*y2
                    A3 = x3*x3 + y3*y3
                    A4 = x4*x4 + y4*y4
                    
                    # Matrix M * [D, E, F]^T = [-A1, -A2, -A3]^T
                    # M = [[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]
                    
                    # Cramer's rule or Gaussian elimination with Fractions
                    # To be fast, we can check if the determinant of the 4x4 matrix is 0.
                    # Det = | A1 x1 y1 1 |
                    #       | A2 x2 y2 1 |
                    #       | A3 x3 y3 1 |
                    #       | A4 x4 y4 1 |
                    
                    # This is O(1) with big integers.
                    # We can expand along the last column or use a generic 4x4 det.
                    
                    # Simplified check:
                    # If we subtract row 1 from others:
                    # | A1-x1? No.
                    
                    # Let's use the property that 4 points are concyclic iff the cross ratio is real? No, that's for complex plane.
                    # In real plane, determinant is the way.
                    
                    # We can compute the circle through 1,2,3 and check 4.
                    # If 1,2,3 are collinear, they don't define a circle, but we already checked no 3 collinear.
                    
                    # D, E, F
                    # x1*D + y1*E + F = -A1
                    # x2*D + y2*E + F = -A2
                    # x3*D + y3*E + F = -A3
                    
                    # Subtract 1 from 2 and 3:
                    # (x2-x1)D + (y2-y1)E = -(A2-A1)
                    # (x3-x1)D + (y3-y1)E = -(A3-A1)
                    
                    dx2 = x2 - x1
                    dy2 = y2 - y1
                    rhs2 = -(A2 - A1)
                    
                    dx3 = x3 - x1
                    dy3 = y3 - y1
                    rhs3 = -(A3 - A1)
                    
                    det = dx2 * dy3 - dx3 * dy2
                    if det == 0:
                        # Collinear, already handled, but just in case
                        return False 
                    
                    # D = (rhs2*dy3 - rhs3*dy2) / det
                    # E = (dx2*rhs3 - dx3*rhs2) / det
                    
                    # Check if 4 lies on the circle:
                    # x4*D + y4*E + F = -A4
                    # F = -A1 - x1*D - y1*E
                    # x4*D + y4*E - A1 - x1*D - y1*E = -A4
                    # (x4-x1)D + (y4-y1)E = A1 - A4
                    
                    # (x4-x1) * (rhs2*dy3 - rhs3*dy2) + (y4-y1) * (dx2*rhs3 - dx3*rhs2) = (A1 - A4) * det
                    
                    lhs = (x4 - x1) * (rhs2 * dy3 - rhs3 * dy2) + (y4 - y1) * (dx2 * rhs3 - dx3 * rhs2)
                    rhs = (A1 - A4) * det
                    
                    if lhs == rhs:
                        return False
    return True

def _validate(points):
    n = len(points)
    if n < 3:
        return False
    # Check distinct
    if len(set(points)) < n:
        return False
    # Check pairwise rational distances
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_sq_rational(points[i], points[j]):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _search_6_points(N):
    """Search for 6 points in [-N, N]^2."""
    start = time.time()
    # Precompute squares
    max_sq = (2 * N) * (2 * N) + (2 * N) * (2 * N)
    squares = set(i * i for i in range(math.isqrt(max_sq) + 1))
    
    def is_rational_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) in squares

    # Fix P1 = (0,0)
    p1 = (0, 0)
    
    # Iterate P2 = (a, 0) with a > 0
    for a in range(1, N + 1):
        if time.time() > start + 10:
            break
        p2 = (a, 0)
        
        # Iterate P3 = (b, c) with c > 0
        for b in range(-N, N + 1):
            for c in range(1, N + 1):
                if time.time() > start + 10:
                    break
                p3 = (b, c)
                
                # Check triangle 1-2-3
                if not is_rational_dist(p1, p3):
                    continue
                if not is_rational_dist(p2, p3):
                    continue
                
                # Find P4
                candidates4 = []
                for d in range(-N, N + 1):
                    for e in range(-N, N + 1):
                        if time.time() > start + 10:
                            break
                        p4 = (d, e)
                        # Must be distinct
                        if p4 == p1 or p4 == p2 or p4 == p3:
                            continue
                        # Must be rational distance to 1, 2, 3
                        if is_rational_dist(p1, p4) and is_rational_dist(p2, p4) and is_rational_dist(p3, p4):
                            # Check no 3 collinear
                            pts = [p1, p2, p3, p4]
                            if _no_three_collinear(pts):
                                candidates4.append(p4)
                
                if len(candidates4) < 1:
                    continue
                
                # For each P4, find P5
                for p4 in candidates4:
                    if time.time() > start + 10:
                        break
                    candidates5 = []
                    for f in range(-N, N + 1):
                        for g in range(-N, N + 1):
                            if time.time() > start + 10:
                                break
                            p5 = (f, g)
                            if p5 in (p1, p2, p3, p4):
                                continue
                            if is_rational_dist(p1, p5) and is_rational_dist(p2, p5) and is_rational_dist(p3, p5) and is_rational_dist(p4, p5):
                                pts = [p1, p2, p3, p4, p5]
                                if _no_three_collinear(pts):
                                    candidates5.append(p5)
                    
                    if len(candidates5) < 1:
                        continue
                    
                    # For each P5, find P6
                    for p5 in candidates5:
                        if time.time() > start + 10:
                            break
                        for h in range(-N, N + 1):
                            for i in range(-N, N + 1):
                                if time.time() > start + 10:
                                    break
                                p6 = (h, i)
                                if p6 in (p1, p2, p3, p4, p5):
                                    continue
                                if is_rational_dist(p1, p6) and is_rational_dist(p2, p6) and is_rational_dist(p3, p6) and is_rational_dist(p4, p6) and is_rational_dist(p5, p6):
                                    pts = [p1, p2, p3, p4, p5, p6]
                                    if _no_three_collinear(pts):
                                        if _no_four_concyclic(pts):
                                            return pts
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Known 5-point set
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    
    deadline = _START_TIME + 130.0
    
    # Try to find 6 points
    if time.time() < deadline - 10:
        # Search with increasing N
        for N in [20, 50, 100]:
            if time.time() > deadline - 5:
                break
            pts6 = _search_6_points(N)
            if pts6:
                if _validate(pts6):
                    best_score = 6
                    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts6]}
                    print(f"Found 6 points with N={N}")
                    break
    
    return best_result

# EVOLVE-BLOCK-END
