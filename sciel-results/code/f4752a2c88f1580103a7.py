"""
Search for integral point sets in general position (no three collinear, no four concyclic)
with rational coordinates of the form (x, y*sqrt(D)) for squarefree D.
The score is the number of points (maximize).
"""

import time
import math
import random
from fractions import Fraction
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 120.0  # seconds total
_SAFETY_MARGIN = 5.0  # return 5s before limit

def _time_left():
    return _TIME_LIMIT - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _is_square(n):
    """Check if a non-negative integer is a perfect square."""
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _sq_dist_is_rational(p1, p2, D):
    """
    Check if squared distance between two points is rational.
    Points are (x1, y1*sqrt(D)), (x2, y2*sqrt(D)) where x,y are Fractions.
    Squared distance = (x1-x2)^2 + D*(y1-y2)^2
    This is rational iff D*(y1-y2)^2 is rational, i.e., D is a perfect square OR (y1-y2)=0.
    Wait, D is squarefree. So D*(y1-y2)^2 is rational iff (y1-y2)^2 * D is rational.
    Since y1, y2 are rational, (y1-y2) is rational, so (y1-y2)^2 is rational.
    Thus D*(y1-y2)^2 is rational iff D is rational (which it is, D is an integer).
    Actually, the distance squared is (x1-x2)^2 + D*(y1-y2)^2.
    For this to be rational, since (x1-x2)^2 is rational and D*(y1-y2)^2 is rational (D integer, (y1-y2)^2 rational),
    the sum is always rational. 
    
    Wait, the task says "All pairwise distances must be rational". Distance = sqrt(sq_dist).
    So we need sq_dist to be a perfect square of a rational number.
    
    Let me re-read: "All pairwise distances must be rational (equivalently integral after scaling)"
    So we need sqrt((x1-x2)^2 + D*(y1-y2)^2) to be rational.
    This means (x1-x2)^2 + D*(y1-y2)^2 must be a perfect square in Q.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    sq_d = dx*dx + D * dy*dy
    # Check if sq_d is a perfect square of a rational
    if sq_d < 0:
        return False
    num = sq_d.numerator
    den = sq_d.denominator
    return _is_square(num) and _is_square(den)

def _dist_is_rational(p1, p2, D):
    """Check if distance between two points is rational."""
    return _sq_dist_is_rational(p1, p2, D)

def _no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i][0], points[i][1]
                x2, y2 = points[j][0], points[j][1]
                x3, y3 = points[k][0], points[k][1]
                # Collinear if (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
                # But y values are y*sqrt(D), so we need to be careful.
                # Actually, collinearity: (x2-x1)/(y2*sqrt(D)-y1*sqrt(D)) == (x3-x1)/(y3*sqrt(D)-y1*sqrt(D))
                # This simplifies to (x2-x1)/(y2-y1) == (x3-x1)/(y3-y1) when D != 0
                # So: (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
                lhs = (x2 - x1) * (y3 - y1)
                rhs = (x3 - x1) * (y2 - y1)
                if lhs == rhs:
                    return False
    return True

def _no_four_concyclic(points, D):
    """
    Check no four points are concyclic.
    Four points (x_i, y_i*sqrt(D)) are concyclic if there exists a circle passing through all four.
    A circle in this coordinate system: (x-a)^2 + (y*sqrt(D)-b)^2 = r^2
    => x^2 - 2ax + a^2 + D*y^2 - 2b*sqrt(D)*y + b^2 = r^2
    => x^2 + D*y^2 - 2ax - 2b*sqrt(D)*y + (a^2 + b^2 - r^2) = 0
    
    For four points to be concyclic, the 4x4 system must be consistent.
    Using the determinant approach: four points are concyclic iff the determinant of a certain 4x4 matrix is zero.
    
    For points (x_i, Y_i) where Y_i = y_i*sqrt(D), the condition is:
    | x_i^2 + Y_i^2, x_i, Y_i, 1 | = 0 for all four points
    
    But Y_i = y_i*sqrt(D), so x_i^2 + Y_i^2 = x_i^2 + D*y_i^2.
    
    The determinant:
    | x1^2 + D*y1^2, x1, y1*sqrt(D), 1 |
    | x2^2 + D*y2^2, x2, y2*sqrt(D), 1 |
    | x3^2 + D*y3^2, x3, y3*sqrt(D), 1 |
    | x4^2 + D*y4^2, x4, y4*sqrt(D), 1 | = 0
    
    Factor sqrt(D) from the third column:
    sqrt(D) * | x1^2 + D*y1^2, x1, y1, 1 |
              | x2^2 + D*y2^2, x2, y2, 1 |
              | x3^2 + D*y3^2, x3, y3, 1 |
              | x4^2 + D*y4^2, x4, y4, 1 | = 0
    
    Since sqrt(D) != 0, we need:
    | x1^2 + D*y1^2, x1, y1, 1 |
    | x2^2 + D*y2^2, x2, y2, 1 |
    | x3^2 + D*y3^2, x3, y3, 1 |
    | x4^2 + D*y4^2, x4, y4, 1 | = 0
    """
    n = len(points)
    for combo in combinations(range(n), 4):
        i, j, k, l = combo
        # Build 4x4 matrix
        det = 0
        # Use cofactor expansion or direct formula
        # For a 4x4 matrix, compute determinant
        a = [[points[i][0]**2 + D * points[i][1]**2, points[i][0], points[i][1], Fraction(1)],
             [points[j][0]**2 + D * points[j][1]**2, points[j][0], points[j][1], Fraction(1)],
             [points[k][0]**2 + D * points[k][1]**2, points[k][0], points[k][1], Fraction(1)],
             [points[l][0]**2 + D * points[l][1]**2, points[l][0], points[l][1], Fraction(1)]]
        
        # Compute 4x4 determinant
        # det = sum over permutations... or use recursive expansion
        # Let's use the formula for 4x4
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1])
                  - m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0])
                  + m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0])
                  - m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        
        if det4(a) == 0:
            return False
    return True

def _validate(points, D):
    """Full validation of a point set."""
    n = len(points)
    if n < 3:
        return False
    
    # Check all pairwise distances are rational
    for i in range(n):
        for j in range(i+1, n):
            if not _dist_is_rational(points[i], points[j], D):
                return False
    
    # Check no three collinear
    if not _no_three_collinear(points):
        return False
    
    # Check no four concyclic
    if not _no_four_concyclic(points, D):
        return False
    
    return True

def _format_output(points, D):
    """Format points for output."""
    return {'D': D, 'points': [[[p[0].numerator, p[0].denominator], [p[1].numerator, p[1].denominator]] for p in points]}

def construct():
    """
    Search for integral point sets in general position.
    Returns the best (largest) valid set found.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with the known 5-point set
    best_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)), (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)), (Fraction(16), Fraction(63))]
    best_D = 1
    best_count = 5
    
    # Verify the initial set
    if _validate(best_points, best_D):
        pass
    else:
        # Should not happen, but fallback
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))]
        best_count = 3
    
    # Try to extend by adding points
    # Strategy: search for new points that maintain all constraints
    # Use D=1 first (integer points), then try other D values
    
    # For D=1, search over a grid of integer points
    # The known 5-point set uses points in a region around origin
    
    # Try to find a 6th point
    # For D=1, we need integer points with integer distances to all existing points
    # This is equivalent to finding points in the intersection of circles
    
    # Let's try a local search: for each candidate point, check if it can be added
    
    # Candidate generation: try points near the existing ones
    existing = best_points
    
    # Try adding points from a grid
    # The existing points span x in [-56, 16], y in [0, 63]
    # Try a wider grid
    
    # For efficiency, precompute squared distances
    def can_add(candidate, current_points, D):
        """Check if candidate can be added to current_points."""
        # Check distance to all existing points
        for p in current_points:
            if not _dist_is_rational(candidate, p, D):
                return False
        # Check collinearity with all pairs
        for i in range(len(current_points)):
            for j in range(i+1, len(current_points)):
                p1 = current_points[i]
                p2 = current_points[j]
                # Check if candidate, p1, p2 are collinear
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]
                xc, yc = candidate[0], candidate[1]
                lhs = (x2 - x1) * (yc - y1)
                rhs = (xc - x1) * (y2 - y1)
                if lhs == rhs:
                    return False
        # Check concyclicity with all triples
        for i in range(len(current_points)):
            for j in range(i+1, len(current_points)):
                for k in range(j+1, len(current_points)):
                    p1 = current_points[i]
                    p2 = current_points[j]
                    p3 = current_points[k]
                    # Check if candidate, p1, p2, p3 are concyclic
                    a = [[p1[0]**2 + D * p1[1]**2, p1[0], p1[1], Fraction(1)],
                         [p2[0]**2 + D * p2[1]**2, p2[0], p2[1], Fraction(1)],
                         [p3[0]**2 + D * p3[1]**2, p3[0], p3[1], Fraction(1)],
                         [candidate[0]**2 + D * candidate[1]**2, candidate[0], candidate[1], Fraction(1)]]
                    
                    def det4(m):
                        return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1])
                              - m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0])
                              + m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0])
                              - m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
                    
                    if det4(a) == 0:
                        return False
        return True
    
    # Try to find a 6th point for D=1
    # Search over a range of integer points
    # The existing points are at: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Let's search in a box around these points
    
    # For D=1, candidate points must have integer coordinates
    # and integer distances to all existing points
    
    # Try a grid search
    # Limit the search to reasonable bounds
    x_min, x_max = -100, 100
    y_min, y_max = -50, 150
    
    found_6 = False
    for x in range(x_min, x_max+1):
        if _time_left() < 10:
            break
        for y in range(y_min, y_max+1):
            if _time_left() < 5:
                break
            candidate = (Fraction(x), Fraction(y))
            # Skip if already in the set
            if candidate in existing:
                continue
            if can_add(candidate, existing, 1):
                new_set = existing + [candidate]
                if _validate(new_set, 1):
                    best_points = new_set
                    best_D = 1
                    best_count = 6
                    found_6 = True
                    break
        if found_6 or _time_left() < 10:
            break
    
    # If we found 6, try to find a 7th
    if best_count >= 6 and _time_left() > 20:
        existing = best_points
        found_7 = False
        # Wider search
        x_min2, x_max2 = -200, 200
        y_min2, y_max2 = -100, 300
        
        for x in range(x_min2, x_max2+1):
            if _time_left() < 15:
                break
            for y in range(y_min2, y_max2+1):
                if _time_left() < 5:
                    break
                candidate = (Fraction(x), Fraction(y))
                if candidate in existing:
                    continue
                if can_add(candidate, existing, 1):
                    new_set = existing + [candidate]
                    if _validate(new_set, 1):
                        best_points = new_set
                        best_D = 1
                        best_count = 7
                        found_7 = True
                        break
            if found_7 or _time_left() < 15:
                break
    
    # Try other D values if time permits
    if _time_left() > 30 and best_count < 7:
        # Try D=2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97
        # This is getting complex. Let's stick with D=1 for now.
        pass
    
    return _format_output(best_points, best_D)
