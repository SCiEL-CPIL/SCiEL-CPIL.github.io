"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _sqdist(D, p1, p2):
    # p1, p2 are (x, y) rational pairs
    # distance^2 = (x1-x2)^2 + D*(y1-y2)^2
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + D*dy*dy

def _collinear3(p1, p2, p3):
    # Check if three points are collinear
    # (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) == 0
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) != 0

def _concyclic4(p1, p2, p3, p4):
    # Check if four points are concyclic
    # Use the determinant condition:
    # | x1^2+y1^2, x1, y1, 1 |
    # | x2^2+y2^2, x2, y2, 1 | = 0
    # | x3^2+y3^2, x3, y3, 1 |
    # | x4^2+y4^2, x4, y4, 1 |
    # where yi are the actual y coordinates (not y*sqrt(D))
    # But since we're working with rational x and y*sqrt(D), we need to be careful.
    # Actually, for points (x, y*sqrt(D)), the condition for concyclicity is that
    # there exists a circle x^2 + y^2 + Ax + By + C = 0 (in the actual Euclidean plane)
    # passing through all four points. Substituting y = yi*sqrt(D):
    # xi^2 + Di*yi^2 + A*xi + B*yi*sqrt(D) + C = 0
    # This must hold for all four points. We can eliminate sqrt(D) by requiring
    # the coefficient of sqrt(D) to vanish, i.e., sum of yi terms weighted appropriately.
    # Alternatively, use the standard 4x4 determinant with actual coordinates.
    # Let's use the determinant method with exact arithmetic.
    
    # For point (xi, yi*sqrt(D)), define X = xi, Y = yi*sqrt(D)
    # The circle equation: X^2 + Y^2 + aX + bY + c = 0
    # => xi^2 + D*yi^2 + a*xi + b*yi*sqrt(D) + c = 0
    # For this to hold for all 4 points, we need the system to be consistent.
    # Since sqrt(D) is irrational (assuming D is not a perfect square), we separate:
    # (xi^2 + D*yi^2 + a*xi + c) + b*yi*sqrt(D) = 0
    # This gives two equations per point (rational part and sqrt(D) part must both be 0):
    # xi^2 + D*yi^2 + a*xi + c = 0  ... (1)
    # b*yi = 0  ... (2)
    # From (2), either b=0 or yi=0 for all i. If any yi != 0, then b must be 0.
    # So if not all yi are 0, then b=0, and we just need a circle in the X direction... 
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the standard approach: four points are concyclic iff the determinant is zero.
    # For points (xi, yi) in the plane, they are concyclic iff:
    # | xi^2+yi^2, xi, yi, 1 |
    # | xj^2+yj^2, xj, yj, 1 | = 0
    # | xk^2+yk^2, xk, yk, 1 |
    # | xl^2+yl^2, xl, yl, 1 |
    
    # In our case, the actual y-coordinate is yi*sqrt(D). So:
    # actual_yi = yi * sqrt(D)
    # actual_yi^2 = D * yi^2
    
    # So the determinant becomes:
    # | xi^2 + D*yi^2, xi, yi*sqrt(D), 1 |
    # This has sqrt(D) in it. To use exact arithmetic, we can compute this determinant
    # and check if it's zero. But we need to handle the sqrt(D) terms.
    
    # Alternative: use the fact that four points are concyclic iff the cross ratio is real,
    # or use the algebraic condition. Let's just compute the determinant symbolically.
    
    # Let me define the matrix M where M[i] = [xi^2 + D*yi^2, xi, yi, 1] * some factor?
    # No, the correct matrix for concyclicity with points (Xi, Yi) is:
    # | Xi^2+Yi^2, Xi, Yi, 1 |
    # where Yi = yi*sqrt(D).
    
    # So the matrix is:
    # | xi^2 + D*yi^2, xi, yi*sqrt(D), 1 |
    
    # The determinant of this 4x4 matrix must be zero. Since it contains sqrt(D),
    # we can write it as A + B*sqrt(D) where A, B are rational. The determinant is zero
    # iff A = 0 and B = 0.
    
    # This is getting complex. Let me use a simpler numerical check with high precision,
    # or use the property that for our specific construction (integer points with D=1),
    # we can use the standard integer determinant.
    
    # For D=1, the points are just integer points, so we can use the standard test.
    if D == 1:
        # Standard 4x4 determinant for integer points
        def det4(m):
            # m is 4x4 list of lists
            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                   - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                   + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                   - m[0][3]*(m[1][0]*(m[2][1]*m[2][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
        
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        return det4(rows) != 0
    
    # For general D, use exact rational arithmetic with sqrt(D) separation
    # Build the 4x4 matrix with entries involving sqrt(D)
    # M[i][0] = xi^2 + D*yi^2 (rational)
    # M[i][1] = xi (rational)
    # M[i][2] = yi*sqrt(D) (irrational)
    # M[i][3] = 1 (rational)
    
    # Compute determinant as A + B*sqrt(D)
    # Using cofactor expansion along column 2 (the sqrt(D) column):
    # det = sum_i (-1)^(i+2) * M[i][2] * C[i][2]
    # where C[i][2] is the cofactor (3x3 minor)
    
    # M[i][2] = yi * sqrt(D)
    # So det = sqrt(D) * sum_i (-1)^(i+2) * yi * C[i][2]
    # For det = 0, we need sum_i (-1)^(i+2) * yi * C[i][2] = 0
    
    # But wait, this assumes the other entries are rational, which they are.
    # So the determinant is sqrt(D) * S where S is rational.
    # det = 0 iff S = 0.
    
    # Compute S = sum_i (-1)^(i+2) * yi * det(M_i)
    # where M_i is the 3x3 matrix obtained by removing row i and column 2.
    
    def det3(m):
        # m is 3x3
        return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
               - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
               + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    
    S = Fraction(0)
    for i in range(4):
        # Remove row i and column 2
        minor = []
        for r in range(4):
            if r == i:
                continue
            row = []
            for c in range(4):
                if c == 2:
                    continue
                if c == 0:
                    row.append(Fraction(int([p1,p2,p3,p4][r][0])**2 + D*int([p1,p2,p3,p4][r][1])**2))
                elif c == 1:
                    row.append(Fraction(int([p1,p2,p3,p4][r][0])))
                else:
                    row.append(Fraction(1))
            minor.append(row)
        
        sign = 1 if (i+2) % 2 == 0 else -1
        S += sign * int([p1,p2,p3,p4][i][1]) * det3(minor)
    
    return S != 0

def _validate(D, points):
    """Validate a set of points. Returns True if valid."""
    n = len(points)
    if n < 3:
        return False
    
    # Check all pairwise distances are rational (i.e., squared distance is a perfect square of a rational)
    for i in range(n):
        for j in range(i+1, n):
            d2 = _sqdist(D, points[i], points[j])
            # d2 should be a perfect square of a rational number
            # d2 = (a/b)^2 for some integers a, b
            # i.e., d2 * b^2 = a^2, so d2 must be a perfect square rational
            num, den = d2.numerator, d2.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if not _collinear3(points[i], points[j], points[k]):
                    return False
    
    # Check no four concyclic
    for combo in combinations(range(n), 4):
        if not _concyclic4(D, points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with known 5-point configuration
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_D = 1
    
    # Try to find a 6-point configuration by local search
    # Use the known 5 points as seed and try to add a 6th point
    
    # Candidate points to try (small integer coordinates)
    candidates = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            candidates.append((x, y))
    
    # Filter candidates that have rational distance to all existing points
    def _has_rational_dist_to_all(D, p, existing):
        for ep in existing:
            d2 = _sqdist(D, p, ep)
            if not _is_square(d2.numerator) or not _is_square(d2.denominator):
                return False
        return True
    
    def _not_collinear_with_any_pair(D, p, existing):
        for i in range(len(existing)):
            for j in range(i+1, len(existing)):
                if not _collinear3(existing[i], existing[j], p):
                    return False
        return True
    
    def _not_concyclic_with_any_triple(D, p, existing):
        for combo in combinations(range(len(existing)), 3):
            if not _concyclic4(D, existing[combo[0]], existing[combo[1]], existing[combo[2]], p):
                return False
        return True
    
    # Try adding a 6th point to the 5-point set
    for x in range(-200, 201):
        for y in range(-200, 201):
            if time.time() > deadline:
                break
            p = (x, y)
            if p in best_points:
                continue
            
            if _has_rational_dist_to_all(best_D, p, best_points) and \
               _not_collinear_with_any_pair(best_D, p, best_points) and \
               _not_concyclic_with_any_triple(best_D, p, best_points):
                new_points = best_points + [p]
                if _validate(best_D, new_points):
                    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in new_points]}
    
    # If we found a 6-point set, return it. Otherwise return the 5-point set.
    return {'D': best_D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
