import time
import math
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_pairwise_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Distance squared = (x1-x2)^2 + D*(y1-y2)^2
            # We want this to be a square of a rational.
            # dx = x1 - x2, dy = y1 - y2
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx * dx + D * dy * dy
            # d2 is a Fraction. We need to check if it is a perfect square of a Fraction.
            # A Fraction p/q is a square if p and q are perfect squares.
            if d2 < 0:
                return False
            num = d2.numerator
            den = d2.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    return True

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Determinant of matrix [[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]
                # (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    n = len(points)
    # Circle equation: x^2 + y^2 + Dx + Ey + F = 0 (in transformed space?)
    # Actually, in (x, y*sqrt(D)) space, the circle equation is x^2 + D*y^2 + Ax + By + C = 0.
    # We need to check if 4 points satisfy a common equation of this form.
    # This is equivalent to checking if the determinant of the matrix formed by the points is 0.
    # Matrix rows: [x_i^2 + D*y_i^2, x_i, y_i, 1]
    # If det == 0, they are concyclic.
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Construct 4x4 matrix
                    # M = [[x1^2 + D*y1^2, x1, y1, 1], ...]
                    # We can use Gaussian elimination or Cramer's rule.
                    # Since we are using Fractions, exact arithmetic is fine.
                    m = [
                        [p1[0]**2 + D * p1[1]**2, p1[0], p1[1], 1],
                        [p2[0]**2 + D * p2[1]**2, p2[0], p2[1], 1],
                        [p3[0]**2 + D * p3[1]**2, p3[0], p3[1], 1],
                        [p4[0]**2 + D * p4[1]**2, p4[0], p4[1], 1]
                    ]
                    # Check determinant
                    # Simple 4x4 determinant calculation
                    def det4(m):
                        return (
                            m[0][0] * (m[1][1]*m[2][2]*m[3][3] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][3]*m[2][2]*m[3][1]) -
                            m[0][1] * (m[1][0]*m[2][2]*m[3][3] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][3]*m[2][2]*m[3][0]) +
                            m[0][2] * (m[1][0]*m[2][1]*m[3][3] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][3]*m[2][1]*m[3][0]) -
                            m[0][3] * (m[1][0]*m[2][1]*m[3][2] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][2]*m[2][1]*m[3][0])
                        )
                    if det4(m) == 0:
                        return False
    return True

def _validate(points, D):
    if len(points) < 3:
        return False
    if not _check_pairwise_distances_rational(points, D):
        return False
    if not _check_no_three_collinear(points):
        return False
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Baseline 5 points
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Convert to Fractions for calculation
    pts_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    best_set = pts_frac
    best_size = len(pts_frac)
    
    # Search for a 6th point
    # Range for x, y
    # The base points are in [-56, 16] x [0, 63]
    # Try integers in a slightly larger box
    found_6 = False
    for x_num in range(-100, 101):
        for y_num in range(-50, 101):
            if time.time() > deadline:
                break
            new_pt = (Fraction(x_num), Fraction(y_num))
            candidate = pts_frac + [new_pt]
            
            # Quick check: distinct
            if len(set(candidate)) != len(candidate):
                continue
                
            # Check distances to existing points
            valid = True
            for i in range(5):
                dx = new_pt[0] - pts_frac[i][0]
                dy = new_pt[1] - pts_frac[i][1]
                d2 = dx * dx + D * dy * dy
                if not _is_square(d2.numerator) or not _is_square(d2.denominator):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check collinearity with pairs
            for i in range(5):
                for j in range(i+1, 5):
                    p1, p2 = pts_frac[i], pts_frac[j]
                    if (new_pt[0] - p1[0]) * (p2[1] - p1[1]) == (p2[0] - p1[0]) * (new_pt[1] - p1[1]):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
                
            # Check concyclicity with triples
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        p1, p2, p3 = pts_frac[i], pts_frac[j], pts_frac[k]
                        m = [
                            [p1[0]**2 + D * p1[1]**2, p1[0], p1[1], 1],
                            [p2[0]**2 + D * p2[1]**2, p2[0], p2[1], 1],
                            [p3[0]**2 + D * p3[1]**2, p3[0], p3[1], 1],
                            [new_pt[0]**2 + D * new_pt[1]**2, new_pt[0], new_pt[1], 1]
                        ]
                        # Det check
                        # Reuse det4 logic
                        def det4(m):
                            return (
                                m[0][0] * (m[1][1]*m[2][2]*m[3][3] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][3]*m[2][2]*m[3][1]) -
                                m[0][1] * (m[1][0]*m[2][2]*m[3][3] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][3]*m[2][2]*m[3][0]) +
                                m[0][2] * (m[1][0]*m[2][1]*m[3][3] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][3]*m[2][1]*m[3][0]) -
                                m[0][3] * (m[1][0]*m[2][1]*m[3][2] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][2]*m[2][1]*m[3][0])
                            )
                        if det4(m) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_set = candidate
                best_size = 6
                found_6 = True
                break
        if found_6:
            break
            
    # If found 6, try to find 7?
    # Given time, probably not.
    
    # Format output
    out_points = []
    for x, y in best_set:
        out_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
        
    return {'D': D, 'points': out_points}
