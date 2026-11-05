"""Program for constructing integral point sets in general position."""
import random
from fractions import Fraction
import math

# EVOLVE-BLOCK-START

def is_squarefree(n):
    """Check if n is squarefree."""
    if n < 0:
        n = -n
    if n == 0:
        return False
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def make_squarefree(n):
    """Make n squarefree by removing square factors."""
    if n < 0:
        sign = -1
        n = -n
    else:
        sign = 1
    if n == 0:
        return 0
    result = n
    i = 2
    while i * i <= result:
        while result % (i * i) == 0:
            result //= (i * i)
        i += 1
    return sign * result

def rational_dist_sq(p1, p2, D):
    """Compute squared distance between two points.
    Points are given as (x_num, x_den, y_num, y_den)."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    
    dx = x1 - x2
    dy = y1 - y2
    # dist^2 = dx^2 + D * dy^2
    return dx * dx + D * dy * dy

def is_perfect_square_frac(frac):
    """Check if a Fraction is a perfect square of a rational number."""
    if frac < 0:
        return False
    if frac == 0:
        return True
    num = frac.numerator
    den = frac.denominator
    sqrt_num = int(math.isqrt(num))
    sqrt_den = int(math.isqrt(den))
    if sqrt_num * sqrt_num != num:
        return False
    if sqrt_den * sqrt_den != den:
        return False
    return True

def all_pairwise_rational(points, D):
    """Check if all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = rational_dist_sq(points[i], points[j], D)
            if not is_perfect_square_frac(d2):
                return False
    return True

def _three_collinear_check(p1, p2, p3):
    """Check if three points are collinear."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _three_collinear_check(points[i], points[j], points[k]):
                    return False
    return True

def _four_concyclic_check(p1, p2, p3, p4, D):
    """Check if 4 points are concyclic."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    x4 = Fraction(p4[0], p4[1])
    y4 = Fraction(p4[2], p4[1]) if False else Fraction(p4[2], p4[3])
    
    u1 = x1 * x1 + D * y1 * y1
    u2 = x2 * x2 + D * y2 * y2
    u3 = x3 * x3 + D * y3 * y3
    u4 = x4 * x4 + D * y4 * y4
    
    # Compute determinant of 4x4 matrix
    # |u1, x1, y1, 1|
    # |u2, x2, y2, 1|
    # |u3, x3, y3, 1|
    # |u4, x4, y4, 1|
    
    # Use cofactor expansion along last column
    det = (u1 * (x2 * y3 - x3 * y2) - x1 * (u2 * y3 - u3 * y2) + y1 * (u2 * x3 - u3 * x2)
         - u2 * (x1 * y3 - x3 * y1) + x2 * (u1 * y3 - u3 * y1) - y2 * (u1 * x3 - u3 * x1)
         + u3 * (x1 * y2 - x2 * y1) - x3 * (u1 * y2 - u2 * y1) + y3 * (u1 * x2 - u2 * x1)
         - u4 * (x1 * y2 - x2 * y1) + x4 * (u1 * y2 - u2 * y1) - y4 * (u1 * x2 - u2 * x1))
    
    return det == 0

def no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    if n < 4:
        return True
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic_check(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def validate(points, D):
    """Full validation of the point set."""
    if not is_squarefree(D):
        return False
    if D < 1 or D > 10**12:
        return False
    
    n = len(points)
    if n < 3 or n > 12:
        return False
    
    for p in points:
        for val in [p[0], p[1], p[2], p[3]]:
            if abs(val) > 10**40:
                return False
        if p[1] == 0 or p[3] == 0:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
    
    if not all_pairwise_rational(points, D):
        return False
    
    if not no_three_collinear(points):
        return False
    
    if not no_four_concyclic(points, D):
        return False
    
    return True

def find_point_with_rational_distances(existing_points, D, max_coord=200):
    """Find a new point with rational distance to all existing points."""
    for _ in range(5000):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        if x == 0 and y == 0:
            continue
        p = (x, 1, y, 1)
        
        # Check rational distance to all existing points
        valid = True
        for q in existing_points:
            d2 = rational_dist_sq(p, q, D)
            if not is_perfect_square_frac(d2):
                valid = False
                break
        if not valid:
            continue
        
        # Check no three collinear
        collinear = False
        for i in range(len(existing_points)):
            for j in range(i + 1, len(existing_points)):
                if _three_collinear_check(p, existing_points[i], existing_points[j]):
                    collinear = True
                    break
            if collinear:
                break
        if collinear:
            continue
        
        # Check no four concyclic
        concyclic = False
        for i in range(len(existing_points)):
            for j in range(i + 1, len(existing_points)):
                for k in range(j + 1, len(existing_points)):
                    if _four_concyclic_check(p, existing_points[i], existing_points[j], existing_points[k], D):
                        concyclic = True
                        break
                if concyclic:
                    break
            if concyclic:
                break
        if concyclic:
            continue
        
        return p
    
    return None

def construct():
    """Construct a set of integral points in general position."""
    # Known 5-point set with D=1
    base_points = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    
    # Validate the base set
    if validate(base_points, 1):
        # Try to extend to 6+ points
        best_set = base_points
        best_D = 1
        
        for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47]:
            # Try to build up from 3 points
            for max_coord in [100, 200, 500]:
                # Start with origin
                points = [(0, 1, 0, 1)]
                
                # Add points one by one
                for target in range(4, 9):  # Try to get up to 8 points
                    found = False
                    for attempt in range(100):
                        p = find_point_with_rational_distances(points, D, max_coord)
                        if p is not None:
                            points.append(p)
                            found = True
                            break
                    
                    if not found and len(points) >= 3:
                        break
                    if len(points) > len(best_set):
                        if validate(points, D):
                            best_set = points[:]
                            best_D = D
                            if len(points) >= 8:
                                break
                
                if len(best_set) >= 8:
                    break
        
        if len(best_set) >= 5:
            return {'D': best_D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in best_set]}
        
        # Fallback to base 5-point set
        return {'D': 1, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in base_points]}
    
    # Fallback: return a simple 3-point set
    points = [(0, 1, 0, 1), (3, 1, 4, 1), (6, 1, 8, 1)]
    if validate(points, 1):
        return {'D': 1, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in points]}
    
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[6, 1], [8, 1]]]}

# EVOLVE-BLOCK-END
