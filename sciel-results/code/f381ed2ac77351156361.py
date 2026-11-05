"""Program for constructing integral point sets in general position."""
import random
from fractions import Fraction
import math
import time

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

def rational_dist_sq(p1, p2, D):
    """Compute squared distance between two points.
    Points are given as (x_num, x_den, y_num, y_den)."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    dx = x1 - x2
    dy = y1 - y2
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

def no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1 = Fraction(points[i][0], points[i][1])
                y1 = Fraction(points[i][2], points[i][3])
                x2 = Fraction(points[j][0], points[j][1])
                y2 = Fraction(points[j][2], points[j][3])
                x3 = Fraction(points[k][0], points[k][1])
                y3 = Fraction(points[k][2], points[k][3])
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    return False
    return True

def _four_concyclic_check(p1, p2, p3, p4, D):
    """Check if 4 points are concyclic using determinant."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    x4 = Fraction(p4[0], p4[1])
    y4 = Fraction(p4[2], p4[3])
    
    u1 = x1 * x1 + D * y1 * y1
    u2 = x2 * x2 + D * y2 * y2
    u3 = x3 * x3 + D * y3 * y3
    u4 = x4 * x4 + D * y4 * y4
    
    # Determinant of [[u_i, x_i, y_i, 1]]
    a11, a12, a13, a14 = u1, x1, y1, Fraction(1)
    a21, a22, a23, a24 = u2, x2, y2, Fraction(1)
    a31, a32, a33, a34 = u3, x3, y3, Fraction(1)
    a41, a42, a43, a44 = u4, x4, y4, Fraction(1)
    
    det = (a11 * (a22 * a33 * a44 - a22 * a34 * a43 - a23 * a32 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a24 * a33 * a42)
         - a12 * (a21 * a33 * a44 - a21 * a34 * a43 - a23 * a31 * a44 + a23 * a34 * a41 + a24 * a31 * a43 - a24 * a33 * a41)
         + a13 * (a21 * a32 * a44 - a21 * a34 * a42 - a22 * a31 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a24 * a32 * a41)
         - a14 * (a21 * a32 * a43 - a21 * a33 * a42 - a22 * a31 * a43 + a22 * a33 * a41 + a23 * a31 * a42 - a23 * a32 * a41))
    
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

def _three_collinear_check(p1, p2, p3):
    """Check if three points are collinear."""
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3])
    x2 = Fraction(p2[0], p2[1])
    y2 = Fraction(p2[2], p2[3])
    x3 = Fraction(p3[0], p3[1])
    y3 = Fraction(p3[2], p3[3])
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _check_point_valid(p, points, D):
    """Check if point p can be added to points."""
    # Check rational distances to all existing points
    for q in points:
        d2 = rational_dist_sq(p, q, D)
        if not is_perfect_square_frac(d2):
            return False
    # Check no three collinear
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _three_collinear_check(p, points[i], points[j]):
                return False
    # Check no four concyclic (only if we'd have 4+ points)
    if len(points) >= 3:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    if _four_concyclic_check(p, points[i], points[j], points[k], D):
                        return False
    return True

def _generate_candidate_point(D, max_coord=1000):
    """Generate a random point (x, y) with x^2 + D*y^2 = z^2 (perfect square)."""
    # Use parameterization: for D=1, x = m^2-n^2, y = 2mn, z = m^2+n^2
    # For general D, try random and check
    for _ in range(50):
        x = random.randint(-max_coord, max_coord)
        y = random.randint(-max_coord, max_coord)
        if x == 0 and y == 0:
            continue
        val = x * x + D * y * y
        z = int(math.isqrt(val))
        if z * z == val:
            return (x, 1, y, 1)
    return None

def _generate_candidate_point_param(D, max_m=50):
    """Generate points using Pythagorean-like parameterization."""
    # For D=1: x = m^2-n^2, y = 2mn
    # For general D: try to find m,n such that (m^2-n^2)^2 + D*(2mn)^2 is a square
    # This gives m^4 - 2m^2n^2 + n^4 + 4D*m^2*n^2 = m^4 + (4D-2)m^2n^2 + n^4
    # For this to be a square, we need specific conditions
    for m in range(2, max_m):
        for n in range(1, m):
            x = m * m - n * n
            y = 2 * m * n
            val = x * x + D * y * y
            z = int(math.isqrt(val))
            if z * z == val:
                return (x, 1, y, 1)
    return None

def hill_climb(D, target_n, time_limit=100):
    """Hill-climbing search for a set of target_n points."""
    start_time = time.time()
    
    # Start with origin
    points = [(0, 1, 0, 1)]
    
    # Try to add points one by one
    for target in range(2, target_n + 1):
        found = False
        max_attempts = 5000
        attempts = 0
        while attempts < max_attempts and not found:
            if time.time() - start_time > time_limit:
                break
            # Generate candidate
            if random.random() < 0.5:
                p = _generate_candidate_point(D, max_coord=200)
            else:
                p = _generate_candidate_point_param(D, max_m=30)
            
            if p is None:
                attempts += 1
                continue
            
            if _check_point_valid(p, points, D):
                points.append(p)
                found = True
            attempts += 1
        
        if not found:
            break
    
    return points

def construct():
    """Construct a set of integral points in general position."""
    random.seed(42)
    start_time = time.time()
    
    # Known 5-point set as baseline
    base_points = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    
    best_points = base_points
    best_D = 1
    
    # Try hill-climbing with various D values
    D_candidates = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95]
    
    for D in D_candidates:
        if time.time() - start_time > 100:
            break
        points = hill_climb(D, 7, time_limit=100 - (time.time() - start_time))
        if len(points) > len(best_points) and validate(points, D):
            best_points = points
            best_D = D
    
    # Final validation
    if validate(best_points, best_D):
        return {'D': best_D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in best_points]}
    
    # Fallback
    fallback = [(0, 1, 0, 1), (3, 1, 4, 1), (0, 1, 4, 1)]
    if validate(fallback, 1):
        return {'D': 1, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in fallback]}
    
    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [4, 1]]]}

# EVOLVE-BLOCK-END
