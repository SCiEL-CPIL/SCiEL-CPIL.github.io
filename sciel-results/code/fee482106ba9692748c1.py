"""Integral point sets in general position.
Strategy:
1. Try to construct 6+ point sets by searching over rational coordinates.
2. For D=1 (integer points), search for 6-point sets with integer distances,
   no 3 collinear, no 4 concyclic.
3. If not found within time budget, fall back to the known 5-point set.
"""
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
TIME_LIMIT = 110  # seconds, safety margin before 120s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _is_square(n):
    """Check if n is a perfect square (for non-negative integers)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist_sq_is_square(px, py, qx, qy, D):
    """
    Check if squared distance between (px, py*sqrt(D)) and (qx, qy*sqrt(D)) is a perfect square.
    dist^2 = (px-qx)^2 + D*(py-qy)^2
    For this to be a rational squared distance, we need (px-qx)^2 + D*(py-qy)^2 to be a perfect square of a rational.
    If all coords are integers and D=1, just check if it's a perfect square integer.
    For general case with Fraction coords, we check if the value is a square of a Fraction.
    """
    dx = px - qx
    dy = py - qy
    # dist^2 = dx^2 + D * dy^2
    val = dx * dx + D * dy * dy
    # Check if val is a perfect square of a Fraction
    if val < 0:
        return False
    # For Fraction: val = num/den. Check if num/den is a square.
    # This means num must be a perfect square * k^2 and den must be a perfect square * k^2 for some k,
    # but more simply: val is a square iff there exists a Fraction f such that f^2 = val.
    # For a Fraction num/den in lowest terms, it's a square iff num and den are both perfect squares.
    if isinstance(val, Fraction):
        num = val.numerator
        den = val.denominator
        return _is_square(num) and _is_square(den)
    else:
        # Integer
        return _is_square(int(val))

def _check_all_distances_rational(points, D):
    """Check all pairwise distances are rational (squared distance is a perfect square)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            px, py = points[i]
            qx, qy = points[j]
            if not _dist_sq_is_square(px, py, qx, qy, D):
                return False
    return True

def _check_no_3_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # Cross product (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                # Note: y coordinates here are the rational parts (multiplied by sqrt(D) in actual geometry)
                # For collinearity in the (x, y*sqrt(D)) plane:
                # (x2-x1)*(y3-y1)*sqrt(D) - (y2-y1)*sqrt(D)*(x3-x1) = 0
                # This simplifies to (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1) = 0 (sqrt(D) cancels if D>0)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _check_no_4_concyclic(points, D):
    """
    Check no four points are concyclic.
    Four points are concyclic iff the determinant of the circle equation is zero.
    For points (xi, yi*sqrt(D)), a circle has equation:
    x^2 + y^2*D + Ax + By*sqrt(D) + C = 0
    Wait, let's be more careful. A circle in the (x, y) plane (where actual y = y_rational * sqrt(D)):
    x^2 + (y_rational*sqrt(D))^2 + ax + b*(y_rational*sqrt(D)) + c = 0
    x^2 + D*y_rational^2 + ax + b*sqrt(D)*y_rational + c = 0
    
    For this to hold for rational x, y_rational, we can separate rational and irrational parts (if D is not a square):
    Rational part: x^2 + D*y_rational^2 + ax + c = 0
    Irrational part: b*sqrt(D)*y_rational = 0 => b*y_rational = 0
    
    This approach is tricky. Let's use the determinant method.
    Four points (x1,y1), (x2,y2), (x3,y3), (x4,y4) are concyclic iff:
    |x1^2+y1^2, x1, y1, 1|
    |x2^2+y2^2, x2, y2, 1| = 0
    |x3^2+y3^2, x3, y3, 1|
    |x4^2+y4^2, x4, y4, 1|
    
    But here y_i = y_rational_i * sqrt(D). So y_i^2 = D * y_rational_i^2.
    The determinant involves terms like x_i^2 + D*y_rational_i^2, x_i, y_rational_i*sqrt(D), 1.
    
    This determinant will be of the form A + B*sqrt(D) where A, B are rational.
    For it to be zero, both A and B must be zero.
    
    Let's compute it. Let's denote the matrix M with rows [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1].
    The determinant is linear in each row. Let's expand.
    
    Actually, let's use a different approach. Three non-collinear points define a unique circle.
    The fourth point is concyclic iff it lies on that circle.
    
    Let's find the circle through points 0, 1, 2 and check if point 3 is on it.
    
    Circle equation: x^2 + D*y^2 + ax + b*y*sqrt(D) + c = 0
    For point i: x_i^2 + D*y_i^2 + a*x_i + b*y_i*sqrt(D) + c = 0
    
    This is a system in a, b*sqrt(D), c. Let b' = b*sqrt(D).
    x_i^2 + D*y_i^2 + a*x_i + b'*y_i + c = 0
    
    For three points, we can solve for a, b', c.
    Then check if the fourth point satisfies the equation.
    
    The system is:
    x0*x0 + D*y0*y0 + a*x0 + b'*y0 + c = 0
    x1*x1 + D*y1*y1 + a*x1 + b'*y1 + c = 0
    x2*x2 + D*y2*y2 + a*x2 + b'*y2 + c = 0
    
    Let's write this as M * [a, b', c]^T = v
    where M = [[x0, y0, 1], [x1, y1, 1], [x2, y2, 1]]
    and v = [-x0^2 - D*y0^2, -x1^2 - D*y1^2, -x2^2 - D*y2^2]
    
    If M is invertible (which it is if no 3 collinear), then [a, b', c] is uniquely determined.
    Then check if x3^2 + D*y3^2 + a*x3 + b'*y3 + c == 0.
    
    All computations are in Q (rationals), since x_i, y_i, D are rational/integer.
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Check if points i, j, k, l are concyclic
                    x0, y0 = points[i]
                    x1, y1 = points[j]
                    x2, y2 = points[k]
                    x3, y3 = points[l]
                    
                    # Build 3x3 system for a, b', c
                    # M = [[x0, y0, 1], [x1, y1, 1], [x2, y2, 1]]
                    # v = [-x0^2 - D*y0^2, -x1^2 - D*y1^2, -x2^2 - D*y2^2]
                    
                    # Determinant of M
                    det = (x0 * (y1 * 1 - 1 * y2) - y0 * (x1 * 1 - 1 * x2) + 1 * (x1 * y2 - y1 * x2))
                    if det == 0:
                        # Three points collinear, already checked, but just in case
                        continue
                    
                    # Cramer's rule
                    # a = det_a / det
                    det_a = (-x0*x0 - D*y0*y0) * (y1 * 1 - 1 * y2) - y0 * ((-x1*x1 - D*y1*y1) * 1 - 1 * (-x2*x2 - D*y2*y2)) + 1 * (x1 * (-x2*x2 - D*y2*y2) - y1 * (-x2*x2 - D*y2*y2))
                    # Wait, let me redo Cramer's rule carefully.
                    # M = [[x0, y0, 1], [x1, y1, 1], [x2, y2, 1]]
                    # v = [v0, v1, v2] where vi = -xi^2 - D*yi^2
                    
                    # a = det([[v0, y0, 1], [v1, y1, 1], [v2, y2, 1]]) / det
                    # b' = det([[x0, v0, 1], [x1, v1, 1], [x2, v2, 1]]) / det
                    # c = det([[x0, y0, v0], [x1, y1, v1], [x2, y2, v2]]) / det
                    
                    v0 = -x0*x0 - D*y0*y0
                    v1 = -x1*x1 - D*y1*y1
                    v2 = -x2*x2 - D*y2*y2
                    
                    # det_a (replace first column with v)
                    det_a = v0*(y1*1 - 1*y2) - y0*(v1*1 - 1*v2) + 1*(v1*y2 - y1*v2)
                    # det_b (replace second column with v)
                    det_b = x0*(v1*1 - 1*v2) - v0*(x1*1 - 1*x2) + 1*(x1*v2 - v1*x2)
                    # det_c (replace third column with v)
                    det_c = x0*(y1*v2 - 1*v1) - y0*(x1*v2 - 1*v2) + v0*(x1*y2 - y1*x2)
                    
                    a = det_a / det
                    b_prime = det_b / det
                    c = det_c / det
                    
                    # Check if point 3 is on the circle
                    val = x3*x3 + D*y3*y3 + a*x3 + b_prime*y3 + c
                    if val == 0:
                        return False
    return True

def _validate(points, D):
    """Full validation of a point set."""
    if len(points) < 3:
        return False
    if not _check_all_distances_rational(points, D):
        return False
    if not _check_no_3_collinear(points):
        return False
    if not _check_no_4_concyclic(points, D):
        return False
    return True

def _try_integer_search(max_coord=200, max_points=6):
    """
    Search for integer point sets with integer distances, no 3 collinear, no 4 concyclic.
    We'll do a greedy/beam search: start with 2 points, add points one by one.
    """
    best = None
    best_size = 0
    
    # Generate candidate points in a grid
    # To keep it manageable, we'll use a smaller grid
    grid_size = 50
    candidates = []
    for x in range(-grid_size, grid_size + 1):
        for y in range(-grid_size, grid_size + 1):
            candidates.append((Fraction(x), Fraction(y)))
    
    # We'll use a simple approach: try to build sets by adding points
    # Start with pairs that have integer distance
    valid_pairs = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            x1, y1 = candidates[i]
            x2, y2 = candidates[j]
            if _dist_sq_is_square(x1, y1, x2, y2, 1):
                valid_pairs.append((candidates[i], candidates[j]))
    
    if not valid_pairs:
        return None
    
    # Greedy: start with a pair, try to add points
    # To limit search, we'll try a subset of pairs
    import random
    random.seed(42)
    
    # Try multiple starting pairs
    tried_pairs = set()
    for _ in range(200):  # Try 200 random starting pairs
        if _time_remaining() < 10:
            break
        if not valid_pairs:
            break
        pair = random.choice(valid_pairs)
        key = (pair[0], pair[1])
        if key in tried_pairs:
            continue
        tried_pairs.add(key)
        
        current_set = [pair[0], pair[1]]
        
        # Try to add more points
        for _ in range(max_points - 2):
            if _time_remaining() < 10:
                break
            added = False
            # Try random candidates
            for _ in range(50):
                if _time_remaining() < 10:
                    break
                cand = random.choice(candidates)
                if cand in current_set:
                    continue
                new_set = current_set + [cand]
                if _check_all_distances_rational(new_set, 1) and _check_no_3_collinear(new_set):
                    # Check concyclicity only if we have 4+ points
                    if len(new_set) >= 4:
                        if not _check_no_4_concyclic(new_set, 1):
                            continue
                    current_set = new_set
                    added = True
                    break
            if not added:
                break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best = current_set
            if best_size >= max_points:
                break
    
    return best

def _known_5_points():
    """Return the known 5-point set."""
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return [(Fraction(x), Fraction(y)) for x, y in points]

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # First, try to find a 6-point set
    best_6 = _try_integer_search(max_coord=100, max_points=6)
    
    if best_6 and len(best_6) >= 6:
        # Validate
        if _validate(best_6, 1):
            return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_6]}
    
    # Fall back to 5-point set
    points_5 = _known_5_points()
    if _validate(points_5, 1):
        return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points_5]}
    
    # Absolute fallback: return the known set (should always be valid)
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
