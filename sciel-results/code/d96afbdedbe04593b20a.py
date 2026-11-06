"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _sq(n):
    return n * n

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _is_squarefree(n):
    if n == 1:
        return True
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _check_set(points, D):
    """
    Check a set of points (list of (x_num, x_den, y_num, y_den)) with squarefree D.
    Returns True if valid: distinct, rational distances, no 3 collinear, no 4 concyclic.
    All exact rational arithmetic.
    """
    n = len(points)
    if n < 3:
        return False

    # Convert to fractions
    pts = []
    for (xn, xd, yn, yd) in points:
        x = Fraction(xn, xd)
        y = Fraction(yn, yd)
        pts.append((x, y))

    # Check distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if pts[i] == pts[j]:
                return False

    # Check rational distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            dist_sq = dx * dx + D * dy * dy
            if dist_sq < 0:
                return False
            if not _is_square(dist_sq.numerator * dist_sq.denominator):
                # dist_sq must be a perfect square rational
                # dist_sq = p/q, need p*q to be a perfect square
                num = dist_sq.numerator
                den = dist_sq.denominator
                if not _is_square(num * den):
                    return False

    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (x1, y1) = pts[i]
                (x2, y2) = pts[j]
                (x3, y3) = pts[k]
                # Cross product: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False

    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1) = pts[i]
                    (x2, y2) = pts[j]
                    (x3, y3) = pts[k]
                    (x4, y4) = pts[l]
                    # Four points concyclic iff determinant is zero
                    # Using the general circle equation: x^2 + y^2*D + Ax + By + C = 0
                    # For point (x,y): x^2 + D*y^2 + A*x + B*y + C = 0
                    # Set up linear system for A, B, C for 3 points, check 4th
                    # x1^2 + D*y1^2 + A*x1 + B*y1 + C = 0
                    # x2^2 + D*y2^2 + A*x2 + B*y2 + C = 0
                    # x3^2 + D*y3^2 + A*x3 + B*y3 + C = 0
                    # x4^2 + D*y4^2 + A*x4 + B*y4 + C = 0
                    
                    s1 = x1 * x1 + D * y1 * y1
                    s2 = x2 * x2 + D * y2 * y2
                    s3 = x3 * x3 + D * y3 * y3
                    s4 = x4 * x4 + D * y4 * y4
                    
                    # Solve for A, B, C from first three, check fourth
                    # Matrix: [[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]] [A, B, C]^T = [-s1, -s2, -s3]
                    # Determinant of coefficient matrix
                    det = x1 * (y2 * 1 - 1 * y3) - y1 * (x2 * 1 - 1 * x3) + 1 * (x2 * y3 - y2 * x3)
                    if det == 0:
                        # Points 1,2,3 collinear - already checked, but skip
                        continue
                    # Cramer's rule for A
                    detA = (-s1) * (y2 * 1 - 1 * y3) - y1 * ((-s2) * 1 - 1 * (-s3)) + 1 * (x2 * (-s3) - y2 * (-s2))
                    A = Fraction(detA, det)
                    # Check fourth point
                    residual = x4 * x4 + D * y4 * y4 + A * x4
                    # Need B and C too... let's use a different approach
                    
                    # Actually, let's use the determinant of the 4x4 matrix directly
                    # | x_i^2 + D*y_i^2, x_i, y_i, 1 | = 0 for concyclic
                    det4 = (x1 * x1 + D * y1 * y1) * (x2 * (y3 - y4) - y2 * (x3 - x4) + (x3 * y4 - y3 * x4)) \
                          - x1 * (s2 * (y3 - y4) - y2 * (s3 - s4) + (s3 * y4 - y3 * s4)) \
                          + y1 * (s2 * (x3 - x4) - x2 * (s3 - s4) + (s3 * x4 - x3 * s4)) \
                          + 1 * (s2 * (x3 * y4 - y3 * x4) - x2 * (s3 * y4 - y3 * s4) + y2 * (s3 * x4 - x3 * s4))
                    if det4 == 0:
                        return False

    return True

def _try_add_point(points, D, candidates):
    """Try to add a candidate point to the set. Returns True if valid."""
    new_pts = points + [candidates]
    return _check_set(new_pts, D)

def _generate_candidates_near(points, D, max_radius=500):
    """Generate candidate points near existing points."""
    candidates = set()
    # Try integer points in a box around the existing points
    if not points:
        return candidates
    min_x = min(p[0] for p in points) - max_radius
    max_x = max(p[0] for p in points) + max_radius
    min_y = min(p[2] for p in points) - max_radius
    max_y = max(p[2] for p in points) + max_radius
    
    # Limit the number of candidates
    step = 1
    count = 0
    for x in range(int(min_x), int(max_x) + 1, step):
        for y in range(int(min_y), int(max_y) + 1, step):
            candidates.add((x, 1, y, 1))
            count += 1
            if count > 200000:
                return candidates
    return candidates

def _local_search(points, D, iterations=1000):
    """Try to grow the set by adding points one at a time."""
    current = list(points)
    for _ in range(iterations):
        if _time_left() < 5:
            break
        # Try to add a random integer point
        added = False
        for _ in range(50):
            x = random.randint(-500, 500)
            y = random.randint(-500, 500)
            candidate = (x, 1, y, 1)
            if _check_set(current + [candidate], D):
                current.append(candidate)
                added = True
                break
        if not added:
            break
    return current

def construct():
    # Start with a known valid set of 5 points
    base_points = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    D = 1
    
    best = base_points
    
    # Verify base is valid
    if _check_set(base_points, D):
        best = list(base_points)
    else:
        # Fallback to simpler set
        best = [(0, 1, 0, 1), (3, 1, 4, 1), (6, 1, 8, 1)]
        # Actually let's use a known valid triangle
        best = [(0, 1, 0, 1), (3, 1, 0, 1), (0, 1, 4, 1)]
    
    # Try to grow from base
    if _time_left() > 10:
        grown = _local_search(best, D, iterations=200)
        if len(grown) > len(best):
            best = grown
    
    # Try different starting configurations
    if _time_left() > 30:
        # Try to find a 6-point set by systematic search
        # Use a grid search with constraints
        best_6 = None
        # Start with 4 points and try to add 2 more
        base4 = [(0, 1, 0, 1), (5, 1, 0, 1), (0, 1, 12, 1), (9, 1, 12, 1)]
        if _check_set(base4, D):
            # Try to add points
            for x1 in range(-20, 21):
                if _time_left() < 5:
                    break
                for y1 in range(-20, 21):
                    if _time_left() < 5:
                        break
                    c1 = (x1, 1, y1, 1)
                    if not _check_set(base4 + [c1], D):
                        continue
                    for x2 in range(-20, 21):
                        if _time_left() < 5:
                            break
                        for y2 in range(-20, 21):
                            if _time_left() < 5:
                                break
                            c2 = (x2, 1, y2, 1)
                            if _check_set(base4 + [c1, c2], D):
                                best_6 = base4 + [c1, c2]
                                break
                        if best_6:
                            break
                    if best_6:
                        break
                if best_6:
                    break
        if best_6 and len(best_6) > len(best):
            best = best_6
    
    # Final attempt with more time
    if _time_left() > 20 and len(best) < 8:
        grown = _local_search(best, D, iterations=500)
        if len(grown) > len(best):
            best = grown
    
    # Format output
    points_out = []
    for (xn, xd, yn, yd) in best:
        points_out.append([[xn, xd], [yn, yd]])
    
    return {'D': D, 'points': points_out}
# EVOLVE-BLOCK-END
