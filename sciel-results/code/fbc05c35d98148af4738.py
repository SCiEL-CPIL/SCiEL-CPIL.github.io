"""Search-based construction of integral point sets in general position."""
import time
import random
import math
from math import isqrt
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_squarefree(n):
    if n <= 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True

def _g2(p1, p2, D=1):
    """Return squared distance as Fraction between two points (x1,y1), (x2,y2) in Q(sqrt(D))."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + D * dy * dy

def _is_rational_dist(p1, p2, D=1):
    """Check if distance between p1 and p2 is rational (exact)."""
    d2 = _g2(p1, p2, D)
    if d2 < 0:
        return False
    num = d2.numerator
    den = d2.denominator
    if num < 0:
        return False
    rn = isqrt(num)
    rd = isqrt(den)
    return rn * rn == num and rd * rd == den

def _no_three_collinear(pts):
    """Check no three points are collinear using exact rational arithmetic."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(pts):
    """Check no four points are concyclic using exact rational arithmetic."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                    rows = []
                    for p in (p1, p2, p3, p4):
                        x, y = p
                        rows.append([x, y, x * x + y * y, Fraction(1)])
                    a, b, c, d = rows[0]
                    e, f, g, h = rows[1]
                    i_, j_, k_, l_ = rows[2]
                    m, n_, o, p_ = rows[3]
                    det = (a * (f * (k_ * p_ - l_ * o) - g * (j_ * p_ - l_ * n_) + h * (j_ * o - k_ * n_))
                           - b * (e * (k_ * p_ - l_ * o) - g * (i_ * p_ - l_ * m) + h * (i_ * o - k_ * m))
                           + c * (e * (j_ * p_ - l_ * n_) - f * (i_ * p_ - l_ * m) + h * (i_ * n_ - j_ * m))
                           - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
                    if det == 0:
                        return False
    return True

def _validate(points_xy, D=1):
    """Validate a set of points: rational distances, no 3 collinear, no 4 concyclic."""
    if len(points_xy) < 3:
        return False
    # Check distinctness
    for i in range(len(points_xy)):
        for j in range(i + 1, len(points_xy)):
            if points_xy[i][0] == points_xy[j][0] and points_xy[i][1] == points_xy[j][1]:
                return False
    # Check all pairwise distances are rational
    for i in range(len(points_xy)):
        for j in range(i + 1, len(points_xy)):
            if not _is_rational_dist(points_xy[i], points_xy[j], D):
                return False
    # Check no three collinear
    if not _no_three_collinear(points_xy):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points_xy):
        return False
    return True

def _validate_partial(new_pt, existing_pts, D=1):
    """Check if new_pt can be added to existing_pts without violating constraints."""
    # Check distinctness
    for p in existing_pts:
        if new_pt[0] == p[0] and new_pt[1] == p[1]:
            return False
    # Check rational distance to all existing points
    for p in existing_pts:
        if not _is_rational_dist(new_pt, p, D):
            return False
    # Check no three collinear with any pair of existing points
    n = len(existing_pts)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = existing_pts[i], existing_pts[j]
            cross = (p2[0] - p1[0]) * (new_pt[1] - p1[1]) - (p2[1] - p1[1]) * (new_pt[0] - p1[0])
            if cross == 0:
                return False
    # Check no four concyclic with any triple of existing points
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = existing_pts[i], existing_pts[j], existing_pts[k]
                rows = [
                    [p1[0], p1[1], p1[0] * p1[0] + p1[1] * p1[1], Fraction(1)],
                    [p2[0], p2[1], p2[0] * p2[0] + p2[1] * p2[1], Fraction(1)],
                    [p3[0], p3[1], p3[0] * p3[0] + p3[1] * p3[1], Fraction(1)],
                    [new_pt[0], new_pt[1], new_pt[0] * new_pt[0] + new_pt[1] * new_pt[1], Fraction(1)]
                ]
                a, b, c, d = rows[0]
                e, f, g, h = rows[1]
                i_, j_, k_, l_ = rows[2]
                m, n_, o, p_ = rows[3]
                det = (a * (f * (k_ * p_ - l_ * o) - g * (j_ * p_ - l_ * n_) + h * (j_ * o - k_ * n_))
                       - b * (e * (k_ * p_ - l_ * o) - g * (i_ * p_ - l_ * m) + h * (i_ * o - k_ * m))
                       + c * (e * (j_ * p_ - l_ * n_) - f * (i_ * p_ - l_ * m) + h * (i_ * n_ - j_ * m))
                       - d * (e * (j_ * o - k_ * n_) - f * (i_ * o - k_ * m) + g * (i_ * n_ - j_ * m)))
                if det == 0:
                    return False
    return True

def construct():
    """Construct a valid integral point set, maximizing the number of points."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_points = None
    best_D = 1
    best_count = 0
    
    # Start with the known 5-point solution
    known_points = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                    (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                    (Fraction(16), Fraction(63))]
    if _validate(known_points, 1):
        best_points = known_points
        best_D = 1
        best_count = len(known_points)
    
    random.seed(42)
    
    # Strategy 1: Try to extend the known 5-point set with a 6th point
    if time.time() < deadline:
        base_points = known_points
        # Search for a 6th point that satisfies all constraints
        for attempt in range(200000):
            if time.time() >= deadline:
                break
            x = Fraction(random.randint(-200, 200), 1)
            y = Fraction(random.randint(-200, 200), 1)
            new_pt = (x, y)
            if _validate_partial(new_pt, base_points, 1):
                candidate = base_points + [new_pt]
                if _validate(candidate, 1):
                    best_points = candidate
                    best_D = 1
                    best_count = len(candidate)
                    # Try to add a 7th point
                    if time.time() < deadline:
                        for attempt2 in range(50000):
                            if time.time() >= deadline:
                                break
                            x2 = Fraction(random.randint(-200, 200), 1)
                            y2 = Fraction(random.randint(-200, 200), 1)
                            new_pt2 = (x2, y2)
                            if _validate_partial(new_pt2, candidate, 1):
                                candidate2 = candidate + [new_pt2]
                                if _validate(candidate2, 1):
                                    best_points = candidate2
                                    best_D = 1
                                    best_count = len(candidate2)
                                    break
                    break
    
    # Strategy 2: Random search for 6+ points
    if time.time() < deadline:
        for target_n in range(6, 9):
            if best_count >= target_n:
                continue
            for attempt in range(100000):
                if time.time() >= deadline:
                    break
                D = random.choice([1, 1, 1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97])
                if not _is_squarefree(D):
                    continue
                points = []
                valid = True
                for _ in range(target_n):
                    x = Fraction(random.randint(-100, 100), 1)
                    y = Fraction(random.randint(-100, 100), 1)
                    new_pt = (x, y)
                    if not _validate_partial(new_pt, points, D):
                        valid = False
                        break
                    points.append(new_pt)
                if valid and len(points) == target_n:
                    if _validate(points, D):
                        best_points = points
                        best_D = D
                        best_count = len(points)
                        break
    
    # Strategy 3: Local perturbation of known points
    if time.time() < deadline:
        for attempt in range(50000):
            if time.time() >= deadline:
                break
            D = 1
            points = []
            # Start with known points and perturb slightly
            for i, (x, y) in enumerate(known_points):
                dx = random.randint(-10, 10)
                dy = random.randint(-10, 10)
                points.append((x + dx, y + dy))
            # Try to add a 6th point
            for _ in range(100):
                if time.time() >= deadline:
                    break
                x = Fraction(random.randint(-100, 100), 1)
                y = Fraction(random.randint(-100, 100), 1)
                new_pt = (x, y)
                if _validate_partial(new_pt, points, D):
                    candidate = points + [new_pt]
                    if _validate(candidate, D):
                        best_points = candidate
                        best_D = D
                        best_count = len(candidate)
                        break
            if best_count >= 6:
                break
    
    # Final validation
    if best_points is None:
        # Fallback to known valid set
        best_points = known_points
        best_D = 1
    
    # Format output
    result_points = []
    for x, y in best_points:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': result_points}

# EVOLVE-BLOCK-END
