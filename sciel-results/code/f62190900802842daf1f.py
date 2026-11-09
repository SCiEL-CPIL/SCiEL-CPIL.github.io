"""Combined program: algebraic structure + focused grid-search refinement."""
from fractions import Fraction
from math import isqrt, gcd
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _is_squarefree(n):
    if n <= 1:
        return True
    n = abs(n)
    if n % 4 == 0:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n > 1:
        if n % 4 == 0:
            return False
        r = isqrt(n)
        if r * r == n:
            return False
    return True

# --- Fast integer-geometry helpers (for D=1 integer points) ---

def _dist2_int(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _is_int_dist(p1, p2):
    return _is_square(_dist2_int(p1, p2))

def _collinear_int(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _concyclic_int(p1, p2, p3, p4):
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    a4, b4 = p4
    if _collinear_int(p1, p2, p3):
        return False
    rhs1 = -(a2 * a2 - a1 * a1 + b2 * b2 - b1 * b1)
    rhs2 = -(a3 * a3 - a1 * a1 + b3 * b3 - b1 * b1)
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - b1
    det = x1 * y2 - x2 * y1
    if det == 0:
        return False
    num_D = rhs1 * y2 - rhs2 * y1
    num_E = x1 * rhs2 - x2 * rhs1
    num_F = -((a1 * a1 + b1 * b1) * det + num_D * a1 + num_E * b1)
    check = (a4 * a4 + b4 * b4) * det + num_D * a4 + num_E * b4 + num_F
    return check == 0

# --- General (Fraction-based) helpers for D != 1 ---

def _dist_sq_frac(p1, p2, D):
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _is_rational_dist_frac(p1, p2, D):
    d2 = _dist_sq_frac(p1, p2, D)
    return _is_square(d2.numerator) and _is_square(d2.denominator)

def _collinear_frac(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _concyclic_frac(p1, p2, p3, p4, D):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    A = x1 * x1 + D * y1 * y1
    B = x2 * x2 + D * y2 * y2
    C = x3 * x3 + D * y3 * y3
    E = x4 * x4 + D * y4 * y4

    def det3(xa, ya, xb, yb, xc, yc):
        return xa * (yb - yc) - ya * (xb - xc) + (xb * yc - xc * yb)

    det = (A * det3(x2, y2, x3, y3, x4, y4)
           - B * det3(x1, y1, x3, y3, x4, y4)
           + C * det3(x1, y1, x2, y2, x4, y4)
           - E * det3(x1, y1, x2, y2, x3, y3))
    return det == 0

def _validate_int(points):
    """Validate a set of integer points (D=1)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_int_dist(points[i], points[j]):
                return False
    for i, j, k in combinations(range(n), 3):
        if _collinear_int(points[i], points[j], points[k]):
            return False
    for i, j, k, l in combinations(range(n), 4):
        if _concyclic_int(points[i], points[j], points[k], points[l]):
            return False
    return True

def _validate_frac(points, D):
    """Validate a set of Fraction points with given D."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_rational_dist_frac(points[i], points[j], D):
                return False
    for i, j, k in combinations(range(n), 3):
        if _collinear_frac(points[i], points[j], points[k]):
            return False
    for i, j, k, l in combinations(range(n), 4):
        if _concyclic_frac(points[i], points[j], points[k], points[l], D):
            return False
    return True

def _hilbert_parameterization(D, max_t=15, max_r=10):
    """Generate points on circles in Q(sqrt(D)) rationally distant from origin."""
    points = []
    for r in range(1, max_r + 1):
        for t_num in range(-max_t, max_t + 1):
            for t_den in range(1, max_t + 1):
                if gcd(t_num, t_den) != 1:
                    continue
                t = Fraction(t_num, t_den)
                denom = 1 + D * t * t
                if denom == 0:
                    continue
                x = r * (1 - D * t * t) / denom
                y = r * 2 * t / denom
                if abs(x.numerator) > 10**40 or abs(x.denominator) > 10**40:
                    continue
                if abs(y.numerator) > 10**40 or abs(y.denominator) > 10**40:
                    continue
                points.append((x, y))
    return list(set(points))

def _search_6th_int(base_points, deadline):
    """Search for a 6th integer point extending base_points (D=1)."""
    n = len(base_points)
    best = None
    # Try progressively larger search boxes
    for limit in [20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000]:
        for x in range(-limit, limit + 1):
            if time.time() > deadline:
                return best
            for y in range(-limit, limit + 1):
                if time.time() > deadline:
                    return best
                p = (x, y)
                if p in base_points:
                    continue
                # Distances to all base points
                ok = True
                for q in base_points:
                    if not _is_int_dist(p, q):
                        ok = False
                        break
                if not ok:
                    continue
                # Collinearity with any pair
                for i in range(n):
                    for j in range(i + 1, n):
                        if _collinear_int(p, base_points[i], base_points[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # Concyclicity with any triple
                for i, j, k in combinations(range(n), 3):
                    if _concyclic_int(p, base_points[i], base_points[j], base_points[k]):
                        ok = False
                        break
                if not ok:
                    continue
                # Found a valid 6th point
                return base_points + [p]
    return best

def _search_general(D, deadline, max_points=8):
    """Search for a valid set with given D using Hilbert parameterization + greedy."""
    candidates = _hilbert_parameterization(D)
    if not candidates:
        return None

    best_set = None
    # Try multiple random orderings
    import random
    rng = random.Random(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        current = []
        for p in shuffled:
            if time.time() > deadline:
                break
            # Distances
            ok = True
            for q in current:
                if p == q or not _is_rational_dist_frac(p, q, D):
                    ok = False
                    break
            if not ok:
                continue
            # Collinearity
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    if _collinear_frac(p, current[i], current[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Concyclicity
            if len(current) >= 3:
                for i, j, k in combinations(range(len(current)), 3):
                    if _concyclic_frac(p, current[i], current[j], current[k], D):
                        ok = False
                        break
            if not ok:
                continue
            current.append(p)
            if best_set is None or len(current) > len(best_set):
                best_set = current[:]
            if len(best_set) >= max_points:
                return best_set
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # Known valid 5-point set (D=1)
    base_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if not _validate_int(base_5):
        # Fallback to a simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}

    best_result = base_5
    best_count = 5

    # 1) Focused search for a 6th integer point (most promising path)
    found_6 = _search_6th_int(base_5, deadline)
    if found_6 and len(found_6) > best_count:
        best_result = found_6
        best_count = len(found_6)

    # 2) If we have 6, try to find a 7th
    if best_count == 6 and time.time() < deadline - 10:
        found_7 = _search_6th_int(best_result, deadline)
        if found_7 and len(found_7) > best_count:
            best_result = found_7
            best_count = len(found_7)

    # 3) If time remains, try a few other D values with Hilbert parameterization
    if time.time() < deadline - 10 and best_count < 8:
        for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19]:
            if time.time() > deadline - 5:
                break
            if not _is_squarefree(D):
                continue
            found = _search_general(D, deadline, max_points=8)
            if found and len(found) > best_count:
                best_count = len(found)
                # Store as Fraction points with this D
                best_result = found
                best_D = D
                break
        else:
            best_D = 1
    else:
        best_D = 1

    # Format result
    points_formatted = []
    for p in best_result:
        x, y = p
        if isinstance(x, Fraction):
            x_num, x_den = x.numerator, x.denominator
        else:
            x_num, x_den = int(x), 1
        if isinstance(y, Fraction):
            y_num, y_den = y.numerator, y.denominator
        else:
            y_num, y_den = int(y), 1
        points_formatted.append([[x_num, x_den], [y_num, y_den]])

    return {'D': best_D, 'points': points_formatted}

# EVOLVE-BLOCK-END
