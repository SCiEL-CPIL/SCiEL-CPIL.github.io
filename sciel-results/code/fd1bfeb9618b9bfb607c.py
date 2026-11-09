"""Search for integral point sets in general position."""
import time
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

_T0 = time.time()
_DEADLINE = _T0 + 145.0


def _elapsed():
    return time.time() - _T0


def _deadline_hit():
    return _elapsed() > _DEADLINE


def _is_squarefree(n):
    if n <= 0:
        return False
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True


def _is_rational_sqrt(d2):
    """d2 is a Fraction (non-negative). Return (rn, rd) such that sqrt(d2) = rn/rd,
    or None if not a perfect square rational."""
    if d2 < 0:
        return None
    num = d2.numerator
    den = d2.denominator
    if num == 0:
        return (0, 1)
    g = gcd(num, den)
    num //= g
    den //= g
    rn = isqrt(num)
    rd = isqrt(den)
    if rn * rn == num and rd * rd == den:
        return (rn, rd)
    return None


def _check_integral(points, D):
    """points: list of (x, y) Fractions representing (x, y*sqrt(D)).
    Return (bool, list_of_int_distances) or (False, None)."""
    n = len(points)
    ok = True
    dists = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + D * dy * dy
            s = _is_rational_sqrt(d2)
            if s is None:
                return False, None
            dists.append(s)
    return True, dists


def _collinear(p1, p2, p3):
    """Three points (x, y) as Fractions: are they collinear?"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)


def _four_concyclic(p1, p2, p3, p4, D):
    """Check if 4 points (x, y*sqrt(D)) are concyclic.
    Circle: x^2 + D y^2 + A x + B y + C = 0
    4x4 determinant of [x^2+Dy^2, x, y, 1] must be nonzero."""
    M = []
    for (x, y) in (p1, p2, p3, p4):
        M.append([x * x + D * y * y, x, y, Fraction(1)])
    n = 4
    A = [row[:] for row in M]
    for c in range(n):
        pivot = None
        for r in range(c, n):
            if A[r][c] != 0:
                pivot = r
                break
        if pivot is None:
            return True
        if pivot != c:
            A[c], A[pivot] = A[pivot], A[c]
        for r in range(c + 1, n):
            factor = A[r][c] / A[c][c]
            for cc in range(c, n):
                A[r][cc] -= factor * A[c][cc]
    det = Fraction(1)
    for i in range(n):
        det *= A[i][i]
    return det == 0


def _check_general_position(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True


def _build_result(points, D):
    out = []
    for (x, y) in points:
        out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    return {'D': D, 'points': out}


def _score(points, D):
    return len(points)


def _find_sixth_point(base_points):
    """
    Try to find a 6th point with integer coordinates that has integer distances
    to all 5 base points and satisfies general position.
    """
    if _deadline_hit():
        return None
    
    # Use the first two points to generate candidates
    p1, p2 = base_points[0], base_points[1]
    x1, y1 = p1[0].numerator, p1[1].numerator
    x2, y2 = p2[0].numerator, p2[1].numerator
    
    # Distance between p1 and p2
    dx12 = x2 - x1
    dy12 = y2 - y1
    d12_sq = dx12 * dx12 + dy12 * dy12
    d12 = isqrt(d12_sq)
    if d12 * d12 != d12_sq:
        return None # Should not happen for integral set

    # Iterate over possible distances r1, r2 from the new point to p1, p2
    # The new point must be within a reasonable range.
    # Max distance to any point is roughly 2 * max_coord.
    # Let's bound r1, r2 to 1000.
    max_r = 1000
    candidates = []
    
    for r1 in range(1, max_r + 1):
        for r2 in range(1, max_r + 1):
            # Check triangle inequality
            if abs(r1 - r2) > d12 or r1 + r2 < d12:
                continue
            
            # x = x1 + (d12^2 + r1^2 - r2^2) / (2 * d12)
            num_x = d12_sq + r1 * r1 - r2 * r2
            den_x = 2 * d12
            if num_x % den_x != 0:
                continue
            x = x1 + num_x // den_x
            
            # y^2 = r1^2 - (x - x1)^2
            dy_sq = r1 * r1 - (x - x1) * (x - x1)
            if dy_sq < 0:
                continue
            y_sq = isqrt(dy_sq)
            if y_sq * y_sq != dy_sq:
                continue
            
            # Two possible y values
            for sign in [1, -1]:
                y = y1 + sign * y_sq
                p = (Fraction(x), Fraction(y))
                # Check if p is already in base_points
                if p in base_points:
                    continue
                # Check integer distances to all other base points
                valid = True
                for p_other in base_points[2:]:
                    dx = x - p_other[0].numerator
                    dy = y - p_other[1].numerator
                    d_sq = dx * dx + dy * dy
                    d = isqrt(d_sq)
                    if d * d != d_sq:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check general position with base_points + p
                test_set = base_points + [p]
                if _check_general_position(test_set, 1):
                    return p
    return None


def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    # Known 5-point integral point set with D=1
    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]

    if not _check_general_position(known_5, 1):
        # Fallback to a simpler set if the known one fails (should not happen)
        known_5 = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    
    best = known_5
    best_score = len(best)

    if not _deadline_hit():
        # Try to find a 6th point
        p6 = _find_sixth_point(known_5)
        if p6 is not None:
            candidate_6 = known_5 + [p6]
            if _check_general_position(candidate_6, 1):
                best = candidate_6
                best_score = 6

    # If we still have time and didn't find 6, try a broader search or random?
    # For now, return best found.
    
    return _build_result(best, 1)
