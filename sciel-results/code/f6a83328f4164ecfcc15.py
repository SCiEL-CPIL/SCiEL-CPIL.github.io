"""Search for integral point sets in general position."""
import time
import random
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

# Deadline guard
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
    """Check if a Fraction d2 is a perfect square rational.
    Return (rn, rd) such that sqrt(d2) = rn/rd, or None."""
    num = d2.numerator
    den = d2.denominator
    g = gcd(num, den)
    num //= g
    den //= g
    rn = isqrt(num)
    rd = isqrt(den)
    if rn * rn == num and rd * rd == den:
        return (rn, rd)
    return None


def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)


def _four_concyclic(p1, p2, p3, p4, D):
    """Check if 4 points (x, y*sqrt(D)) are concyclic using the determinant."""
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


def _valid_distance(x, y, px, py, D):
    """Check if distance squared between (x,y) and (px,py) is a perfect square rational."""
    dx = Fraction(x) - px
    dy = Fraction(y) - py
    d2 = dx * dx + D * dy * dy
    if d2 < 0:
        return False
    return _is_rational_sqrt(d2) is not None


def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    best = None
    best_score = -1

    # Known small solutions to seed from
    # 5 points with D=1 (integer points, integer distances)
    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]

    # Validate the known 5
    ok = True
    for i in range(5):
        for j in range(i + 1, 5):
            if not _valid_distance(known_5[i][0], known_5[i][1], known_5[j][0], known_5[j][1], 1):
                ok = False
                break
        if not ok:
            break
    if ok and _check_general_position(known_5, 1):
        best = known_5
        best_score = 5

    if not _deadline_hit():
        # Greedy growth from the known 5-point set
        current = list(known_5)
        max_coord = 300
        improved = True
        while improved and not _deadline_hit():
            improved = False
            best_candidate = None
            for x in range(-max_coord, max_coord + 1):
                if _deadline_hit():
                    break
                for y in range(-max_coord, max_coord + 1):
                    if _deadline_hit():
                        break
                    new_pt = (Fraction(x), Fraction(y))
                    if new_pt in current:
                        continue
                    valid = True
                    for (px, py) in current:
                        if not _valid_distance(x, y, px, py, 1):
                            valid = False
                            break
                    if not valid:
                        continue
                    n = len(current)
                    collinear_ok = True
                    for i in range(n):
                        for j in range(i + 1, n):
                            if _collinear(current[i], current[j], new_pt):
                                collinear_ok = False
                                break
                        if not collinear_ok:
                            break
                    if not collinear_ok:
                        continue
                    concyclic_ok = True
                    for i in range(n):
                        for j in range(i + 1, n):
                            for k in range(j + 1, n):
                                if _four_concyclic(current[i], current[j], current[k], new_pt, 1):
                                    concyclic_ok = False
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        continue
                    best_candidate = new_pt
                    improved = True
                    break
                if improved:
                    break
            if improved:
                current.append(best_candidate)
                if len(current) > best_score:
                    best_score = len(current)
                    best = list(current)

    if not _deadline_hit():
        # Try randomized hill-climbing search with D=1, larger coordinates
        random.seed(42)
        max_coord = 800
        for _attempt in range(80):
            if _deadline_hit():
                break
            pts = []
            tries = 0
            while len(pts) < 3 and tries < 2000:
                tries += 1
                x = random.randint(-max_coord, max_coord)
                y = random.randint(-max_coord, max_coord)
                p = (Fraction(x), Fraction(y))
                if p in pts:
                    continue
                valid = True
                for (px, py) in pts:
                    if not _valid_distance(x, y, px, py, 1):
                        valid = False
                        break
                if not valid:
                    continue
                collinear_ok = True
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        if _collinear(pts[i], pts[j], p):
                            collinear_ok = False
                            break
                    if not collinear_ok:
                        break
                if not collinear_ok:
                    continue
                pts.append(p)
            if len(pts) < 3:
                continue
            current = list(pts)
            for _grow in range(15):
                if _deadline_hit():
                    break
                added = False
                for _tr in range(300):
                    if _deadline_hit():
                        break
                    x = random.randint(-max_coord, max_coord)
                    y = random.randint(-max_coord, max_coord)
                    p = (Fraction(x), Fraction(y))
                    if p in current:
                        continue
                    valid = True
                    for (px, py) in current:
                        if not _valid_distance(x, y, px, py, 1):
                            valid = False
                            break
                    if not valid:
                        continue
                    collinear_ok = True
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            if _collinear(current[i], current[j], p):
                                collinear_ok = False
                                break
                        if not collinear_ok:
                            break
                    if not collinear_ok:
                        continue
                    concyclic_ok = True
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            for k in range(j + 1, len(current)):
                                if _four_concyclic(current[i], current[j], current[k], p, 1):
                                    concyclic_ok = False
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        continue
                    current.append(p)
                    added = True
                    break
                if not added:
                    break
            if len(current) > best_score:
                best_score = len(current)
                best = list(current)

    if best is not None and best_score >= 5:
        return _build_result(best, 1)

    return _build_result(known_5, 1)
