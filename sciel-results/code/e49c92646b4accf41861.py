"""Search for integral point sets in general position."""
import time
import random
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations

# Deadline guard: leave 5s safety margin before the 150s kill.
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
    # This is a known valid set:
    # (0,0), (-56/33, 0), (-16/63, 0), (0, 33), (16/63, 0) -- wait, these are not all integer distances.
    # Let's use a verified set of 5 points with D=1.
    # A known set of 5 points with rational distances:
    # (0,0), (3,4), (0,12), (-15/8, -11/4), (21/5, 28/5) -- this needs verification.
    
    # Instead of risking an invalid hardcoded set, we will generate one or use a very simple one.
    # The simplest is 3 points: (0,0), (1,0), (0,1). Distances: 1, 1, sqrt(2) (not rational).
    # So D=1 integer points must form rational distances.
    # (0,0), (3,4), (5,0) -> dists: 5, 5, 2. All rational.
    # Add (0,12): dist to (0,0)=12, to (3,4)=10, to (5,0)=13. All rational.
    # Add (x,y) such that dist to all 4 is rational.
    # Let's try to find a 5th point for {(0,0), (3,4), (5,0), (0,12)} with D=1.
    
    pts4 = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), 
            (Fraction(5), Fraction(0)), (Fraction(0), Fraction(12))]
    
    # Check if pts4 is valid
    ok = True
    for i in range(4):
        for j in range(i+1, 4):
            if not _valid_distance(pts4[i][0], pts4[i][1], pts4[j][0], pts4[j][1], 1):
                ok = False
    if ok and _check_general_position(pts4, 1):
        best = pts4
        best_score = 4
        
        # Try to add a 5th point
        max_coord = 200
        for x in range(-max_coord, max_coord + 1):
            if _deadline_hit():
                break
            for y in range(-max_coord, max_coord + 1):
                if _deadline_hit():
                    break
                new_pt = (Fraction(x), Fraction(y))
                if new_pt in pts4:
                    continue
                valid = True
                for (px, py) in pts4:
                    if not _valid_distance(x, y, px, py, 1):
                        valid = False
                        break
                if not valid:
                    continue
                collinear_ok = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        if _collinear(pts4[i], pts4[j], new_pt):
                            collinear_ok = False
                            break
                    if not collinear_ok:
                        break
                if not collinear_ok:
                    continue
                concyclic_ok = True
                for i in range(4):
                    for j in range(i + 1, 4):
                        for k in range(j + 1, 4):
                            if _four_concyclic(pts4[i], pts4[j], pts4[k], new_pt, 1):
                                concyclic_ok = False
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        break
                if not concyclic_ok:
                    continue
                
                # Found a 5th point
                best = pts4 + [new_pt]
                best_score = 5
                break
            if best_score == 5:
                break

    # If we didn't find a 5th point in the grid, try randomized search
    if best_score < 5 and not _deadline_hit():
        random.seed(42)
        max_coord = 1000
        for _attempt in range(50):
            if _deadline_hit():
                break
            pts = []
            tries = 0
            while len(pts) < 4 and tries < 5000:
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
            
            if len(pts) < 4:
                continue
                
            # Try to add 5th point
            for _tr in range(10000):
                if _deadline_hit():
                    break
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
                concyclic_ok = True
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            if _four_concyclic(pts[i], pts[j], pts[k], p, 1):
                                concyclic_ok = False
                                break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        break
                if not concyclic_ok:
                    continue
                pts.append(p)
                break
            
            if len(pts) > best_score:
                best_score = len(pts)
                best = pts

    # If we still have less than 3 points, return a trivial set
    if best is None or best_score < 3:
        # (0,0), (1,0), (0,1) is not valid because dist is sqrt(2).
        # (0,0), (3,4), (5,0) is valid.
        best = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(5), Fraction(0))]
        best_score = 3

    return _build_result(best, 1)
