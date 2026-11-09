import time
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations
import random

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

def _rational_sqrt(n):
    if n < 0:
        return None
    r = isqrt(n)
    if r * r == n:
        return (r, 1)
    return None

def _is_rational(x):
    return x.denominator == 1

def _check_integral(points, D):
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
            if d2 < 0:
                ok = False
                break
            num = d2.numerator
            den = d2.denominator
            g = gcd(num, den)
            num //= g
            den //= g
            rn = isqrt(num)
            rd = isqrt(den)
            if rn * rn != num or rd * rd != den:
                ok = False
                break
            dists.append((rn, rd))
    if not ok:
        return False, None
    return True, dists

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _four_concyclic(p1, p2, p3, p4, D):
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
        out.append([ [x.numerator, x.denominator], [y.numerator, y.denominator] ])
    return {'D': D, 'points': out}

def _score(points, D):
    return len(points)

def _solve_for_point(P0, P1, P2, r0_sq, r1_sq, r2_sq, D):
    """
    Solves for (x, y) such that dist(P, P0)^2 = r0_sq, etc.
    Equations:
    (x-x0)^2 + D(y-y0)^2 = r0_sq
    (x-x1)^2 + D(y-y1)^2 = r1_sq
    (x-x2)^2 + D(y-y2)^2 = r2_sq
    
    Subtracting 0 from 1:
    -2x(x1-x0) - 2D y(y1-y0) + (x1^2 - x0^2) + D(y1^2 - y0^2) = r1_sq - r0_sq
    2x(x1-x0) + 2D y(y1-y0) = (x1^2 - x0^2) + D(y1^2 - y0^2) - (r1_sq - r0_sq)
    
    Let A1 = 2(x1-x0), B1 = 2D(y1-y0), C1 = (x1^2 - x0^2) + D(y1^2 - y0^2) - (r1_sq - r0_sq)
    Let A2 = 2(x2-x0), B2 = 2D(y2-y0), C2 = (x2^2 - x0^2) + D(y2^2 - y0^2) - (r2_sq - r0_sq)
    
    A1 x + B1 y = C1
    A2 x + B2 y = C2
    
    Determinant Det = A1 B2 - A2 B1
    x = (C1 B2 - C2 B1) / Det
    y = (A1 C2 - A2 C1) / Det
    """
    x0, y0 = P0
    x1, y1 = P1
    x2, y2 = P2
    
    # Convert to Fractions for exact arithmetic
    x0, y0 = Fraction(x0), Fraction(y0)
    x1, y1 = Fraction(x1), Fraction(y1)
    x2, y2 = Fraction(x2), Fraction(y2)
    r0_sq = Fraction(r0_sq)
    r1_sq = Fraction(r1_sq)
    r2_sq = Fraction(r2_sq)
    D = Fraction(D)
    
    A1 = 2 * (x1 - x0)
    B1 = 2 * D * (y1 - y0)
    C1 = (x1*x1 - x0*x0) + D*(y1*y1 - y0*y0) - (r1_sq - r0_sq)
    
    A2 = 2 * (x2 - x0)
    B2 = 2 * D * (y2 - y0)
    C2 = (x2*x2 - x0*x0) + D*(y2*y2 - y0*y0) - (r2_sq - r0_sq)
    
    Det = A1 * B2 - A2 * B1
    if Det == 0:
        return None
        
    x = (C1 * B2 - C2 * B1) / Det
    y = (A1 * C2 - A2 * C1) / Det
    
    return (x, y)

def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    best = None
    best_score = -1

    # Known 5-point set with D=1
    # These points have integer coordinates and integer distances.
    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]
    
    ok, _ = _check_integral(known_5, 1)
    if ok and _check_general_position(known_5, 1):
        best = known_5
        best_score = 5

    current = list(known_5)
    D = 1
    
    improved = True
    max_r_sq = 5000  # Max squared distance to search
    
    while improved and not _deadline_hit():
        improved = False
        found = False
        
        # Select 3 non-collinear points to define the system
        # Use the first 3 for simplicity, ensure they are non-collinear
        P0, P1, P2 = current[0], current[1], current[2]
        if _collinear(P0, P1, P2):
            # Find a non-collinear triple
            found_triple = False
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    for k in range(j+1, len(current)):
                        if not _collinear(current[i], current[j], current[k]):
                            P0, P1, P2 = current[i], current[j], current[k]
                            found_triple = True
                            break
                    if found_triple: break
                if found_triple: break
            if not found_triple:
                break # All collinear? Shouldn't happen with GP check
        
        # Iterate over possible squared distances
        # We can restrict search space by noting that distances must be rational.
        # For D=1, if we assume integer coordinates, distances are integers.
        # But we allow rational coordinates.
        # However, the linear system yields rational x,y for any rational r_sq.
        # To keep numbers small, we iterate over integer r_sq first.
        
        for r0_sq in range(1, max_r_sq + 1):
            if _deadline_hit(): break
            for r1_sq in range(1, max_r_sq + 1):
                if _deadline_hit(): break
                # Optimization: Triangle inequality on squared distances is tricky, 
                # but we can check basic bounds later.
                for r2_sq in range(1, max_r_sq + 1):
                    if _deadline_hit(): break
                    
                    pt = _solve_for_point(P0, P1, P2, r0_sq, r1_sq, r2_sq, D)
                    if pt is None:
                        continue
                    
                    x, y = pt
                    # Check if x, y are "simple" rationals? 
                    # The problem allows |num|, |den| <= 10^40.
                    # Fractions from integer inputs will have manageable denominators.
                    
                    # Check if this point is distinct from existing
                    if any(x == px and y == py for (px, py) in current):
                        continue
                        
                    # Check integral distances to ALL existing points
                    valid = True
                    for (px, py) in current:
                        dx = x - px
                        dy = y - py
                        d2 = dx * dx + D * dy * dy
                        if d2 < 0:
                            valid = False
                            break
                        num = d2.numerator
                        den = d2.denominator
                        g = gcd(num, den)
                        num //= g
                        den //= g
                        rn = isqrt(num)
                        rd = isqrt(den)
                        if rn * rn != num or rd * rd != den:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check no three collinear
                    collinear_ok = True
                    n = len(current)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if _collinear(current[i], current[j], pt):
                                collinear_ok = False
                                break
                        if not collinear_ok:
                            break
                    if not collinear_ok:
                        continue
                    
                    # Check no four concyclic
                    concyclic_ok = True
                    for i in range(n):
                        for j in range(i + 1, n):
                            for k in range(j + 1, n):
                                if _four_concyclic(current[i], current[j], current[k], pt, D):
                                    concyclic_ok = False
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        continue
                    
                    # Valid
                    current.append(pt)
                    improved = True
                    found = True
                    break
                if found:
                    break
            if found:
                break
        
        if improved:
            if len(current) > best_score:
                best_score = len(current)
                best = list(current)
                # Increase search range slightly?
                max_r_sq += 500
        else:
            break

    if best is not None and best_score >= 5:
        return _build_result(best, D)

    # Fallback
    return _build_result(known_5, 1)
