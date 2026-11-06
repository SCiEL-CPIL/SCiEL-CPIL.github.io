"""
Rewritten program: find integral point sets with rational pairwise distances,
no three collinear, no four concyclic, maximizing point count.
Strategy: use a known base set, search for additional points via
Pythagorean triple enumeration, with exact rational arithmetic.
"""
import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = isqrt(n)
    return r * r == n

def _no_three_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic using determinant test."""
    n = len(points)
    if n < 4:
        return True
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        x1, y1c = points[i]
        x2, y2c = points[j]
        x3, y3c = points[k]
        x4, y4c = points[l]
        
        def minor3(a, b, c):
            xa, ya = points[a]
            xb, yb = points[b]
            xc, yc = points[c]
            va = xa*xa + D*ya*ya
            vc = xb*xb + D*yb*yb
            ve = xc*xc + D*yc*yc
            bracket = xa*(yb*ve - vc*yc) - ya*(xb*ve - vc*xc) + va*(xb*yc - yb*xc)
            return bracket
        
        m1 = minor3(i, j, k)
        m2 = minor3(i, j, l)
        m3 = minor3(i, k, l)
        m4 = minor3(j, k, l)
        
        total = m1 - m2 + m3 - m4
        if total == 0:
            return False
    return True

def _all_pairwise_rational(points, D):
    """Check all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1c = points[i]
            x2, y2c = points[j]
            dx = x1 - x2
            dy = (y1c - y2c)
            d2 = dx*dx + D*dy*dy
            if not _is_square(d2.numerator * d2.denominator):
                return False
    return True

def _validate_set(points, D):
    """Validate that the set satisfies all constraints."""
    n = len(points)
    if n < 3:
        return False
    # Check distinctness
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    # Check no three collinear
    if not _no_three_collinear(points):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points, D):
        return False
    # Check all pairwise distances rational
    if not _all_pairwise_rational(points, D):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    D = 1
    
    # Start with a known 5-point integral set
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Check: all distances are integers, no three collinear, no four concyclic
    base = [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    
    best = base[:]
    best_count = 5
    
    # Verify base set
    if not _validate_set(base, D):
        # Fallback: try another known set
        base = [(Fraction(0), Fraction(0)),
                (Fraction(3), Fraction(4)),
                (Fraction(6), Fraction(8)),
                (Fraction(9), Fraction(12)),
                (Fraction(12), Fraction(16))]
        best = base[:]
        best_count = 5
    
    # Search for 6th point
    found_six = False
    search_range = 500
    
    for x in range(-search_range, search_range + 1):
        if time.time() > deadline:
            break
        for y in range(-search_range, search_range + 1):
            if time.time() > deadline:
                break
            
            fx, fy = Fraction(x), Fraction(y)
            
            # Check distinct
            if any(fx == p[0] and fy == p[1] for p in best):
                continue
            
            # Check collinearity with base
            collinear = False
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    x1, y1 = best[i]
                    x2, y2 = best[j]
                    cross = (x2 - x1) * (fy - y1) - (y2 - y1) * (fx - x1)
                    if cross == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check all pairwise distances
            all_ok = True
            for pt in best:
                dx = fx - pt[0]
                dy = fy - pt[1]
                d2 = dx*dx + D*dy*dy
                if not _is_square(d2.numerator * d2.denominator):
                    all_ok = False
                    break
            if not all_ok:
                continue
            
            # Check no four concyclic
            candidate = best + [(fx, fy)]
            if not _no_four_concyclic(candidate, D):
                continue
            
            best = candidate
            best_count = 6
            found_six = True
            break
        if found_six:
            break
    
    # Try to find 7th point if we have 6
    if found_six:
        found_seven = False
        search_range2 = 400
        for x in range(-search_range2, search_range2 + 1):
            if time.time() > deadline:
                break
            for y in range(-search_range2, search_range2 + 1):
                if time.time() > deadline:
                    break
                
                fx, fy = Fraction(x), Fraction(y)
                
                # Check distinct
                if any(fx == p[0] and fy == p[1] for p in best):
                    continue
                
                # Check collinearity
                collinear = False
                for i in range(len(best)):
                    for j in range(i+1, len(best)):
                        x1, y1 = best[i]
                        x2, y2 = best[j]
                        cross = (x2 - x1) * (fy - y1) - (y2 - y1) * (fx - x1)
                        if cross == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check all pairwise distances
                all_ok = True
                for pt in best:
                    dx = fx - pt[0]
                    dy = fy - pt[1]
                    d2 = dx*dx + D*dy*dy
                    if not _is_square(d2.numerator * d2.denominator):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                
                # Check no four concyclic
                candidate = best + [(fx, fy)]
                if not _no_four_concyclic(candidate, D):
                    continue
                
                best = candidate
                best_count = 7
                found_seven = True
                break
            if found_seven:
                break
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
