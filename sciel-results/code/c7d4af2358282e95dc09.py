"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    r = isqrt(n)
    return r * r == n

def _no_four_concyclic_fast(points, D):
    """Check no four points are concyclic using determinant test.
    Points are (x, y_coeff) where actual y = y_coeff * sqrt(D).
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1c = points[i]
                    x2, y2c = points[j]
                    x3, y3c = points[k]
                    x4, y4c = points[l]
                    
                    def minor3(i1, i2, i3):
                        a, b = points[i1]
                        c, d = points[i2]
                        e, f = points[i3]
                        va = a*a + D*b*b
                        vc = c*c + D*d*d
                        ve = e*e + D*f*f
                        bracket = a*(d*ve - vc*f) - b*(c*ve - vc*e) + va*(c*f - d*e)
                        return bracket
                    
                    m1 = minor3(i, j, k)
                    m2 = minor3(i, j, l)
                    m3 = minor3(i, k, l)
                    m4 = minor3(j, k, l)
                    
                    total = m1 - m2 + m3 - m4
                    if total == 0:
                        return False
    return True

def _pyth_candidates(x, max_y):
    """Generate all y such that x^2 + y^2 is a perfect square, |y| <= max_y."""
    candidates = set()
    x2 = x * x
    if x == 0:
        for y in range(-max_y, max_y + 1):
            candidates.add(y)
        return candidates
    
    # (k-y)(k+y) = x^2
    a = 1
    while a * a <= x2:
        if x2 % a == 0:
            b = x2 // a
            if (a + b) % 2 == 0:
                k_val = (a + b) // 2
                y_val = (b - a) // 2
                if abs(y_val) <= max_y:
                    candidates.add(y_val)
                    if y_val != 0:
                        candidates.add(-y_val)
        a += 1
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Known 5-point integral point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    best = base_frac
    best_count = 5
    
    # Strategy: For each candidate (x, y), check if distance to ALL base points is integer.
    # Use Pythagorean triple enumeration to generate candidates efficiently.
    
    search_range = 2000
    
    found_six = False
    for x in range(0, search_range + 1):
        if time.time() > deadline:
            break
        
        candidates_y = _pyth_candidates(x, search_range)
        
        for y in sorted(candidates_y):
            if time.time() > deadline:
                break
            
            # Check not collinear with any pair of base points
            collinear = False
            for i in range(5):
                for j in range(i+1, 5):
                    x1, y1 = base_frac[i]
                    x2_val, y2_val = base_frac[j]
                    cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                    if cp == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check all pairwise distances are integers
            all_ok = True
            for pt in base_frac:
                dx = x - pt[0]
                dy = y - pt[1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2.numerator * d2.denominator):
                    all_ok = False
                    break
            if not all_ok:
                continue
            
            # Check no four concyclic
            candidate = base_frac + [(Fraction(x), Fraction(y))]
            if not _no_four_concyclic_fast(candidate, D):
                continue
            
            # Found a 6-point set!
            best = candidate
            best_count = 6
            found_six = True
            break
        if found_six:
            break
    
    # If we found 6 points, try to find a 7th
    if found_six:
        found_seven = False
        search_range2 = 3000
        for x in range(0, search_range2 + 1):
            if time.time() > deadline:
                break
            
            candidates_y = _pyth_candidates(x, search_range2)
            
            for y in sorted(candidates_y):
                if time.time() > deadline:
                    break
                
                # Check not collinear with any pair
                collinear = False
                for i in range(6):
                    for j in range(i+1, 6):
                        x1, y1 = best[i]
                        x2_val, y2_val = best[j]
                        cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                        if cp == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check all pairwise distances are integers
                all_ok = True
                for pt in best:
                    dx = x - pt[0]
                    dy = y - pt[1]
                    d2 = dx * dx + dy * dy
                    if not _is_square(d2.numerator * d2.denominator):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                
                # Check no four concyclic
                candidate = best + [(Fraction(x), Fraction(y))]
                if not _no_four_concyclic_fast(candidate, D):
                    continue
                
                best = candidate
                best_count = 7
                found_seven = True
                break
            if found_seven:
                break
    
    # If we still only have 5 points, try a wider search with a different approach
    if best_count == 5:
        search_range3 = 500
        for x in range(0, search_range3 + 1):
            if time.time() > deadline:
                break
            for y in range(0, search_range3 + 1):
                if time.time() > deadline:
                    break
                
                d2_00 = x*x + y*y
                if not _is_square(d2_00):
                    continue
                
                dx1 = x + 56
                dy1 = y - 33
                d2_1 = dx1*dx1 + dy1*dy1
                if not _is_square(d2_1):
                    continue
                
                dx2 = x + 16
                dy2 = y - 63
                d2_2 = dx2*dx2 + dy2*dy2
                if not _is_square(d2_2):
                    continue
                
                dx3 = x
                dy3 = y - 33
                d2_3 = dx3*dx3 + dy3*dy3
                if not _is_square(d2_3):
                    continue
                
                dx4 = x - 16
                dy4 = y - 63
                d2_4 = dx4*dx4 + dy4*dy4
                if not _is_square(d2_4):
                    continue
                
                collinear = False
                for i in range(5):
                    for j in range(i+1, 5):
                        x1, y1 = base_frac[i]
                        x2_val, y2_val = base_frac[j]
                        cp = (x2_val - x1) * (y - y1) - (y2_val - y1) * (x - x1)
                        if cp == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                candidate = base_frac + [(Fraction(x), Fraction(y))]
                if not _no_four_concyclic_fast(candidate, D):
                    continue
                
                best = candidate
                best_count = 6
                break
            if best_count == 6:
                break
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
