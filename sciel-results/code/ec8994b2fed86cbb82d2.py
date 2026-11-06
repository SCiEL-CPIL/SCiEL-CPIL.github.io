"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_no_three_collinear(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
                    return False
    return True

def _check_no_four_concyclic(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    v1 = x1 * x1 + D * y1 * y1
                    v2 = x2 * x2 + D * y2 * y2
                    v3 = x3 * x3 + D * y3 * y3
                    v4 = x4 * x4 + D * y4 * y4
                    # det of [[v1,x1,y1,1],[v2,x2,y2,1],[v3,x3,y3,1],[v4,x4,y4,1]]
                    t1 = v1 * (x2 * (y3 - y4) - x3 * (y2 - y4) + x4 * (y2 - y3))
                    t2 = -x1 * (v2 * (y3 - y4) - v3 * (y2 - y4) + v4 * (y2 - y3))
                    t3 = y1 * (v2 * (x3 - x4) - v3 * (x2 - x4) + v4 * (x2 - x3))
                    t4 = -(v2 * (x3 * y4 - y3 * x4) - v3 * (x2 * y4 - y2 * x4) + v4 * (x2 * y3 - y2 * x3))
                    if t1 + t2 + t3 + t4 == 0:
                        return False
    return True

def _all_dist_sq_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx * dx + D * dy * dy
            if d2 == 0:
                return False
            num = d2.numerator
            den = d2.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    return True

def _full_check(pts, D):
    if not _all_dist_sq_rational(pts, D):
        return False
    if not _check_no_three_collinear(pts, D):
        return False
    if not _check_no_four_concyclic(pts, D):
        return False
    return True

def _search_sixth_point(pts5, D, deadline):
    """Try to find a 6th rational point with rational distances to all 5,
    no three collinear, no four concyclic."""
    n = len(pts5)
    # Precompute squared distances from each existing point to the origin
    # For a new point (x, y), distance squared to point i is:
    # (x - xi)^2 + D*(y - yi)^2 = x^2 - 2*xi*x + xi^2 + D*y^2 - 2*D*yi*y + D*yi^2
    # = (x^2 + D*y^2) + (-2*xi)*x + (-2*D*yi)*y + (xi^2 + D*yi^2)
    # For this to be a rational square for all i, we need x^2 + D*y^2 to be rational
    # (which it is if x, y rational) and the whole expression to be a square.
    #
    # Strategy: enumerate candidate x values from small rationals, then solve for y.
    # For a fixed x, the condition that dist to point i is rational becomes:
    # D*y^2 - 2*D*yi*y + (x - xi)^2 + D*yi^2 = square
    # This is a quadratic in y. We need y rational AND the expression to be a perfect square.
    #
    # Simpler approach: brute force small rational x, y and check.
    # Also try points on lines through pairs of existing points.
    
    # First: try small integer and simple rational coordinates
    candidates = []
    # Integer points in a range
    for xi in range(-200, 201):
        for yi in range(-200, 201):
            candidates.append((Fraction(xi), Fraction(yi)))
    
    # Simple fractions
    for den in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for xi_num in range(-200, 201):
            for yi_num in range(-200, 201):
                candidates.append((Fraction(xi_num, den), Fraction(yi_num, den)))
    
    # Points on lines through pairs of existing points (parameterized)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts5[i]
            x2, y2 = pts5[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            # Parameter t: point = (x1 + t*dx, y1 + t*dy) for rational t
            # Try small rational t values
            for t_num in range(-20, 21):
                for t_den in [1, 2, 3, 4, 5]:
                    if t_den == 1 and t_num == 0:
                        continue  # skip existing points
                    t = Fraction(t_num, t_den)
                    cx = x1 + t * dx
                    cy = y1 + t * dy
                    candidates.append((cx, cy))
    
    # Check each candidate
    seen = set()
    for cx, cy in candidates:
        if time.time() > deadline:
            break
        # Skip if same as existing
        key = (cx, cy)
        if key in seen:
            continue
        seen.add(key)
        # Check distinctness
        if any(cx == px and cy == py for px, py in pts5):
            continue
        # Quick check: distances to all 5 points must be rational squares
        valid = True
        for px, py in pts5:
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + D * dy * dy
            if d2 == 0:
                valid = False
                break
            if not _is_square(d2.numerator) or not _is_square(d2.denominator):
                valid = False
                break
        if not valid:
            continue
        # Full check with all 6 points
        pts6 = pts5 + [(cx, cy)]
        if _full_check(pts6, D):
            return pts6
    return None

def _try_more_D(pts5_D1, deadline):
    """Try to find a 6-point configuration with D != 1."""
    # For D > 1, we need all pairwise distances rational.
    # Distance squared between (x1, y1) and (x2, y2) is (x1-x2)^2 + D*(y1-y2)^2
    # For this to be a rational square, we need specific relationships.
    # Try small squarefree D values and search for configurations.
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97, 101]:
        if time.time() > deadline:
            break
        # Search for 5-point base configurations with this D
        # Start with origin and try to build up
        base = [(Fraction(0), Fraction(0))]
        # Try to add points with rational distances
        # For point (x, y) to have rational distance to origin: x^2 + D*y^2 = square
        # Try small values
        found = []
        for xi in range(-50, 51):
            for yi in range(-50, 51):
                if xi == 0 and yi == 0:
                    continue
                d2 = Fraction(xi * xi + D * yi * yi)
                if _is_square(d2.numerator) and _is_square(d2.denominator):
                    found.append((Fraction(xi), Fraction(yi)))
        
        # Try to build a 5-point set from these
        # This is expensive, so limit attempts
        if len(found) < 5:
            continue
        
        # Try all combinations of 4 additional points (plus origin)
        # Too many, so use a heuristic: try subsets
        import itertools
        for combo in itertools.combinations(range(min(20, len(found))), 4):
            if time.time() > deadline:
                break
            pts = [base[0]] + [found[idx] for idx in combo]
            if _full_check(pts, D):
                # Try to extend to 6 points
                pts6 = _search_sixth_point(pts, D, deadline)
                if pts6 is not None:
                    return pts6, D
                return pts, D
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Known 5-point configuration with D=1
    pts5 = [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    D = 1
    
    # Verify base configuration
    if not _full_check(pts5, D):
        # Fallback: return a valid 3-point set
        pts3 = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(5))]
        return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts3]}
    
    best_pts = pts5
    best_D = D
    
    # Try to find a 6th point
    pts6 = _search_sixth_point(pts5, D, deadline)
    if pts6 is not None:
        best_pts = pts6
        best_D = D
    
    # Try other D values if time permits
    if time.time() < deadline - 30:
        result = _try_more_D(pts5, deadline)
        if result is not None:
            pts_try, D_try = result
            if len(pts_try) > len(best_pts):
                best_pts = pts_try
                best_D = D_try
    
    # Final safety check
    if not _full_check(best_pts, best_D):
        # Return the verified 5-point set
        best_pts = pts5
        best_D = D
    
    return {'D': best_D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_pts]}
# EVOLVE-BLOCK-END
