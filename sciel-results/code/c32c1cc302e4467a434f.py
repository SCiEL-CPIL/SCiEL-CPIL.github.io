import time
from fractions import Fraction
from math import gcd, isqrt
from itertools import combinations, product

def _is_squarefree(n):
    if n < 2:
        return True
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _g2(x1, y1, x2, y2, D):
    """Squared distance as Fraction between (x1,y1*sqrt(D)) and (x2,y2*sqrt(D))."""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _is_rational_dist(g2val):
    """Check if g2 is a perfect square of a rational (i.e., sqrt(g2) is rational)."""
    if g2val < 0:
        return False
    if g2val == 0:
        return True
    n = g2val.numerator
    d = g2val.denominator
    # Need n/d to be a perfect square rational
    # n = a^2 * k, d = b^2 * k for same squarefree k
    # Equivalently: n*d must be a perfect square and n/gcd(n,d) and d/gcd(n,d) must be perfect squares
    # Simpler: check if n is a perfect square times a squarefree part matching d's
    # Best: factor-free check: n/d is square iff for every prime p, v_p(n) and v_p(d) have same parity
    # Practical: check if n * d is a perfect square AND gcd condition
    # Actually: n/d is a rational square iff n and d have the same squarefree part
    # i.e., n * d is a perfect square
    nd = n * d
    r = isqrt(nd)
    return r * r == nd

def _collinear(p1, p2, p3, D):
    """Check if three points are collinear using cross product = 0."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # (x2-x1)(y3-y1) - (x3-x1)(y2-y1) == 0
    # But y values are y*sqrt(D), so cross product = (x2-x1)(y3-y1)*D - (x3-x1)(y2-y1)*D
    # = D * [(x2-x1)(y3-y1) - (x3-x1)(y2-y1)]
    # So just check the rational part
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0

def _four_concyclic(p1, p2, p3, p4, D):
    """Check if four points are concyclic using determinant = 0."""
    # Circle through 3 points, check if 4th lies on it
    # Using the general circle equation: x^2 + y^2*D + ax + by + c = 0
    # For point (xi, yi*sqrt(D)): xi^2 + yi^2*D + a*xi + b*yi*sqrt(D) + c = 0
    # This has a sqrt(D) term unless b=0... 
    # Actually let me use the determinant form
    # |x^2 + D*y^2, x, y*sqrt(D), 1| = 0 for all four points
    # This is tricky with sqrt(D). Let me use a different approach.
    # 
    # Four points (xi, yi*sqrt(D)) are concyclic iff there exists a circle.
    # Circle: (x-h)^2 + (y*sqrt(D)-k)^2 = r^2
    # x^2 - 2hx + h^2 + D*y^2 - 2Dky + Dk^2 = r^2
    # x^2 + D*y^2 - 2hx - 2Dky + (h^2 + Dk^2 - r^2) = 0
    # So: x^2 + D*y^2 + ax + by + c = 0 where a=-2h, b=-2Dk, c=h^2+Dk^2-r^2
    # All coefficients a, b, c are rational (since h,k,r^2 rational for rational points)
    # 
    # For 4 points, we have 4 equations in 3 unknowns (a,b,c), so the 4x4 determinant must be 0
    # |x_i^2 + D*y_i^2, x_i, y_i, 1| for i=1..4, determinant = 0
    
    det = 0
    # Compute 4x4 determinant by expansion
    pts = [p1, p2, p3, p4]
    M = []
    for (x, y) in pts:
        M.append([x*x + D*y*y, x, y, 1])
    # 4x4 determinant
    def det4(m):
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
               - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
               + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
               - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    return det4(M) == 0

def _valid_set(points, D):
    """Check all constraints for a set of points."""
    n = len(points)
    if n < 3:
        return False
    # Check all pairwise distances are rational
    for i in range(n):
        for j in range(i+1, n):
            g2 = _g2(points[i][0], points[i][1], points[j][0], points[j][1], D)
            if not _is_rational_dist(g2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(points[i], points[j], points[k], D):
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _search_D1():
    """Search for integral point sets with D=1 (integer coordinates)."""
    deadline = time.time() + 100
    best = None
    
    # Start with known 5-point set
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to add more points by local search
    # Strategy: try to find a 6th point that is at rational distance from all 5
    # and maintains general position
    
    # Search in a bounded region
    for x in range(-200, 201):
        if time.time() > deadline:
            break
        for y in range(-200, 201):
            if time.time() > deadline:
                break
            new_pt = (x, y)
            # Check distinct
            if new_pt in base:
                continue
            # Check rational distances to all existing points
            ok = True
            for (bx, by) in base:
                d2 = (x-bx)**2 + (y-by)**2
                r = isqrt(d2)
                if r * r != d2:
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear
            ok = True
            for i in range(len(base)):
                for j in range(i+1, len(base)):
                    if _collinear(new_pt, base[i], base[j], 1):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check no four concyclic
            ok = True
            for i in range(len(base)):
                for j in range(i+1, len(base)):
                    for k in range(j+1, len(base)):
                        if _four_concyclic(new_pt, base[i], base[j], base[k], 1):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Found a valid 6-point set!
            new_set = base + [new_pt]
            return new_set
    
    # If no 6th point found, try different 5-point bases
    # Try small lattice searches for 4-5 point sets and extend
    # For now, return the base 5-point set
    return base

def _search_general(D, max_coord=50):
    """Search for integral point sets with given D in a bounded region."""
    deadline = time.time() + 80
    best = None
    best_size = 0
    
    # Generate candidate points with small coordinates
    candidates = []
    for x in range(-max_coord, max_coord+1):
        for y in range(-max_coord, max_coord+1):
            candidates.append((x, y))
    
    # Greedy: try to build a large set
    # Start with pairs that have rational distance
    # This is expensive, so limit it
    
    # Try a simple approach: start from origin, add points one by one
    current = [(0, 0)]
    
    for pt in candidates:
        if time.time() > deadline:
            break
        if pt == (0, 0):
            continue
        # Check distance to origin
        d2 = pt[0]**2 + D * pt[1]**2
        if not _is_rational_dist(Fraction(d2, 1)):
            continue
        # Check no three collinear (trivially ok with 2 points)
        current.append(pt)
        if len(current) > best_size:
            best_size = len(current)
            best = list(current)
        if len(current) >= 6:
            break
    
    # Try to extend further
    if best and len(best) < 8:
        for pt in candidates:
            if time.time() > deadline:
                break
            if pt in best:
                continue
            ok = True
            for existing in best:
                d2 = (pt[0]-existing[0])**2 + D*(pt[1]-existing[1])**2
                if not _is_rational_dist(Fraction(d2, 1)):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinearity
            ok = True
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    if _collinear(pt, best[i], best[j], D):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check concyclicity
            ok = True
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    for k in range(j+1, len(best)):
                        if _four_concyclic(pt, best[i], best[j], best[k], D):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            best.append(pt)
            if len(best) > best_size:
                best_size = len(best)
            if len(best) >= 8:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, try D=1 search
    result_D1 = _search_D1()
    
    if time.time() > deadline:
        # Return what we have
        pts = result_D1
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Try a few other D values
    best_result = None
    best_size = len(result_D1)
    if best_size >= 3:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result_D1]}
    
    for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]:
        if time.time() > deadline - 5:
            break
        if not _is_squarefree(D):
            continue
        result = _search_general(D, max_coord=30)
        if result and len(result) > best_size:
            best_size = len(result)
            best_result = {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in result]}
    
    if best_result is None:
        # Fallback: return the known 5-point set
        pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    return best_result
