"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _is_squarefree(n):
    if n == 1:
        return True
    if n % 2 == 0:
        if n % 4 == 0:
            return False
        n //= 2
    i = 3
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 2
    return True

def _gcd_num(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a if a != 0 else 1

def _check_pairwise_rational(p1, p2, D):
    """Check if distance between p1 and p2 is rational."""
    x1n, x1d = p1[0]
    y1n, y1d = p1[1]
    x2n, x2d = p2[0]
    y2n, y2d = p2[1]
    
    # dx = x1 - x2, dy = y1 - y2
    dx_num = x1n * x2d - x2n * x1d
    dx_den = x1d * x2d
    
    dy_num = y1n * y2d - y2n * y1d
    dy_den = y1d * y2d
    
    # dist^2 = dx^2 + D * dy^2
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    
    if den == 0:
        return False
    if num < 0:
        return False
    
    g = _gcd_num(num, den)
    num_reduced = num // g
    den_reduced = den // g
    
    sqrt_num = int(math.isqrt(num_reduced))
    sqrt_den = int(math.isqrt(den_reduced))
    
    if sqrt_num * sqrt_num != num_reduced:
        return False
    if sqrt_den * sqrt_den != den_reduced:
        return False
    if sqrt_den == 0:
        return False
        
    return True

def _no_three_collinear(points, D):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi_n, xi_d = points[i][0]
                yi_n, yi_d = points[i][1]
                xj_n, xj_d = points[j][0]
                yj_n, yj_d = points[j][1]
                xk_n, xk_d = points[k][0]
                yk_n, yk_d = points[k][1]
                
                dx_ji_num = xj_n * xi_d - xi_n * xj_d
                dx_ji_den = xj_d * xi_d
                dy_ki_num = yk_n * yi_d - yi_n * yk_d
                dy_ki_den = yk_d * yi_d
                dx_ki_num = xk_n * xi_d - xi_n * xk_d
                dx_ki_den = xk_d * xi_d
                dy_ji_num = yj_n * yi_d - yi_n * yj_d
                dy_ji_den = yj_d * yi_d
                
                cross_num2 = dx_ji_num * dy_ki_num * dx_ki_den * dy_ji_den - dx_ki_num * dy_ji_num * dx_ji_den * dy_ki_den
                cross_den2 = dx_ji_den * dy_ki_den * dx_ki_den * dy_ji_den
                
                if cross_den2 == 0:
                    continue
                if cross_num2 != 0:
                    continue
                return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    xi_n, xi_d = points[i][0]
                    yi_n, yi_d = points[i][1]
                    xj_n, xj_d = points[j][0]
                    yj_n, yj_d = points[j][1]
                    xk_n, xk_d = points[k][0]
                    yk_n, yk_d = points[k][1]
                    xl_n, xl_d = points[l][0]
                    yl_n, yl_d = points[l][1]
                    
                    xi = Fraction(xi_n, xi_d)
                    yi = Fraction(yi_n, yi_d)
                    xj = Fraction(xj_n, xj_d)
                    yj = Fraction(yj_n, yj_d)
                    xk = Fraction(xk_n, xk_d)
                    yk = Fraction(yk_n, yk_d)
                    xl = Fraction(xl_n, xl_d)
                    yl = Fraction(yl_n, yl_d)
                    
                    A1 = xj - xi
                    B1 = yj - yi
                    C1 = -(xj*xj + D*yj*yj) + (xi*xi + D*yi*yi)
                    
                    A2 = xk - xi
                    B2 = yk - yi
                    C2 = -(xk*xk + D*yk*yk) + (xi*xi + D*yi*yi)
                    
                    det = A1*B2 - A2*B1
                    if det == 0:
                        continue
                    
                    a = (C1*B2 - C2*B1) / det
                    b_prime = (A1*C2 - A2*C1) / det
                    c = -(xi*xi + D*yi*yi) - a*xi - b_prime*yi
                    
                    if xl*xl + D*yl*yl + a*xl + b_prime*yl + c != 0:
                        continue
                    return False
    return True

def _validate(points, D):
    """Validate the full configuration."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    for i in range(n):
        for j in range(i+1, n):
            if not _check_pairwise_rational(points[i], points[j], D):
                return False
    if not _no_three_collinear(points, D):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _find_6_point_set(D=1, time_limit=30.0):
    """
    Search for a 6-point integral point set using a smarter approach.
    We look for points on two parallel lines y=0 and y=h.
    Key insight: use Pythagorean triples to find valid x-coordinates.
    """
    start = time.time()
    
    # Try different heights h
    for h in range(1, 30):
        if time.time() - start > time_limit:
            break
        
        # Find all valid dx values where dx^2 + h^2 is a perfect square
        valid_dx = set()
        max_k = 500
        for k in range(h + 1, max_k):
            dx2 = k * k - h * h
            dx = int(math.isqrt(dx2))
            if dx * dx == dx2 and dx > 0:
                valid_dx.add(dx)
                valid_dx.add(-dx)
        
        if len(valid_dx) < 6:
            continue
        
        # Sort for determinism
        valid_dx_sorted = sorted(valid_dx)
        
        # We need 3 points on y=0 and 3 on y=h
        # Fix bottom point at x=0 for symmetry
        # Top points must be at positions where distance from (0,0) is rational
        # So top x-coordinates must be in valid_dx
        
        # Try to find 3 top x-coordinates that also work for 2 more bottom points
        # This is a constraint satisfaction problem
        
        # Strategy: pick 3 top points, then find bottom points that work with all 3
        top_candidates = [dx for dx in valid_dx_sorted if abs(dx) <= 200]
        
        if len(top_candidates) < 3:
            continue
        
        # Try combinations of 3 top points
        for idx in range(len(top_candidates)):
            if time.time() - start > time_limit:
                break
            t1 = top_candidates[idx]
            for idx2 in range(idx + 1, len(top_candidates)):
                t2 = top_candidates[idx2]
                for idx3 in range(idx2 + 1, len(top_candidates)):
                    t3 = top_candidates[idx3]
                    
                    # Find bottom x-coordinates that work with all three top points
                    # Bottom x must satisfy: (x - t1)^2 + h^2 is square, etc.
                    # So (x - t1) in valid_dx, (x - t2) in valid_dx, (x - t3) in valid_dx
                    
                    set1 = {t1 + dx for dx in valid_dx}
                    set2 = {t2 + dx for dx in valid_dx}
                    set3 = {t3 + dx for dx in valid_dx}
                    
                    possible_x = set1 & set2 & set3
                    possible_x = [x for x in possible_x if abs(x) <= 200 and x != 0]
                    
                    if len(possible_x) < 2:
                        continue
                    
                    # Pick 2 bottom points from possible_x
                    for x2, x3 in combinations(possible_x, 2):
                        if x2 == x3:
                            continue
                        
                        pts = [
                            [[0, 1], [0, 1]],
                            [[x2, 1], [0, 1]],
                            [[x3, 1], [0, 1]],
                            [[t1, 1], [h, 1]],
                            [[t2, 1], [h, 1]],
                            [[t3, 1], [h, 1]]
                        ]
                        
                        # Quick distinctness check
                        if len(set(tuple(tuple(c) for c in p) for p in pts)) < 6:
                            continue
                        
                        if _validate(pts, D):
                            return pts
    return None

def _find_7_point_set(D=1, time_limit=20.0):
    """
    Try to extend a 6-point set to 7 points.
    """
    start = time.time()
    
    # First find a 6-point set
    pts6 = _find_6_point_set(D, time_limit=10.0)
    if not pts6:
        return None
    
    # Try to add a 7th point
    # The 7th point must be at rational distance from all 6 existing points
    # And not create collinearity or concyclicity
    
    # For points on two lines, the 7th point could be on a third line
    # Or we could try a different configuration
    
    # This is complex; let's try a simpler approach
    # Try adding a point on y=0 or y=h that works with all 6
    
    h = pts6[3][1][0]  # height of top line
    
    # Try x-coordinates on y=0
    for x in range(-100, 101):
        if time.time() - start > time_limit:
            break
        new_pt = [[x, 1], [0, 1]]
        
        # Check if this point is already in the set
        if any(p[0][0] == x and p[0][1] == 1 and p[1][0] == 0 and p[1][1] == 1 for p in pts6):
            continue
        
        candidate = pts6 + [new_pt]
        
        if _validate(candidate, D):
            return candidate
    
    # Try x-coordinates on y=h
    for x in range(-100, 101):
        if time.time() - start > time_limit:
            break
        new_pt = [[x, 1], [h, 1]]
        
        # Check if this point is already in the set
        if any(p[0][0] == x and p[0][1] == 1 and p[1][0] == h and p[1][1] == 1 for p in pts6):
            continue
        
        candidate = pts6 + [new_pt]
        
        if _validate(candidate, D):
            return candidate
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_result = None
    best_score = 0
    
    # Start with the known 5-point solution
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    
    deadline = _START_TIME + 140.0
    
    # Try to find a 6-point set
    if time.time() < deadline - 20:
        pts6 = _find_6_point_set(D=1, time_limit=deadline - time.time() - 20)
        if pts6:
            if _validate(pts6, 1):
                best_score = 6
                best_result = {'D': 1, 'points': pts6}
        
        # If we found 6, try to push for 7
        if best_score == 6 and time.time() < deadline - 10:
            pts7 = _find_7_point_set(D=1, time_limit=deadline - time.time() - 10)
            if pts7:
                if _validate(pts7, 1):
                    best_score = 7
                    best_result = {'D': 1, 'points': pts7}
    
    # Final validation
    if best_result is None:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
        
    return best_result

# EVOLVE-BLOCK-END
