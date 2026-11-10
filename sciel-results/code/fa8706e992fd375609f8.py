import time
import math
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 115.0  # hard wall-clock budget (safety margin vs 120s)

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _check_rational_distance(p1, p2, D):
    # p1, p2 are (x_num, x_den, y_num, y_den)
    dx_num = p1[0] * p2[1] - p2[0] * p1[1]
    dx_den = p1[1] * p2[1]
    dy_num = p1[2] * p2[3] - p2[2] * p1[3]
    dy_den = p1[3] * p2[3]
    
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    
    if den == 0:
        return False
    return _is_square(num)

def _check_collinear(p1, p2, p3):
    x1n, x1d, y1n, y1d = p1
    x2n, x2d, y2n, y2d = p2
    x3n, x3d, y3n, y3d = p3
    
    x1 = Fraction(x1n, x1d)
    y1 = Fraction(y1n, y1d)
    x2 = Fraction(x2n, x2d)
    y2 = Fraction(y2n, y2d)
    x3 = Fraction(x3n, x3d)
    y3 = Fraction(y3n, y3d)
    
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def _check_concyclic(p1, p2, p3, p4):
    pts = []
    for p in (p1, p2, p3, p4):
        x = Fraction(p[0], p[1])
        y = Fraction(p[2], p[3])
        pts.append((x, y))
    
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    x4, y4 = pts[3]
    
    x4sq = x4*x4 + y4*y4
    r1 = (x1*x1 + y1*y1 - x4sq, x1 - x4, y1 - y4)
    r2 = (x2*x2 + y2*y2 - x4sq, x2 - x4, y2 - y4)
    r3 = (x3*x3 + y3*y3 - x4sq, x3 - x4, y3 - y4)
    
    det = (r1[0] * (r2[1]*r3[2] - r2[2]*r3[1]) - 
           r1[1] * (r2[0]*r3[2] - r2[2]*r3[0]) + 
           r1[2] * (r2[0]*r3[1] - r2[1]*r3[0]))
    return det == 0

def _is_valid_set(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_rational_distance(points[i], points[j], D):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _check_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _check_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _try_add_point(current_points, candidate, D):
    for p in current_points:
        if not _check_rational_distance(candidate, p, D):
            return False
    n = len(current_points)
    for i in range(n):
        for j in range(i+1, n):
            if _check_collinear(candidate, current_points[i], current_points[j]):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _check_concyclic(candidate, current_points[i], current_points[j], current_points[k]):
                    return False
    return True

def _generate_circle_candidates(points, D, max_param=20):
    """Generate candidates on circles through pairs of existing points.
    
    A point on a circle through two existing points P1, P2 has rational distance
    to both P1 and P2. We parametrize the circle and generate candidates.
    """
    candidates = []
    n = len(points)
    
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = points[i], points[j]
            x1 = Fraction(p1[0], p1[1])
            y1 = Fraction(p1[2], p1[3])
            x2 = Fraction(p2[0], p2[1])
            y2 = Fraction(p2[2], p2[3])
            
            # Circle through P1, P2 and origin (for simplicity)
            # Actually, let's use a circle through P1, P2 with rational center
            # The perpendicular bisector of P1P2 gives the center locus
            
            # For now, try circles through P1, P2 and a third reference point
            # Let's try circle through P1, P2, and (0,0) if not collinear
            if not _check_collinear(p1, p2, [0, 1, 0, 1]):
                # Circle through (0,0), P1, P2
                # Equation: x^2 + y^2 + Ax + By = 0
                # A*x1 + B*y1 = -(x1^2 + y1^2)
                # A*x2 + B*y2 = -(x2^2 + y2^2)
                rhs1 = -(x1*x1 + y1*y1)
                rhs2 = -(x2*x2 + y2*y2)
                
                det = x1*y2 - x2*y1
                if det != 0:
                    A = (rhs1*y2 - rhs2*y1) / det
                    B = (x1*rhs2 - x2*rhs1) / det
                    r2 = A*A/4 + B*B/4  # radius squared
                    
                    # Parametrize: x = -A/2 + (r/2)*cos(t), y = -B/2 + (r/2)*sin(t)
                    # For rational points, use rational parametrization of circle
                    # x = -A/2 + (r/2)*(1-t^2)/(1+t^2), y = -B/2 + (r/2)*2t/(1+t^2)
                    # This requires r to be rational, which may not be the case
                    
                    # Alternative: try integer/rational points near the circle
                    cx = -A/2
                    cy = -B/2
                    r = (A*A + B*B) / 4  # r^2
                    
                    # Try rational t values
                    for t_num in range(-max_param, max_param+1):
                        for t_den in [1, 2, 3, 4, 5]:
                            if t_num == 0:
                                continue
                            t = Fraction(t_num, t_den)
                            # x = cx + (r2 - (cy)^2) * (1-t^2)/(1+t^2) ... this is getting complex
                            # Let's just try nearby integer points
                            pass
                    
                    # Simpler: try integer points in a region around the circle
                    cx_int = int(cx)
                    cy_int = int(cy)
                    r_int = int(math.sqrt(r2)) if r2 >= 0 else 0
                    if r_int > 0 and r_int < 500:
                        for dx in range(-r_int-2, r_int+3):
                            for dy in range(-r_int-2, r_int+3):
                                if dx*dx + dy*dy > (r_int+2)**2:
                                    continue
                                cand_x = cx_int + dx
                                cand_y = cy_int + dy
                                cand = [cand_x, 1, cand_y, 1]
                                candidates.append(cand)
    
    # Also generate candidates from lines through pairs (for collinearity avoidance)
    # And from a grid
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with the known 5-point set
    base_points = [
        [0, 1, 0, 1], 
        [-56, 1, 33, 1], 
        [-16, 1, 63, 1], 
        [0, 1, 33, 1], 
        [16, 1, 63, 1]
    ]
    D = 1
    
    if not _is_valid_set(base_points, D):
        base_points = [
            [0, 1, 0, 1],
            [3, 1, 4, 1],
            [0, 1, 4, 1]
        ]
        D = 1

    current_points = base_points[:]
    current_set = set(tuple(p) for p in current_points)
    
    # Try multiple search strategies
    strategies = [
        'grid_small', 'grid_medium', 'circle', 'rational'
    ]
    
    for strategy in strategies:
        if time.time() > deadline - 10.0:
            break
        
        added_any = True
        while added_any and time.time() < deadline - 10.0:
            added_any = False
            
            if strategy == 'grid_small':
                max_coord = 50
                candidates = []
                for x in range(-max_coord, max_coord + 1):
                    for y in range(-max_coord, max_coord + 1):
                        cand = [x, 1, y, 1]
                        if tuple(cand) not in current_set:
                            candidates.append(cand)
                
            elif strategy == 'grid_medium':
                max_coord = 200
                candidates = []
                for x in range(-max_coord, max_coord + 1):
                    for y in range(-max_coord, max_coord + 1):
                        cand = [x, 1, y, 1]
                        if tuple(cand) not in current_set:
                            candidates.append(cand)
                
            elif strategy == 'circle':
                candidates = _generate_circle_candidates(current_points, D, max_param=10)
                # Filter out existing
                candidates = [c for c in candidates if tuple(c) not in current_set]
                
            elif strategy == 'rational':
                candidates = []
                for _ in range(5000):
                    if time.time() > deadline - 10.0:
                        break
                    x_num = random.randint(-30, 30)
                    x_den = random.randint(1, 10)
                    y_num = random.randint(-30, 30)
                    y_den = random.randint(1, 10)
                    cand = [x_num, x_den, y_num, y_den]
                    g1 = _gcd(abs(x_num), x_den)
                    g2 = _gcd(abs(y_num), y_den)
                    if g1 > 0:
                        cand[0] //= g1; cand[1] //= g1
                    if g2 > 0:
                        cand[2] //= g2; cand[3] //= g2
                    if cand[0] < 0:
                        cand[0] = -cand[0]; cand[1] = -cand[1]
                    if cand[2] < 0:
                        cand[2] = -cand[2]; cand[3] = -cand[3]
                    if tuple(cand) not in current_set:
                        candidates.append(cand)
            
            # Filter candidates
            valid_candidates = []
            for p in candidates:
                if _try_add_point(current_points, p, D):
                    valid_candidates.append(p)
            
            if valid_candidates:
                valid_candidates.sort(key=lambda p: p[0]*p[0]*p[1]*p[1] + p[2]*p[2]*p[3]*p[3])
                new_point = valid_candidates[0]
                current_points.append(new_point)
                current_set.add(tuple(new_point))
                added_any = True
                
                if time.time() > deadline - 15.0:
                    break
        
        # If we found points, try to add more with the same strategy
        # If not, move to next strategy
    
    # Final attempt with larger grid if time permits
    if time.time() < deadline - 20.0:
        max_coord = 500
        added_any = True
        while added_any and time.time() < deadline - 15.0:
            added_any = False
            candidates = []
            for x in range(-max_coord, max_coord + 1, 3):
                for y in range(-max_coord, max_coord + 1, 3):
                    cand = [x, 1, y, 1]
                    if tuple(cand) not in current_set:
                        candidates.append(cand)
            
            valid_candidates = []
            for p in candidates:
                if _try_add_point(current_points, p, D):
                    valid_candidates.append(p)
            
            if valid_candidates:
                valid_candidates.sort(key=lambda p: p[0]*p[0]*p[1]*p[1] + p[2]*p[2]*p[3]*p[3])
                new_point = valid_candidates[0]
                current_points.append(new_point)
                current_set.add(tuple(new_point))
                added_any = True
    
    # Convert to required format
    final_points = []
    for p in current_points:
        x_num, x_den, y_num, y_den = p
        final_points.append([[x_num, x_den], [y_num, y_den]])
        
    return {'D': D, 'points': final_points}

if __name__ == "__main__":
    construct()
