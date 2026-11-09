import time
import random
from fractions import Fraction
from math import gcd, isqrt, sqrt

_DEADLINE = time.time() + 125  # 125s wall-clock, 25s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    return gcd(abs(a), abs(b))

def _check_valid(points):
    """Check all constraints: distinct, rational distances, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if d2 <= 0:
                return False
            if not _is_square(d2):
                return False
    # No three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    return False
    # No four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    a1 = x1 * x1 + y1 * y1
                    a2 = x2 * x2 + y2 * y2
                    a3 = x3 * x3 + y3 * y3
                    a4 = x4 * x4 + y4 * y4
                    b2 = a2 - a1
                    b3 = a3 - a1
                    b4 = a4 - a1
                    c2 = x2 - x1
                    c3 = x3 - x1
                    c4 = x4 - x1
                    d2v = y2 - y1
                    d3v = y3 - y1
                    d4v = y4 - y1
                    det3 = (b2 * (c3 * d4v - d3v * c4)
                            - c2 * (b3 * d4v - d3v * b4)
                            + d2v * (b3 * c4 - c3 * b4))
                    if det3 == 0:
                        return False
    return True

def _find_candidate_two_circle(P0, P1, r0, r1):
    """
    Find intersection points of circle centered at P0 radius r0 and P1 radius r1.
    Returns list of integer points (x, y) if they are integers.
    """
    x0, y0 = P0
    x1, y1 = P1
    dx = x1 - x0
    dy = y1 - y0
    d2 = dx * dx + dy * dy
    if d2 == 0:
        return []
    
    # d is distance between centers
    # We work in integers/Fractions
    # Let u = (x-x0, y-y0)
    # u . (dx, dy) = (r0^2 - r1^2 + d2) / 2
    # Let dot = (r0^2 - r1^2 + d2) / 2
    # If dot is not integer, no integer solution?
    # Actually, x,y integers => dot must be integer?
    # (x-x0)dx + (y-y0)dy = dot
    # LHS is integer. So dot must be integer.
    num_dot = r0 * r0 - r1 * r1 + d2
    if num_dot % 2 != 0:
        return []
    dot = num_dot // 2
    
    # u is projection of u on v plus perpendicular component
    # v = (dx, dy)
    # proj = (dot / d2) * v
    # For x,y to be integers, we need to handle the division carefully.
    # Let's just solve the system using Fractions to be safe, then check if integer.
    
    # System:
    # (x-x0)^2 + (y-y0)^2 = r0^2
    # (x-x1)^2 + (y-y1)^2 = r1^2
    
    # Subtracting:
    # 2*x*(x0-x1) + 2*y*(y0-y1) + (x1^2+y1^2 - x0^2-y0^2) = r0^2 - r1^2
    # Let A = 2*(x0-x1), B = 2*(y0-y1), C = (x1^2+y1^2 - x0^2-y0^2) - (r0^2 - r1^2)
    # A*x + B*y + C = 0
    
    A = 2 * (x0 - x1)
    B = 2 * (y0 - y1)
    C = (x1 * x1 + y1 * y1 - x0 * x0 - y0 * y0) - (r0 * r0 - r1 * r1)
    
    # Line: A*x + B*y = -C
    # We need integer solutions (x,y) to this line that also lie on the circle.
    # This is a Diophantine problem.
    # However, we can parameterize the line.
    # Find one particular solution? Or just iterate?
    # Since we are iterating r0, r1, we can just check if the intersection is rational and integer.
    
    # Let's use the geometric formula with Fractions.
    d = Fraction(isqrt(d2), 1) # Wait, d might not be integer?
    # In our problem, distances between existing points ARE integers.
    # So d2 is a perfect square.
    if not _is_square(d2):
        return []
    d = isqrt(d2)
    
    # a = (r0^2 - r1^2 + d^2) / (2d)
    num_a = r0 * r0 - r1 * r1 + d * d
    den_a = 2 * d
    if num_a % den_a != 0:
        return []
    a = num_a // den_a
    
    # h^2 = r0^2 - a^2
    h2 = r0 * r0 - a * a
    if h2 < 0:
        return []
    if not _is_square(h2):
        return []
    h = isqrt(h2)
    
    # Center of the base segment P0P1
    # Midpoint M = ((x0+x1)/2, (y0+y1)/2)
    # Vector P0->P1 is (dx, dy).
    # Perpendicular vector is (-dy, dx).
    # The intersection points are M + (a - d/2) * (dx, dy)/d  +/- h * (-dy, dx)/d
    # Wait, a is distance from P0.
    # Vector from P0 to P1 is v = (dx, dy).
    # Point on line P0P1 at distance a from P0 is P0 + (a/d) * v.
    # Let's call this Q.
    # Qx = x0 + a * dx / d
    # Qy = y0 + a * dy / d
    # For Q to be rational, a*dx/d must be rational.
    # Since a, dx, d are integers, this is rational.
    # The intersection points are Q +/- (h/d) * (-dy, dx)
    
    # Let's compute Qx, Qy as Fractions
    Qx = Fraction(x0 * d + a * dx, d)
    Qy = Fraction(y0 * d + a * dy, d)
    
    # Vector w = (-dy, dx)
    # Displacement = (h/d) * w
    # Dx = h * (-dy) / d
    # Dy = h * dx / d
    
    Dx = Fraction(-h * dy, d)
    Dy = Fraction(h * dx, d)
    
    candidates = []
    # Point 1: Q + (Dx, Dy)
    x1_c = Qx + Dx
    y1_c = Qy + Dy
    if x1_c.denominator == 1 and y1_c.denominator == 1:
        candidates.append((x1_c.numerator, y1_c.numerator))
        
    # Point 2: Q - (Dx, Dy)
    if h > 0: # If h=0, points are same
        x2_c = Qx - Dx
        y2_c = Qy - Dy
        if x2_c.denominator == 1 and y2_c.denominator == 1:
            candidates.append((x2_c.numerator, y2_c.numerator))
            
    return candidates

def _try_extend(current, max_points=12, time_budget_s=110):
    """Try to add points to the set using two-circle intersection search."""
    current = list(current)
    n = len(current)
    if n < 3 or not _check_valid(current):
        return current
    
    deadline_inner = time.time() + time_budget_s
    tried_global = set()
    
    # Precompute distances between existing points
    # We will use pairs (i, j) as anchors
    
    while n < max_points and time.time() < deadline_inner and time.time() < _DEADLINE - 15:
        found = False
        
        # Try each pair of existing points as anchors
        # Sort pairs by distance? Smaller distance might mean smaller radii needed?
        # Let's just iterate all pairs.
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        
        random.shuffle(pairs)
        
        for i, j in pairs:
            if time.time() > deadline_inner or time.time() > _DEADLINE - 15:
                break
            P0 = current[i]
            P1 = current[j]
            
            # Max radius to check
            # Let's start with 200 and increase if needed?
            # Or just scan a fixed range.
            max_r = 300
            
            for r0 in range(1, max_r + 1):
                if time.time() > deadline_inner or time.time() > _DEADLINE - 15:
                    break
                for r1 in range(1, max_r + 1):
                    cands = _find_candidate_two_circle(P0, P1, r0, r1)
                    for cx, cy in cands:
                        if (cx, cy) in current or (cx, cy) in tried_global:
                            continue
                        tried_global.add((cx, cy))
                        
                        # Check distances to all other points
                        ok = True
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            qx, qy = current[k]
                            dx = cx - qx
                            dy = cy - qy
                            if not _is_square(dx * dx + dy * dy):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        new_set = current + [(cx, cy)]
                        if _check_valid(new_set):
                            current.append((cx, cy))
                            n += 1
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                tried_global = set()
                break # Restart outer loop with new set
                
    return current

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 125
    
    # Start with the known 5-point set
    base_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best = None
    if _check_valid(base_5):
        best = list(base_5)
    
    # Try to extend the 5-point set
    if best is not None and len(best) == 5:
        extended = _try_extend(best, max_points=12, time_budget_s=100)
        if len(extended) > len(best):
            best = extended
    
    # If still 5, try other starting configurations or random restarts
    if best is None or len(best) <= 5:
        other_starts = [
            [(0, 0), (5, 0), (0, 12)],
            [(0, 0), (3, 4), (5, 0)],
            [(0, 0), (7, 24), (15, 8)],
            [(0, 0), (9, 12), (15, 0)],
            [(0, 0), (12, 5), (13, 0)],
            [(0, 0), (20, 21), (29, 0)],
            [(0, 0), (8, 15), (17, 0)],
            [(0, 0), (11, 60), (61, 0)],
            [(0, 0), (28, 45), (53, 0)],
            [(0, 0), (33, 56), (65, 0)],
            [(0, 0), (48, 55), (73, 0)],
            [(0, 0), (65, 72), (97, 0)],
            [(0, 0), (20, 99), (101, 0)],
        ]
        for start in other_starts:
            if _time_left() < 10:
                break
            if _check_valid(start):
                ext = _try_extend(start, max_points=12, time_budget_s=min(_time_left() - 10, 50))
                if best is None or len(ext) > len(best):
                    best = ext
    
    # Fallback
    if best is None or len(best) < 3:
        best = base_5
    
    if not _check_valid(best):
        best = base_5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
