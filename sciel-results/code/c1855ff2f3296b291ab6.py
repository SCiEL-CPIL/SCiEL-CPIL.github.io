import time
from math import isqrt, gcd
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_valid(pts):
    """Check all constraints for a set of integer points."""
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    # Check all pairwise distances are integers
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[i], pts[j], pts[k], pts[l]
                    a11 = x1 - x2
                    a12 = y1 - y2
                    b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                    a21 = x1 - x3
                    a22 = y1 - y3
                    b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                    det = a11 * a22 - a21 * a12
                    if det == 0:
                        continue
                    A_num = b1 * a22 - b2 * a12
                    B_num = a11 * b2 - a21 * b1
                    C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                    lhs = x4 * A_num + y4 * B_num + C_num
                    rhs = -(x4*x4 + y4*y4) * det
                    if lhs == rhs:
                        return False
    return True

def _find_new_point(base_pts, x_min, x_max, y_min, y_max, deadline):
    """Search for a new point that can be added to base_pts."""
    n = len(base_pts)
    # Pre-compute constants for line and circle checks
    # Line: (x2-x1)(y-y1) - (y2-y1)(x-x1) = 0
    # Circle: x^2+y^2 + Ax + By + C = 0
    # For each triple, compute A, B, C
    
    triples = list(combinations(range(n), 3))
    circle_params = []
    for i, j, k in triples:
        x1, y1 = base_pts[i]
        x2, y2 = base_pts[j]
        x3, y3 = base_pts[k]
        # Circle through (x1,y1), (x2,y2), (x3,y3)
        # x^2 + y^2 + Dx + Ey + F = 0
        # x1^2 + y1^2 + D x1 + E y1 + F = 0
        # ...
        # Subtract eq1 from eq2:
        # (x2^2-x1^2) + (y2^2-y1^2) + D(x2-x1) + E(y2-y1) = 0
        # D(x2-x1) + E(y2-y1) = -(x2^2-x1^2 + y2^2-y1^2)
        # Similarly for 3,1
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = -(x2*x2 - x1*x1 + y2*y2 - y1*y1)
        
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = -(x3*x3 - x1*x1 + y3*y3 - y1*y1)
        
        det = a1 * b2 - a2 * b1
        if det == 0:
            continue # Collinear, no unique circle
        
        D = (c1 * b2 - c2 * b1) / det
        E = (a1 * c2 - a2 * c1) / det
        F = -(x1*x1 + y1*y1 + D*x1 + E*y1)
        
        # Store as fractions to avoid floating point errors?
        # Or use the integer form:
        # det * (x^2 + y^2) + D_int * x + E_int * y + F_int = 0
        # D_int = c1 * b2 - c2 * b1
        # E_int = a1 * c2 - a2 * c1
        # F_int = -(x1*x1 + y1*y1) * det - x1 * D_int - y1 * E_int
        D_int = c1 * b2 - c2 * b1
        E_int = a1 * c2 - a2 * c1
        F_int = -(x1*x1 + y1*y1) * det - x1 * D_int - y1 * E_int
        circle_params.append((det, D_int, E_int, F_int))

    # Pre-compute line params
    # Line through i, j: A x + B y + C = 0
    # A = y1 - y2
    # B = x2 - x1
    # C = x1 y2 - x2 y1
    lines = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = base_pts[i]
            x2, y2 = base_pts[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            lines.append((A, B, C))

    # Search
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            return None
        for y in range(y_min, y_max + 1):
            # Distance check
            valid_dist = True
            for p in base_pts:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2):
                    valid_dist = False
                    break
            if not valid_dist:
                continue
            
            # Collinearity check
            valid_collinear = True
            for A, B, C in lines:
                if A * x + B * y + C == 0:
                    valid_collinear = False
                    break
            if not valid_collinear:
                continue
            
            # Concyclic check
            valid_concyclic = True
            for det, D_int, E_int, F_int in circle_params:
                if det * (x * x + y * y) + D_int * x + E_int * y + F_int == 0:
                    valid_concyclic = False
                    break
            if not valid_concyclic:
                continue
            
            return (x, y)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Seed 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = seed_points if _check_valid(seed_points) else None
    
    if not best:
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    # Try to find a 6th point
    # Search a large box
    # The seed points are in [-56, 16] x [0, 63]
    # Let's search [-500, 500] x [-500, 500]
    # If not found, try larger or different D
    
    p6 = _find_new_point(best, -500, 500, -500, 500, deadline)
    if p6:
        best = best + [p6]
    
    if len(best) == 6 and time.time() < deadline - 10:
        # Try to find a 7th point
        p7 = _find_new_point(best, -500, 500, -500, 500, deadline)
        if p7:
            best = best + [p7]

    # If still 5 points, try other D values?
    # Or try a different base set?
    # For now, return the best found.
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
