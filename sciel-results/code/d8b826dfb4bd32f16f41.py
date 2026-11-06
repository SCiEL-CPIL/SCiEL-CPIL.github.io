import math
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _dist_is_rational(p1, p2):
    d2 = _sq_dist(p1, p2)
    return _is_square(d2)

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]
                    x3, y3 = p3[0], p3[1]
                    x4, y4 = p4[0], p4[1]
                    
                    a1 = (x2*x2 - x1*x1) + (y2*y2 - y1*y1)
                    b1 = x2 - x1
                    c1 = y2 - y1
                    a2 = (x3*x3 - x1*x1) + (y3*y3 - y1*y1)
                    b2 = x3 - x1
                    c2 = y3 - y1
                    
                    det = b1 * c2 - b2 * c1
                    if det == 0:
                        continue
                    
                    A = (-a1 * c2 + a2 * c1) / det
                    B = (b1 * a2 - b2 * a1) / det
                    C = -(x1*x1 + y1*y1 + A * x1 + B * y1)
                    
                    lhs = x4*x4 + y4*y4 + A * x4 + B * y4 + C
                    if abs(lhs) < 1e-9:
                        return False
    return True

def _validate(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_is_rational(points[i], points[j]):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Known 5-point set
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate(base5):
        # Fallback to a simpler set if validation fails (should not)
        base5 = [(0, 0), (3, 4), (0, 4), (4, 3), (4, 0)]
        pass
    
    best_points = base5
    
    # Try to find a 6th point by random search over rationals
    # We search in a box [-500, 500] x [-500, 500]
    # But we need to be fast.
    
    # Let's try a few random points
    for _ in range(5000):
        if time.time() > deadline:
            break
        
        import random
        x = random.randint(-500, 500)
        y = random.randint(-500, 500)
        
        if (x, y) in base5:
            continue
            
        # Check distances to all 5
        valid = True
        for p in base5:
            if not _dist_is_rational((x, y), p):
                valid = False
                break
        
        if not valid:
            continue
            
        # Check collinearity
        if not _no_three_collinear(base5 + [(x, y)]):
            continue
            
        # Check concyclicity
        if not _no_four_concyclic(base5 + [(x, y)]):
            continue
            
        # Found a 6th point!
        best_points = base5 + [(x, y)]
        break
        
    # If we found 6, try to find 7?
    # Unlikely in this time.
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
