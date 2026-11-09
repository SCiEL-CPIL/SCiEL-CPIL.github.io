import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    a1, b1 = p1
    a2, b2 = p2
    a3, b3 = p3
    a4, b4 = p4
    
    if _are_collinear(p1, p2, p3):
        return False
    
    # Circle through p1, p2, p3: x^2 + y^2 + Dx + Ey + F = 0
    # Solve for D, E, F
    # (a2-a1)^2 + (b2-b1)^2 + D(a2-a1) + E(b2-b1) = 0
    # (a3-a1)^2 + (b3-b1)^2 + D(a3-a1) + E(b3-b1) = 0
    
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - b1
    
    det = x1*y2 - x2*y1
    if det == 0:
        return False
    
    rhs1 = -(x1*x1 + y1*y1)
    rhs2 = -(x2*x2 + y2*y2)
    
    num_D = (rhs1*y2 - rhs2*y1)
    num_E = (x1*rhs2 - x2*rhs1)
    
    # F = -(a1^2 + b1^2 + D*a1 + E*b1)
    num_F = -((a1*a1 + b1*b1)*det + num_D*a1 + num_E*b1)
    
    # Check if p4 is on the circle: a4^2 + b4^2 + D*a4 + E*b4 + F = 0
    check = (a4*a4 + b4*b4)*det + num_D*a4 + num_E*b4 + num_F
    return check == 0

def _validate_point_set(points):
    n = len(points)
    # Check all pairwise distances are perfect squares
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _try_add_point(base_points, x, y):
    new_point = (x, y)
    if new_point in base_points:
        return False
    for p in base_points:
        dx = x - p[0]
        dy = y - p[1]
        d2 = dx*dx + dy*dy
        if not _is_square(d2):
            return False
    n = len(base_points)
    for i in range(n):
        for j in range(i+1, n):
            if _are_collinear(base_points[i], base_points[j], new_point):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_concyclic(base_points[i], base_points[j], base_points[k], new_point):
                    return False
    return True

def _generate_pythagorean_candidates(base_points, max_r, time_limit):
    """Generate candidate points at integer distance from all base points using Pythagorean triples."""
    candidates = set()
    start = time.time()
    
    n = len(base_points)
    for i in range(n):
        for j in range(i+1, n):
            p1 = base_points[i]
            p2 = base_points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            d2 = dx*dx + dy*dy
            d = math.isqrt(d2)
            if d*d != d2:
                continue
            
            # Find r1, r2 such that circles intersect at integer points
            for r1 in range(1, max_r+1):
                if time.time() > time_limit:
                    return list(candidates)
                for r2 in range(abs(r1-d), r1+d+1):
                    if r2 < 1:
                        continue
                    # Check if circles intersect
                    num_a = r1*r1 - r2*r2 + d2
                    if num_a % (2*d) != 0:
                        continue
                    a = num_a // (2*d)
                    
                    h2 = r1*r1 - a*a
                    if h2 < 0:
                        continue
                    h = math.isqrt(h2)
                    if h*h != h2:
                        continue
                    if h == 0:
                        x = p1[0] + a*dx//d
                        y = p1[1] + a*dy//d
                        cand = (x, y)
                        if cand not in base_points:
                            candidates.add(cand)
                        x2 = p1[0] - (2*a - d)*dx//d
                        y2 = p1[1] - (2*a - d)*dy//d
                        cand2 = (x2, y2)
                        if cand2 not in base_points:
                            candidates.add(cand2)
                    else:
                        if (-dy*h) % d != 0 or (dx*h) % d != 0:
                            continue
                        px = -dy*h//d
                        py = dx*h//d
                        
                        x1 = p1[0] + a*dx//d + px
                        y1 = p1[1] + a*dy//d + py
                        cand1 = (x1, y1)
                        
                        x2 = p1[0] + a*dx//d - px
                        y2 = p1[1] + a*dy//d - py
                        cand2 = (x2, y2)
                        
                        if cand1 not in base_points:
                            candidates.add(cand1)
                        if cand2 not in base_points:
                            candidates.add(cand2)
    
    return list(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate_point_set(base5):
        base5 = [(0, 0), (3, 4), (0, 5)]
        if not _validate_point_set(base5):
            return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    best_set = base5.copy()
    
    # Strategy 1: Pythagorean candidate generation with larger range
    if time.time() < deadline - 15:
        candidates = _generate_pythagorean_candidates(base5, 500, deadline - 15)
        candidates = [p for p in candidates if p not in base5]
        
        cx = sum(p[0] for p in base5) / 5
        cy = sum(p[1] for p in base5) / 5
        candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            if _try_add_point(base5, cand[0], cand[1]):
                new_set = base5 + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    break
    
    # Try to find 7th point
    if len(best_set) == 6 and time.time() < deadline - 10:
        candidates7 = _generate_pythagorean_candidates(best_set, 300, deadline - 10)
        candidates7 = [p for p in candidates7 if p not in best_set]
        cx = sum(p[0] for p in best_set) / len(best_set)
        cy = sum(p[1] for p in best_set) / len(best_set)
        candidates7.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in candidates7:
            if time.time() > deadline - 3:
                break
            if _try_add_point(best_set, cand[0], cand[1]):
                new_set = best_set + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

# EVOLVE-BLOCK-END
