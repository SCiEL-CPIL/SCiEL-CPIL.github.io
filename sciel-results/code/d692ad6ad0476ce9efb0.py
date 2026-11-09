import time
import math

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
    
    rhs1 = -(a2**2 - a1**2 + b2**2 - b1**2)
    rhs2 = -(a3**2 - a1**2 + b3**2 - b1**2)
    
    x1 = a2 - a1
    y1 = b2 - b1
    x2 = a3 - a1
    y2 = b3 - b1
    
    det = x1*y2 - x2*y1
    if det == 0:
        return False
    
    num_D = rhs1*y2 - rhs2*y1
    num_E = x1*rhs2 - x2*rhs1
    num_F = -((a1**2 + b1**2)*det + num_D*a1 + num_E*b1)
    
    check = (a4**2 + b4**2)*det + num_D*a4 + num_E*b4 + num_F
    return check == 0

def _validate_point_set(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not _is_square(d2):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
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

def _find_candidates_with_int_dist(base_points, max_coord):
    candidates = set()
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            valid = True
            for p in base_points:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    valid = False
                    break
            if valid:
                candidates.add((x, y))
    return list(candidates)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    base5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not _validate_point_set(base5):
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[3, 1], [4, 1]], [[0, 1], [5, 1]]]}
    
    best_set = base5.copy()
    
    if time.time() < deadline - 40:
        candidates = _find_candidates_with_int_dist(base5, 120)
        candidates = [p for p in candidates if p not in base5]
        
        cx = sum(p[0] for p in base5) / 5
        cy = sum(p[1] for p in base5) / 5
        candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if _try_add_point(base5, cand[0], cand[1]):
                new_set = base5 + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    break
    
    if len(best_set) == 6 and time.time() < deadline - 20:
        candidates7 = _find_candidates_with_int_dist(best_set, 100)
        candidates7 = [p for p in candidates7 if p not in best_set]
        cx = sum(p[0] for p in best_set) / len(best_set)
        cy = sum(p[1] for p in best_set) / len(best_set)
        candidates7.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for cand in candidates7:
            if time.time() > deadline:
                break
            if _try_add_point(best_set, cand[0], cand[1]):
                new_set = best_set + [cand]
                if _validate_point_set(new_set):
                    best_set = new_set
                    break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

# EVOLVE-BLOCK-END
