"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _check_no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    return False
    return True

def _check_no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    b1 = a2 - a1
                    b2 = a3 - a1
                    b3 = a4 - a1
                    c1 = x2 - x1
                    c2 = x3 - x1
                    c3 = x4 - x1
                    d1 = y2 - y1
                    d2 = y3 - y1
                    d3 = y4 - y1
                    det = (b1 * (c2 * d3 - c3 * d2)
                           - c1 * (b2 * d3 - b3 * d2)
                           + d1 * (b2 * c3 - b3 * c2))
                    if det == 0:
                        return False
    return True

def _all_distances_rational(pts, D):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            d2 = dx * dx + D * dy * dy
            if not _is_square(d2):
                return False
    return True

def _validate_set(pts, D=1):
    if len(pts) < 3:
        return False
    seen = set()
    for p in pts:
        if p in seen:
            return False
        seen.add(p)
    return (_check_no_three_collinear(pts) and 
            _check_no_four_concyclic(pts) and 
            _all_distances_rational(pts, D))

def _find_sixth_point(base_pts, search_radius=200):
    """
    Given a valid 5-point set, search for a 6th integer point
    that has integer distance to all 5 points and satisfies
    no-3-collinear and no-4-concyclic with the existing set.
    """
    n = len(base_pts)
    if n != 5:
        return None
    
    # Calculate bounding box
    xs = [p[0] for p in base_pts]
    ys = [p[1] for p in base_pts]
    min_x = min(xs) - search_radius
    max_x = max(xs) + search_radius
    min_y = min(ys) - search_radius
    max_y = max(ys) + search_radius
    
    base_set = set(base_pts)
    
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            cand = (x, y)
            if cand in base_set:
                continue
            
            valid_dist = True
            for bx, by in base_pts:
                dx = x - bx
                dy = y - by
                d2 = dx*dx + dy*dy
                if not _is_square(d2):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Check collinearity with base points
            collinear_ok = True
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1 = base_pts[i]
                    x2, y2 = base_pts[j]
                    if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) == 0:
                        collinear_ok = False
                        break
                if not collinear_ok:
                    break
            
            if not collinear_ok:
                continue
            
            # Check concyclicity with base points
            # We need to check if any 4 points including cand are concyclic.
            # This means checking all combinations of 3 base points + cand.
            concyclic_ok = True
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        # Check if base_pts[i], base_pts[j], base_pts[k], cand are concyclic
                        p1 = base_pts[i]
                        p2 = base_pts[j]
                        p3 = base_pts[k]
                        p4 = cand
                        
                        a1 = p1[0]*p1[0] + p1[1]*p1[1]
                        a2 = p2[0]*p2[0] + p2[1]*p2[1]
                        a3 = p3[0]*p3[0] + p3[1]*p3[1]
                        a4 = p4[0]*p4[0] + p4[1]*p4[1]
                        
                        b1 = a2 - a1
                        b2 = a3 - a1
                        b3 = a4 - a1
                        c1 = p2[0] - p1[0]
                        c2 = p3[0] - p1[0]
                        c3 = p4[0] - p1[0]
                        d1 = p2[1] - p1[1]
                        d2 = p3[1] - p1[1]
                        d3 = p4[1] - p1[1]
                        
                        det = (b1 * (c2 * d3 - c3 * d2)
                               - c1 * (b2 * d3 - b3 * d2)
                               + d1 * (b2 * c3 - b3 * c2))
                        if det == 0:
                            concyclic_ok = False
                            break
                    if not concyclic_ok:
                        break
                if not concyclic_ok:
                    break
            
            if concyclic_ok:
                return cand
                
    return None

def _search_integral_points(max_coord=150, max_points=8, deadline=135.0):
    start_time = time.time()
    
    # Known 5-point sets
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (3, 4), (0, 5), (-3, 4), (5, 0)],
        [(0, 0), (5, 0), (0, 5), (-5, 0), (3, 4)],
        [(0, 0), (9, 12), (0, 15), (-9, 12), (12, 9)],
        [(0, 0), (7, 24), (0, 25), (-7, 24), (24, 7)],
        [(0, 0), (20, 21), (0, 29), (-20, 21), (21, 20)],
        [(0, 0), (12, 16), (0, 20), (-12, 16), (16, 12)],
        [(0, 0), (8, 15), (0, 17), (-8, 15), (15, 8)],
        [(0, 0), (15, 8), (0, 17), (-15, 8), (8, 15)],
        [(0, 0), (21, 20), (0, 29), (-21, 20), (20, 21)],
        [(0, 0), (24, 7), (0, 25), (-24, 7), (7, 24)],
        [(0, 0), (12, 9), (0, 15), (-12, 9), (9, 12)],
        [(0, 0), (16, 12), (0, 20), (-16, 12), (12, 16)],
        [(0, 0), (28, 45), (0, 53), (-28, 45), (45, 28)],
        [(0, 0), (33, 56), (0, 65), (-33, 56), (56, 33)],
        [(0, 0), (40, 42), (0, 58), (-40, 42), (42, 40)],
        [(0, 0), (45, 28), (0, 53), (-45, 28), (28, 45)],
        [(0, 0), (56, 33), (0, 65), (-56, 33), (33, 56)],
    ]
    
    best_set = None
    best_count = 0
    
    # Validate and find best 5-point set
    for bs in base_sets:
        if _validate_set(bs, 1):
            if len(bs) > best_count:
                best_count = len(bs)
                best_set = list(bs)
                
    if best_set is None:
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best_count = 5
    
    # Try to find a 6th point for the best set
    if time.time() < deadline - 10:
        sixth = _find_sixth_point(best_set, search_radius=100)
        if sixth:
            new_set = best_set + [sixth]
            if _validate_set(new_set, 1):
                best_set = new_set
                best_count = 6
    
    # If we didn't find 6, try other 5-point sets
    if best_count < 6:
        for bs in base_sets:
            if time.time() > deadline - 5:
                break
            if not _validate_set(bs, 1):
                continue
            # Skip if we already tried the best one
            if best_set and set(bs) == set(best_set):
                continue
                
            sixth = _find_sixth_point(bs, search_radius=80)
            if sixth:
                new_set = bs + [sixth]
                if _validate_set(new_set, 1):
                    best_set = new_set
                    best_count = 6
                    break # Found a 6-point set, stop searching for 6
    
    return best_set

def construct():
    start_time = time.time()
    deadline = 135.0
    
    best_set = _search_integral_points(max_coord=150, max_points=8, deadline=deadline)
    
    if not _validate_set(best_set, 1):
        best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
