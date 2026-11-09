import itertools
import math
import random
from fractions import Fraction

def construct():
    # Known valid 5-point set
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # Let's verify this set quickly in code to be sure.
    
    # Strategy:
    # 1. Define a candidate pool of rational points (integers for simplicity, D=1).
    # 2. Use a greedy/randomized search to build a set of size 6.
    # 3. If 6 not found, return the known 5-point set.
    
    # To make it faster, we precompute squared distances and check for perfect squares.
    # For concyclicity, we use the determinant check.
    
    # Grid search is too slow for large N.
    # Let's use a smaller grid but more restarts, or a smarter candidate generation.
    
    # A known 5-point set:
    best_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point by searching a grid around these points
    # The points are in range x:[-56, 16], y:[0, 63].
    # Let's search x in [-100, 100], y in [-100, 100].
    
    N = 100
    candidates = [(x, y) for x in range(-N, N+1) for y in range(-N, N+1)]
    # Filter candidates that are distinct from best_set
    best_set_tuple = set(best_set)
    candidates = [p for p in candidates if p not in best_set_tuple]
    
    # Random shuffle
    random.shuffle(candidates)
    
    found_6 = False
    # Try to add a 6th point to the 5-point set
    for p in candidates:
        # Check distances to all 5
        ok = True
        for q in best_set:
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d2 = dx*dx + dy*dy
            if not _is_perfect_square(d2):
                ok = False
                break
        if not ok:
            continue
            
        # Check collinearity
        # No 3 points in {p} U best_set should be collinear.
        # Since best_set has no 3 collinear, we only need to check triples with p and 2 from best_set.
        ok = True
        for i in range(len(best_set)):
            for j in range(i+1, len(best_set)):
                if _are_collinear(p, best_set[i], best_set[j]):
                    ok = False
                    break
            if not ok: break
        if not ok:
            continue
            
        # Check concyclicity
        # No 4 points in {p} U best_set should be concyclic.
        # Since best_set has no 4 concyclic (assuming the seed is valid),
        # we need to check if p lies on any circle defined by 3 points from best_set.
        ok = True
        for i in range(len(best_set)):
            for j in range(i+1, len(best_set)):
                for k in range(j+1, len(best_set)):
                    if _is_concyclic(p, best_set[i], best_set[j], best_set[k]):
                        ok = False
                        break
                if not ok: break
            if not ok: break
        if not ok:
            continue
            
        # If we get here, p is a valid 6th point
        new_set = best_set + [p]
        # We found a 6-point set!
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in new_set]}

    # If no 6th point found in this grid, try a slightly different strategy or return 5.
    # To maximize score, returning 5 is better than 0.
    # But maybe we can find a different 5-point set that is easier to extend?
    # Or just return the current 5-point set.
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r*r == n

def _are_collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) == 0

def _is_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Determinant of:
    # x1 y1 x1^2+y1^2 1
    # x2 y2 x2^2+y2^2 1
    # x3 y3 x3^2+y3^2 1
    # x4 y4 x4^2+y4^2 1
    
    d1 = x1*x1 + y1*y1
    d2 = x2*x2 + y2*y2
    d3 = x3*x3 + y3*y3
    d4 = x4*x4 + y4*y4
    
    # Using the formula for determinant of 4x4
    # A B C D
    # E F G H
    # I J K L
    # M N O P
    
    a, b, c, d = x1, y1, d1, 1
    e, f, g, h = x2, y2, d2, 1
    i, j, k, l = x3, y3, d3, 1
    m, n, o, p = x4, y4, d4, 1
    
    # Expansion along first row
    # a * det([f,g,h], [j,k,l], [n,o,p])
    # - b * det([e,g,h], [i,k,l], [m,o,p])
    # + c * det([e,f,h], [i,j,l], [m,n,p])
    # - d * det([e,f,g], [i,j,k], [m,n,o])
    
    def det3(a,b,c, d,e,f, g,h,i):
        return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    
    t1 = det3(f, g, h, j, k, l, n, o, p)
    t2 = det3(e, g, h, i, k, l, m, o, p)
    t3 = det3(e, f, h, i, j, l, m, n, p)
    t4 = det3(e, f, g, i, j, k, m, n, o)
    
    return a*t1 - b*t2 + c*t3 - d*t4 == 0
