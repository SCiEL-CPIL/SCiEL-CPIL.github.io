"""Search for integral point sets with rational distances, no 3 collinear, no 4 concyclic."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import itertools
from math import isqrt

def _is_square(frac):
    """Check if a Fraction is a perfect square of a rational. Uses exact integer arithmetic."""
    if frac < 0:
        return False
    num = frac.numerator
    den = frac.denominator
    inner = num * den
    if inner < 0:
        return False
    s = isqrt(inner)
    return s * s == inner

def _dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                cross = dx1*dy2 - dy1*dx2
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    rows = []
                    for idx in [i, j, k, m]:
                        x = points[idx][0]
                        y = points[idx][1]
                        rows.append([x*x + y*y, x, y, Fraction(1)])
                    a = rows[0]; b = rows[1]; c = rows[2]; d = rows[3]
                    det = (a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[1]*d[3]-c[3]*d[1]) + b[3]*(c[1]*d[2]-c[2]*d[1]))
                         - a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[2]-c[2]*d[0]))
                         + a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1]) - b[1]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[1]-c[1]*d[0]))
                         - a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1]) - b[1]*(c[0]*d[2]-c[2]*d[0]) + b[2]*(c[0]*d[1]-c[1]*d[0])))
                    if det == 0:
                        return False
    return True

def _valid_set(points):
    """Check all constraints for a set of points."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            d2 = _dist_sq(points[i], points[j])
            if not _is_square(d2):
                return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def _find_6_point_set(deadline):
    """Find a 6-point set using base triangles and systematic search."""
    base_triangles = [
        ((Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(3), Fraction(4))),
        ((Fraction(0), Fraction(0)), (Fraction(4), Fraction(0)), (Fraction(0), Fraction(3))),
        ((Fraction(0), Fraction(0)), (Fraction(6), Fraction(0)), (Fraction(0), Fraction(8))),
        ((Fraction(0), Fraction(0)), (Fraction(9), Fraction(0)), (Fraction(0), Fraction(12))),
    ]
    
    for p1, p2, p3 in base_triangles:
        if time.time() > deadline:
            break
        
        candidates = set()
        
        # Search for points at rational distance from all three base points
        # Using integer distance approach for the 3-4-5 triangle
        if p1 == (Fraction(0), Fraction(0)) and p2 == (Fraction(5), Fraction(0)) and p3 == (Fraction(3), Fraction(4)):
            limit = 150
            for d1 in range(1, limit + 1):
                if time.time() > deadline:
                    break
                for d2 in range(1, limit + 1):
                    if d1 + d2 < 5 or d1 + 5 < d2 or d2 + 5 < d1:
                        continue
                    num_x = d1*d1 - d2*d2 + 25
                    x = Fraction(num_x, 10)
                    for d3 in range(1, limit + 1):
                        if d1 + d3 < 5 or d1 + 5 < d3 or d3 + 5 < d1:
                            continue
                        if d2 + d3 < 5 or d2 + 5 < d3 or d3 + 5 < d2:
                            continue
                        num_y = d1*d1 - d3*d3 + 25 - 6*x
                        y = Fraction(num_y, 8)
                        if x*x + y*y == d1*d1:
                            if (x, y) not in (p1, p2, p3):
                                candidates.add((x, y))
        else:
            # Generic search: brute force in bounded region
            search_limit = 25
            for x_num in range(-search_limit * 4, search_limit * 4 + 1, 2):
                if time.time() > deadline:
                    break
                for x_den in [1, 2]:
                    x = Fraction(x_num, x_den)
                    for y_num in range(-search_limit * 4, search_limit * 4 + 1, 2):
                        for y_den in [1, 2]:
                            y = Fraction(y_num, y_den)
                            pt = (x, y)
                            if pt in (p1, p2, p3):
                                continue
                            ok = True
                            for bp in [p1, p2, p3]:
                                d2 = _dist_sq(pt, bp)
                                if not _is_square(d2):
                                    ok = False
                                    break
                            if ok:
                                candidates.add(pt)
        
        cand_list = list(candidates)
        n_cand = len(cand_list)
        
        if n_cand < 3:
            continue
        
        # Find triangle among candidates
        for i in range(n_cand):
            if time.time() > deadline:
                break
            for j in range(i+1, n_cand):
                d2_ij = _dist_sq(cand_list[i], cand_list[j])
                if not _is_square(d2_ij):
                    continue
                for k in range(j+1, n_cand):
                    d2_ik = _dist_sq(cand_list[i], cand_list[k])
                    if not _is_square(d2_ik):
                        continue
                    d2_jk = _dist_sq(cand_list[j], cand_list[k])
                    if not _is_square(d2_jk):
                        continue
                    full_set = [p1, p2, p3, cand_list[i], cand_list[j], cand_list[k]]
                    if _valid_set(full_set):
                        return full_set
    
    return None

def construct():
    start = time.time()
    deadline = start + 120  # 120s safety margin (evaluator kills at 150s)
    
    # Try to find 6 points
    result = _find_6_point_set(deadline)
    
    if result:
        return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result]}
    
    # Fallback to known 5-point set
    pts = [(Fraction(0), Fraction(0)), 
           (Fraction(-56), Fraction(33)), 
           (Fraction(-16), Fraction(63)), 
           (Fraction(0), Fraction(33)), 
           (Fraction(16), Fraction(63))]
    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in pts]}
# EVOLVE-BLOCK-END
