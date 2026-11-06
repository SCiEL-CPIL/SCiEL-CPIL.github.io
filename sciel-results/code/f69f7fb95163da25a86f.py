"""Search for integral point sets with rational distances, no 3 collinear, no 4 concyclic."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import itertools

def _is_square(frac):
    """Check if a Fraction is a perfect square of a rational."""
    if frac < 0:
        return False
    num = frac.numerator
    den = frac.denominator
    # sqrt(num/den) = sqrt(num*den)/den
    inner = num * den
    if inner < 0:
        return False
    s = int(inner ** 0.5)
    if s * s == inner:
        return True
    s += 1
    if s * s == inner:
        return True
    return False

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

def _find_6_point_set(deadline):
    # Base triangle: (0,0), (5,0), (3,4)
    p1 = (Fraction(0), Fraction(0))
    p2 = (Fraction(5), Fraction(0))
    p3 = (Fraction(3), Fraction(4))
    
    candidates = set()
    
    # Search for points P(x,y) such that dist(P, p1), dist(P, p2), dist(P, p3) are integers
    # d1^2 = x^2 + y^2
    # d2^2 = (x-5)^2 + y^2 = x^2 - 10x + 25 + y^2 = d1^2 - 10x + 25
    # => 10x = d1^2 - d2^2 + 25
    # d3^2 = (x-3)^2 + (y-4)^2 = x^2 - 6x + 9 + y^2 - 8y + 16 = d1^2 - 6x - 8y + 25
    # => 8y = d1^2 - d3^2 + 25 - 6x
    
    limit = 100
    for d1 in range(1, limit):
        for d2 in range(1, limit):
            # Triangle inequality for p1, p2, P
            if d1 + d2 < 5 or d1 + 5 < d2 or d2 + 5 < d1:
                continue
            num_x = d1*d1 - d2*d2 + 25
            # x must be rational, it is.
            x = Fraction(num_x, 10)
            for d3 in range(1, limit):
                # Triangle inequality for p1, p3, P
                if d1 + d3 < 5 or d1 + 5 < d3 or d3 + 5 < d1:
                    continue
                # Triangle inequality for p2, p3, P (dist p2-p3 is 5)
                if d2 + d3 < 5 or d2 + 5 < d3 or d3 + 5 < d2:
                    continue
                
                num_y = d1*d1 - d3*d3 + 25 - 6*x
                y = Fraction(num_y, 8)
                
                # Check if this (x,y) actually satisfies d1^2 = x^2 + y^2
                if x*x + y*y == d1*d1:
                    # Check distinct from base
                    if (x, y) not in (p1, p2, p3):
                        candidates.add((x, y))
        
        if time.time() > deadline:
            break

    # Convert candidates to list
    cand_list = list(candidates)
    
    # Build graph: edge if distance is integer
    n_cand = len(cand_list)
    # We need to find 3 points from cand_list that are mutually at integer distances
    # And also at integer distances to p1, p2, p3 (already guaranteed by construction)
    # So we just need a triangle in the subgraph of candidates.
    
    # Precompute distances between candidates
    # To save time, we can just check on the fly or build adjacency list
    
    # If n_cand is small, brute force triples
    found = None
    if n_cand > 0:
        for i in range(n_cand):
            for j in range(i+1, n_cand):
                p_i = cand_list[i]
                p_j = cand_list[j]
                d2_ij = _dist_sq(p_i, p_j)
                if not _is_square(d2_ij):
                    continue
                for k in range(j+1, n_cand):
                    p_k = cand_list[k]
                    d2_ik = _dist_sq(p_i, p_k)
                    if not _is_square(d2_ik):
                        continue
                    d2_jk = _dist_sq(p_j, p_k)
                    if not _is_square(d2_jk):
                        continue
                    
                    # Found a triangle of candidates
                    # Full set: p1, p2, p3, p_i, p_j, p_k
                    full_set = [p1, p2, p3, p_i, p_j, p_k]
                    
                    # Check no 3 collinear
                    if not _no_three_collinear(full_set):
                        continue
                    # Check no 4 concyclic
                    if not _no_four_concyclic(full_set):
                        continue
                    
                    found = full_set
                    break
            if found:
                break
        if found:
            return found
    
    return None

def construct():
    start = time.time()
    deadline = start + 100 # 100s safety margin
    
    # Try to find 6 points
    result = _find_6_point_set(deadline)
    
    if result:
        return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in result]}
    
    # Fallback to known 5-point set
    pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
