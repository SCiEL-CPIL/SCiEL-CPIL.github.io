import time
import math
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Find a set of points (x, y*sqrt(D)) with rational x,y, squarefree D,
    such that all pairwise distances are rational, no three collinear,
    no four concyclic. Maximize number of points.
    """
    t0 = time.time()
    deadline = 145.0  # seconds, leave margin

    # Helper: check if a number (Fraction) is a perfect square
    def is_square(frac):
        if frac < 0:
            return False
        p = frac.numerator
        q = frac.denominator
        sp = math.isqrt(p)
        sq = math.isqrt(q)
        return sp * sp == p and sq * sq == q

    # Helper: check if 3 points are collinear
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    # Helper: check if 4 points are concyclic
    def are_concyclic(p1, p2, p3, p4, D):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Subtract row 4 from rows 1, 2, 3 in the determinant matrix
        # Matrix columns: [x^2 + D y^2, x, y, 1]
        # After subtraction, last col is [0,0,0,1]
        # Det = det(3x3) of [x^2 + D y^2 - (x4^2 + D y4^2), x - x4, y - y4]
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

        M3 = [
            [(x1*x1 + D*y1*y1) - (x4*x4 + D*y4*y4), x1 - x4, y1 - y4],
            [(x2*x2 + D*y2*y2) - (x4*x4 + D*y4*y4), x2 - x4, y2 - y4],
            [(x3*x3 + D*y3*y3) - (x4*x4 + D*y4*y4), x3 - x4, y3 - y4]
        ]
        return det3(M3) == 0

    def check_valid(D, points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + D*dy*dy
                if not is_square(d2):
                    return False
        for i, j, k in combinations(range(n), 3):
            if are_collinear(points[i], points[j], points[k]):
                return False
        for i, j, k, l in combinations(range(n), 4):
            if are_concyclic(points[i], points[j], points[k], points[l], D):
                return False
        return True

    best = {'D': 1, 'points': []}
    best_count = 0

    # Known 5-point set for D=1
    base5 = [(Fraction(0), Fraction(0)),
             (Fraction(-56), Fraction(33)),
             (Fraction(-16), Fraction(63)),
             (Fraction(0), Fraction(33)),
             (Fraction(16), Fraction(63))]
    
    if check_valid(1, base5):
        best = {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base5]}
        best_count = 5

    def try_find_extension(D, existing, max_coord, time_limit):
        n = len(existing)
        for x in range(-max_coord, max_coord+1):
            if time.time() > time_limit:
                return None
            for y in range(-max_coord, max_coord+1):
                if x == 0 and y == 0:
                    continue
                new_pt = (Fraction(x), Fraction(y))
                
                # Check if new_pt is same as existing
                if any(new_pt == p for p in existing):
                    continue

                # Check distances
                ok = True
                for ex, ey in existing:
                    dx = new_pt[0] - ex
                    dy = new_pt[1] - ey
                    d2 = dx*dx + D*dy*dy
                    if not is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue

                # Check collinearity
                collinear = False
                for i in range(n):
                    for j in range(i+1, n):
                        if are_collinear(existing[i], existing[j], new_pt):
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue

                # Check concyclicity
                concyclic = False
                if n >= 3:
                    for i, j, k in combinations(range(n), 3):
                        if are_concyclic(existing[i], existing[j], existing[k], new_pt, D):
                            concyclic = True
                            break
                if concyclic:
                    continue
                
                return new_pt
        return None

    # Try extending base5
    if time.time() < deadline - 10:
        ext = try_find_extension(1, base5, 200, deadline - 10)
        if ext is not None:
            new_set = base5 + [ext]
            if check_valid(1, new_set):
                best = {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in new_set]}
                best_count = 6

    return best

# EVOLVE-BLOCK-END
