"""Integral point sets in general position."""
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def _is_integer_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = int(math.isqrt(d2))
        return r*r == d2

    def _no_three_collinear(pts):
        for i,j,k in combinations(range(len(pts)), 3):
            x1,y1 = pts[i]; x2,y2 = pts[j]; x3,y3 = pts[k]
            if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                return False
        return True

    def _no_four_concyclic(pts):
        for i,j,k,l in combinations(range(len(pts)), 4):
            x1,y1 = pts[i]; x2,y2 = pts[j]; x3,y3 = pts[k]; x4,y4 = pts[l]
            A = [[x1, y1, x1*x1+y1*y1, 1],
                 [x2, y2, x2*x2+y2*y2, 1],
                 [x3, y3, x3*x3+y3*y3, 1],
                 [x4, y4, x4*x4+y4*y4, 1]]
            det = (A[0][0]*(A[1][1]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) + A[1][3]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]))
                 - A[0][1]*(A[1][0]*(A[2][2]*A[3][3]-A[2][3]*A[3][2]) - A[1][2]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]))
                 + A[0][2]*(A[1][0]*(A[2][1]*A[3][3]-A[2][3]*A[3][1]) - A[1][1]*(A[2][0]*A[3][3]-A[2][3]*A[3][0]) + A[1][3]*(A[2][0]*A[3][1]-A[2][1]*A[3][0]))
                 - A[0][3]*(A[1][0]*(A[2][1]*A[3][2]-A[2][2]*A[3][1]) - A[1][1]*(A[2][0]*A[3][2]-A[2][2]*A[3][0]) + A[1][2]*(A[2][0]*A[3][1]-A[2][1]*A[3][0])))
            if det == 0:
                return False
        return True

    def _valid_set(pts):
        if len(pts) < 3:
            return False
        for i,j in combinations(range(len(pts)), 2):
            if not _is_integer_dist(pts[i], pts[j]):
                return False
        if not _no_three_collinear(pts):
            return False
        if len(pts) >= 4 and not _no_four_concyclic(pts):
            return False
        return True

    # Try several known/candidate 6-point sets
    candidates = [
        [(0,0), (4,0), (9,0), (1,4), (5,4), (10,4)],
        [(0,0), (3,0), (4,0), (1,4), (4,4), (8,4)],
        [(0,0), (5,0), (9,0), (0,12), (5,12), (9,12)],
        [(0,0), (4,0), (5,0), (0,3), (4,3), (5,3)],
        [(0,0), (6,0), (8,0), (2,3), (5,3), (9,3)],
        [(0,0), (3,0), (6,0), (0,4), (3,4), (6,4)],
        [(0,0), (4,0), (5,0), (1,3), (4,3), (8,3)],
        [(0,0), (5,0), (10,0), (0,12), (5,12), (10,12)],
        [(0,0), (6,0), (10,0), (2,3), (6,3), (10,3)],
        [(0,0), (4,0), (9,0), (0,3), (4,3), (9,3)],
    ]
    
    best_points = None
    for cand in candidates:
        if _valid_set(cand):
            best_points = cand
            break

    # If no 6-point set found, try systematic search
    if best_points is None and time.time() < deadline - 30:
        # Search: 2 points on y=0, 2 on y=h, 2 on y=2h or other
        # Or: try to find 6 points in a grid
        # Use the approach: pick a base triangle, then find points at integer distance from all 3
        # Base: (0,0), (a,0), (b,c) with integer distances
        # Then find (x,y) with integer distance to all 3
        
        # Try small base triangles
        base_triangles = [
            [(0,0), (4,0), (1,3)],
            [(0,0), (3,0), (0,4)],
            [(0,0), (5,0), (3,4)],
            [(0,0), (4,0), (0,3)],
            [(0,0), (6,0), (2,3)],
            [(0,0), (5,0), (0,12)],
            [(0,0), (9,0), (0,12)],
        ]
        
        for tri in base_triangles:
            if time.time() > deadline - 30:
                break
            # Verify triangle
            if not (_is_integer_dist(tri[0], tri[1]) and 
                    _is_integer_dist(tri[0], tri[2]) and 
                    _is_integer_dist(tri[1], tri[2])):
                continue
            # Find points at integer distance to all 3
            # For point (x,y): 
            # x^2+y^2 = d1^2
            # (x-a)^2+y^2 = d2^2
            # (x-b)^2+(y-c)^2 = d3^2
            # Subtract first from second: -2ax+a^2 = d2^2-d1^2 => x = (d2^2-d1^2-a^2)/(-2a) if a!=0
            a, b, c = tri[1][0], tri[2][0], tri[2][1]
            if a == 0:
                continue
            # x is determined by d1, d2: x = (d2^2 - d1^2 - a^2) / (2*a) ... wait
            # (x-a)^2 + y^2 - (x^2+y^2) = d2^2 - d1^2
            # x^2-2ax+a^2 - x^2 = d2^2-d1^2
            # -2ax + a^2 = d2^2-d1^2
            # x = (a^2 - d2^2 + d1^2) / (2a)
            # Similarly from tri[0] and tri[2]:
            # (x-b)^2+(y-c)^2 - (x^2+y^2) = d3^2-d1^2
            # x^2-2bx+b^2+y^2-2cy+c^2 - x^2-y^2 = d3^2-d1^2
            # -2bx+b^2-2cy+c^2 = d3^2-d1^2
            # 2bx+2cy = b^2+c^2-d3^2+d1^2
            # So we have x from first eq, then y from second.
            
            # Search over d1, d2, d3
            for d1 in range(1, 50):
                if time.time() > deadline - 30:
                    break
                for d2 in range(1, 50):
                    if time.time() > deadline - 30:
                        break
                    num = a*a - d2*d2 + d1*d1
                    if num % (2*a) != 0:
                        continue
                    x = num // (2*a)
                    y2 = d1*d1 - x*x
                    if y2 < 0:
                        continue
                    y = int(math.isqrt(y2))
                    if y*y != y2:
                        continue
                    # Check distance to tri[2]
                    d3_sq = (x-b)**2 + (y-c)**2
                    d3 = int(math.isqrt(d3_sq))
                    if d3*d3 != d3_sq:
                        continue
                    pt = (x, y)
                    if pt in tri:
                        continue
                    # Now try to add a 6th point
                    base4 = tri + [pt]
                    if not _valid_set(base4):
                        continue
                    # Find 6th point
                    for d1b in range(1, 50):
                        if time.time() > deadline - 30:
                            break
                        for d2b in range(1, 50):
                            if time.time() > deadline - 30:
                                break
                            numb = a*a - d2b*d2b + d1b*d1b
                            if numb % (2*a) != 0:
                                continue
                            xb = numb // (2*a)
                            y2b = d1b*d1b - xb*xb
                            if y2b < 0:
                                continue
                            yb = int(math.isqrt(y2b))
                            if yb*yb != y2b:
                                continue
                            d3b_sq = (xb-b)**2 + (yb-c)**2
                            d3b = int(math.isqrt(d3b_sq))
                            if d3b*d3b != d3b_sq:
                                continue
                            ptb = (xb, yb)
                            if ptb in base4:
                                continue
                            # Check distance to pt
                            if not _is_integer_dist(ptb, pt):
                                continue
                            cand6 = base4 + [ptb]
                            if _valid_set(cand6):
                                best_points = cand6
                                break
                        if best_points is not None:
                            break
                    if best_points is not None:
                        break
                if best_points is not None:
                    break
            if best_points is not None:
                break

    # Fallback: 5-point set
    if best_points is None:
        # Verify base5
        base5 = [(0,0), (-56,33), (-16,63), (0,33), (16,63)]
        if _valid_set(base5):
            best_points = base5
        else:
            # Simple fallback
            best_points = [(0,0), (4,0), (1,3)]

    assert _valid_set(best_points), "Final set failed validation!"
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
