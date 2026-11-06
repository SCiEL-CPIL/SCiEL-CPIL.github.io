"""Integral point sets in general position."""
import math
import time
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Known 6-point integral point set (Erdős–Moser type)
    # Points on a line at y=0 and y=3:
    # (0,0), (3,0), (4,0), (12,0) on bottom
    # (0,3), (4,3), (12,3) on top - but that's 7 points, check general position
    # Actually let me use a verified set.
    
    # Strategy: try known constructions and verify
    best_points = None
    
    # Known 6-point integral point set
    # From literature: (0,0), (4,0), (9,0), (0,3), (4,3), (9,3) - but 6 points on 2 lines
    # Need no three collinear.
    
    # Let me try to find one by search.
    # Points with integer coordinates, pairwise integer distances.
    
    # Try: (0,0), (4,0), (9,0), (1,4), (6,4), (10,4)
    # Check distances...
    
    # Actually, let me use a systematic approach.
    # Start with a known 5-point set and try to add a 6th point.
    
    base5 = [(0,0), (-56,33), (-16,63), (0,33), (16,63)]
    
    # Try adding a 6th point near these
    # For a new point (x,y), distance to each existing point must be integer.
    
    # Let me try a different approach: search for 6-point sets directly.
    # Points on two parallel lines with integer coordinates.
    
    # Known construction: points on y=0 and y=h
    # For h=3: (0,0),(4,0),(9,0) and (0,3),(4,3),(9,3)
    # Distances: within same line: 4,5,9; between lines: 3, sqrt(16+9)=5, sqrt(81+9)=sqrt(90) not integer
    # So that doesn't work.
    
    # Let me try h=4:
    # (0,0),(3,0),(4,0) and (0,4),(3,4),(4,4)
    # Between: 4, 5, 5 - all integers! But 3+3=6 points.
    # Check no three collinear: (0,0),(3,0),(4,0) collinear! Bad.
    
    # Need to break collinearity. Use non-parallel lines or different offsets.
    
    # Try: (0,0), (3,0), (4,0) on y=0, and (1,4), (5,4), (7,4) on y=4
    # Within y=0: 3,4,1 - integers
    # Within y=4: 4,6,2 - integers  
    # Between: (0,0)-(1,4)=5, (0,0)-(5,4)=sqrt(41) no.
    
    # Let me be more systematic. For two lines y=0 and y=h:
    # Points (a,0),(b,0),(c,0) and (d,h),(e,h),(f,h)
    # Need |a-b|,|a-c|,|b-c| integers (they are)
    # Need |d-e|,|d-f|,|e-f| integers (they are)
    # Need sqrt((a-d)^2+h^2), sqrt((a-e)^2+h^2), sqrt((a-f)^2+h^2) integers
    # and similarly for b,c.
    
    # For h=3: need (dx)^2+9 = square, so dx^2 = k^2-9 = (k-3)(k+3)
    # k=5: dx^2=16, dx=4
    # k=7: dx^2=40 no
    # k=13: dx^2=160 no
    # So for h=3, only dx=4 works (and dx=-4).
    
    # For h=4: dx^2 = k^2-16 = (k-4)(k+4)
    # k=5: dx^2=9, dx=3
    # k=6: dx^2=20 no
    # k=8: dx^2=48 no
    # k=12: dx^2=128 no
    # So for h=4, dx=3.
    
    # For h=5: dx^2 = k^2-25
    # k=13: dx^2=144, dx=12
    # k=5: dx=0
    # So for h=5, dx=12.
    
    # For h=12: dx^2 = k^2-144
    # k=13: dx^2=25, dx=5
    # k=15: dx^2=81, dx=9
    # k=17: dx^2=160 no
    # k=20: dx^2=256, dx=16
    # So for h=12, dx in {5,9,16}.
    
    # Let me try h=12 with dx values {5,9,16}.
    # Points on y=0: x1,x2,x3
    # Points on y=12: x1+5, x2+9, x3+16 (or other combinations)
    
    # Try: (0,0), (5,0), (9,0) on y=0
    #       (5,12), (14,12), (25,12) on y=12
    # Check all cross distances:
    # (0,0)-(5,12): sqrt(25+144)=13 ✓
    # (0,0)-(14,12): sqrt(196+144)=sqrt(340) ✗
    
    # Need to be more careful. Let me just brute force small sets.
    
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
            # Four points concyclic iff the determinant is zero
            # |x1 y1 x1^2+y1^2 1|
            # |x2 y2 x2^2+y2^2 1|
            # |x3 y3 x3^2+y3^2 1| = 0
            # |x4 y4 x4^2+y4^2 1|
            A = [[x1, y1, x1*x1+y1*y1, 1],
                 [x2, y2, x2*x2+y2*y2, 1],
                 [x3, y3, x3*x3+y3*y3, 1],
                 [x4, y4, x4*x4+y4*y4, 1]]
            # Compute determinant
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
        # All pairwise distances integer
        for i,j in combinations(range(len(pts)), 2):
            if not _is_integer_dist(pts[i], pts[j]):
                return False
        if not _no_three_collinear(pts):
            return False
        if len(pts) >= 4 and not _no_four_concyclic(pts):
            return False
        return True
    
    # Try known 6-point integral point set
    # From literature search: (0,0), (4,0), (9,0), (1,4), (5,4), (10,4)
    # Let me verify
    test6a = [(0,0), (4,0), (9,0), (1,4), (5,4), (10,4)]
    if _valid_set(test6a):
        best_points = test6a
    
    # Try other combinations
    # (0,0), (3,0), (4,0), (1,4), (4,4), (8,4)
    test6b = [(0,0), (3,0), (4,0), (1,4), (4,4), (8,4)]
    if _valid_set(test6b):
        best_points = test6b
    
    # Let me try a broader search for 6-point sets
    # Points in a bounded region
    if time.time() < deadline - 20:
        # Search for 6-point sets with coordinates in [-10, 20]
        found = False
        # Try points on two horizontal lines
        for h in [3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20]:
            if time.time() > deadline - 20:
                break
            # Find all x values where (x,0) and (x',h) have integer distance
            # For each x on bottom, find valid x' on top
            valid_bottom = list(range(-10, 25))
            valid_top = list(range(-10, 25))
            
            # Precompute: for each bottom x, which top x' give integer distance
            bottom_to_top = {}
            for bx in valid_bottom:
                good_tops = []
                for tx in valid_top:
                    if _is_integer_dist((bx,0), (tx,h)):
                        good_tops.append(tx)
                bottom_to_top[bx] = good_tops
            
            # Try to find 3 bottom and 3 top points
            for bi in combinations(valid_bottom, 3):
                if time.time() > deadline - 20:
                    break
                b1,b2,b3 = bi
                # Check bottom distances
                if not (_is_integer_dist((b1,0),(b2,0)) and _is_integer_dist((b1,0),(b3,0)) and _is_integer_dist((b2,0),(b3,0))):
                    continue
                # Check collinearity of bottom (they're all on y=0, so always collinear!)
                # So we can't have 3 on the same line.
                # This approach won't work for 3+3 on parallel lines.
                break
            
            # So we need a different structure. Let me try 2 on one line, 2 on another, 2 elsewhere.
            # Or just do a general search.
        
        # General search: try to extend the known 5-point set
        # Add a 6th point (x,y) such that dist to all 5 is integer
        if time.time() < deadline - 20:
            for y in range(-100, 101):
                if time.time() > deadline - 20:
                    break
                for x in range(-100, 101):
                    if time.time() > deadline - 20:
                        break
                    new_pt = (x, y)
                    if all(_is_integer_dist(new_pt, p) for p in base5):
                        candidate = base5 + [new_pt]
                        if _valid_set(candidate):
                            best_points = candidate
                            break
                if best_points is not None:
                    break
    
    # If we found a 6-point set, use it. Otherwise use the 5-point set.
    if best_points is None:
        best_points = base5
    
    # Verify the best set
    assert _valid_set(best_points), "Final set failed validation!"
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
