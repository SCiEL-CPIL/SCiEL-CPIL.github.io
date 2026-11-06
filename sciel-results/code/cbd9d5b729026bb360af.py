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
    # Key: avoid 3 collinear points!
    candidates = [
        # Non-collinear 6-point configurations
        [(0,0), (4,0), (1,3), (5,3), (2,6), (8,6)],
        [(0,0), (3,0), (0,4), (3,4), (1,8), (4,8)],
        [(0,0), (5,0), (3,4), (8,4), (1,8), (6,8)],
        [(0,0), (4,0), (0,3), (4,3), (1,6), (5,6)],
        [(0,0), (6,0), (2,3), (8,3), (3,6), (9,6)],
        # Try hexagonal-like arrangements
        [(0,0), (4,0), (2,3), (6,3), (4,6), (8,6)],
        [(0,0), (3,0), (1,2), (5,2), (3,4), (7,4)],
        # More spread out
        [(0,0), (5,0), (2,4), (8,4), (4,8), (10,8)],
        [(0,0), (4,0), (1,3), (4,3), (7,3), (2,6)],
        [(0,0), (6,0), (1,4), (7,4), (3,8), (9,8)],
    ]
    
    best_points = None
    for cand in candidates:
        if _valid_set(cand):
            best_points = cand
            break

    # If no 6-point set found, try systematic search
    if best_points is None and time.time() < deadline - 20:
        # Efficient search: build up from valid 3-point bases
        # Strategy: find all points at integer distance from a base set,
        # then greedily add points that maintain general position
        
        def _find_points_at_int_dist_from(base_pts, max_coord=30):
            """Find all integer points at integer distance from all points in base_pts"""
            candidates = set()
            for x in range(-max_coord, max_coord+1):
                for y in range(-max_coord, max_coord+1):
                    pt = (x, y)
                    if pt in base_pts:
                        continue
                    ok = True
                    for bp in base_pts:
                        dx = x - bp[0]
                        dy = y - bp[1]
                        d2 = dx*dx + dy*dy
                        r = int(math.isqrt(d2))
                        if r*r != d2:
                            ok = False
                            break
                    if ok:
                        candidates.add(pt)
            return list(candidates)
        
        # Try various base triangles (3 points with integer pairwise distances)
        base_triangles = [
            [(0,0), (4,0), (1,3)],
            [(0,0), (3,0), (0,4)],
            [(0,0), (5,0), (3,4)],
            [(0,0), (4,0), (0,3)],
            [(0,0), (6,0), (2,3)],
            [(0,0), (5,0), (0,12)],
            [(0,0), (9,0), (0,12)],
            [(0,0), (8,0), (3,4)],
            [(0,0), (7,0), (4,3)],
            [(0,0), (10,0), (6,8)],
        ]
        
        for tri in base_triangles:
            if time.time() > deadline - 20:
                break
            # Verify triangle has integer distances
            if not (_is_integer_dist(tri[0], tri[1]) and 
                    _is_integer_dist(tri[0], tri[2]) and 
                    _is_integer_dist(tri[1], tri[2])):
                continue
            
            # Find all points at integer distance from all 3 base points
            cand_pts = _find_points_at_int_dist_from(tri, max_coord=25)
            
            # Sort candidates by some heuristic (prefer non-collinear additions)
            def _collinearity_score(pt, base):
                """Lower score = better (less collinear with base points)"""
                score = 0
                for i in range(len(base)):
                    for j in range(i+1, len(base)):
                        x1,y1 = base[i]; x2,y2 = base[j]; x3,y3 = pt
                        if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                            score += 1
                return score
            
            cand_pts.sort(key=lambda p: _collinearity_score(p, tri))
            
            # Try adding points greedily
            best_set = list(tri)
            for pt in cand_pts:
                if time.time() > deadline - 20:
                    break
                new_set = best_set + [pt]
                if len(new_set) > 6:
                    continue
                if _valid_set(new_set):
                    best_set = new_set
            
            if len(best_set) > (len(best_points) if best_points else 0):
                best_points = best_set
            
            if best_points and len(best_points) >= 6:
                break

    # Fallback: 5-point set
    if best_points is None or len(best_points) < 5:
        # Verify base5
        base5 = [(0,0), (-56,33), (-16,63), (0,33), (16,63)]
        if _valid_set(base5):
            best_points = base5
        else:
            # Simple fallback
            best_points = [(0,0), (4,0), (1,3)]
    
    # Try to extend a 5-point set to 6 points using remaining time
    if best_points and len(best_points) == 5 and time.time() < deadline - 10:
        base5 = best_points
        cand_pts = _find_points_at_int_dist_from(base5, max_coord=30)
        for pt in cand_pts:
            if time.time() > deadline - 10:
                break
            new_set = base5 + [pt]
            if _valid_set(new_set):
                best_points = new_set
                break

    assert _valid_set(best_points), "Final set failed validation!"
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
