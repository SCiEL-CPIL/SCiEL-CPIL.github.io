"""Rewritten program: construct a larger integral-point set in general position."""
from fractions import Fraction
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Return a set of points (x, y*sqrt(D)) with rational coordinates such that:
    - All pairwise distances are rational.
    - No three points are collinear.
    - No four points are concyclic.
    - D is squarefree, D <= 10^12.
    - |numerators|, denominators <= 10^40.
    Maximize the number of points.
    
    Strategy: Use D=1 (ordinary integer lattice) and search for a large set of
    integer points with all pairwise distances integer, no three collinear, no four
    concyclic. We use a greedy construction with backtracking/local search.
    """
    
    def dist_sq(p, q):
        return (p[0] - q[0])**2 + (p[1] - q[1])**2
    
    def is_integer_dist(p, q):
        d2 = dist_sq(p, q)
        r = isqrt(d2)
        return r * r == d2
    
    def are_collinear(p, q, r):
        # Cross product
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) == 0
    
    def are_concyclic(p, q, r, s):
        # Four points are concyclic iff the determinant is zero
        # |x^2+y^2, x, y, 1| for each point
        def row(pt):
            x, y = pt
            return [x*x + y*y, x, y, 1]
        M = [row(p), row(q), row(r), row(s)]
        # Compute determinant via Gaussian elimination over rationals
        # Use exact arithmetic with Fraction
        n = 4
        M = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
        det = Fraction(1)
        for i in range(n):
            # Find pivot
            pivot_row = -1
            for r in range(i, n):
                if M[r][i] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                return True  # determinant is 0
            if pivot_row != i:
                M[i], M[pivot_row] = M[pivot_row], M[i]
                det = -det
            det *= M[i][i]
            # Eliminate below
            for r in range(i+1, n):
                if M[r][i] != 0:
                    factor = M[r][i] / M[i][i]
                    for c in range(i, n):
                        M[r][c] -= factor * M[i][c]
        return det == 0
    
    def valid_set(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if not is_integer_dist(points[i], points[j]):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True
    
    # Known 5-point integral set in general position
    # This set is (0,0), (56,33), (16,63), (-16,63), (-56,33) ? 
    # Let's use the one from the seed but verify it.
    base_sets = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (-16, 63), (-56, 33)],
        # Another known 5-point set?
        # (0,0), (3,4), (0,12), (-3,4), (9,12) ? No, need to check general position.
    ]
    
    best_global = []
    
    def try_add_point(points, candidate):
        """Check if candidate can be added to points."""
        # Check integer distance to all existing
        for p in points:
            if not is_integer_dist(p, candidate):
                return False
        # Check no three collinear
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if are_collinear(points[i], points[j], candidate):
                    return False
        # Check no four concyclic
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    if are_concyclic(points[i], points[j], points[k], candidate):
                        return False
        return True
    
    def find_candidates_from_pair(p, q, limit=300):
        """Find lattice points at integer distance from both p and q, within limit."""
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        d2 = dx*dx + dy*dy
        d = isqrt(d2)
        if d * d != d2:
            return []
        
        candidates = []
        mid_x = (p[0] + q[0]) // 2
        mid_y = (p[1] + q[1]) // 2
        
        for x in range(mid_x - limit, mid_x + limit + 1):
            for y in range(mid_y - limit, mid_y + limit + 1):
                cand = (x, y)
                if cand == p or cand == q:
                    continue
                if is_integer_dist(p, cand) and is_integer_dist(q, cand):
                    candidates.append(cand)
        return candidates

    for base in base_sets:
        if not valid_set(base):
            continue
            
        current_best = list(base)
        
        # Try to add points iteratively
        # We will try to add up to 2 more points
        for _ in range(2):
            added = False
            # Generate candidates from all pairs
            all_cands = set()
            for i in range(len(current_best)):
                for j in range(i+1, len(current_best)):
                    cands = find_candidates_from_pair(current_best[i], current_best[j], limit=200)
                    all_cands.update(cands)
            
            # Sort candidates by some heuristic? e.g., distance to centroid
            # Or just iterate
            for cand in all_cands:
                if cand in current_best:
                    continue
                if try_add_point(current_best, cand):
                    current_best.append(cand)
                    added = True
                    break
            
            if not added:
                break
        
        if len(current_best) > len(best_global):
            best_global = current_best
    
    # If no improvement found, use the best base
    if not best_global:
        best_global = list(base_sets[0])
    
    # Final verification
    if not valid_set(best_global):
        best_global = list(base_sets[0])
        
    points_out = [[[x, 1], [y, 1]] for x, y in best_global]
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
