"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def dist_sq(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def are_collinear(p1, p2, p3):
    # Check if three points are collinear using cross product
    # (p2-p1) x (p3-p1) == 0
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]
    return dx1 * dy2 - dy1 * dx2 == 0

def are_concyclic(p1, p2, p3, p4):
    # Four points are concyclic iff the determinant of the circle equation is zero
    # Using the general circle equation: x^2 + y^2 + Dx + Ey + F = 0
    # For point (x,y): x^2 + y^2 + D*x + E*y + F = 0
    # We can use the determinant:
    # |x^2+y^2  x  y  1|
    # |x^2+y^2  x  y  1| = 0
    # |x^2+y^2  x  y  1|
    # |x^2+y^2  x  y  1|
    def row(p):
        return [p[0]*p[0] + p[1]*p[1], p[0], p[1], 1]
    
    r1 = row(p1)
    r2 = row(p2)
    r3 = row(p3)
    r4 = row(p4)
    
    # Compute 4x4 determinant using cofactor expansion
    def det4(m):
        # m is 4x4
        sign = 1
        total = 0
        for col in range(4):
            minor = []
            for i in range(1, 4):
                minor.append([m[i][j] for j in range(4) if j != col])
            # 3x3 determinant
            def det3(mat):
                return (mat[0][0]*(mat[1][1]*mat[2][2]-mat[1][2]*mat[2][1])
                      - mat[0][1]*(mat[1][0]*mat[2][2]-mat[1][2]*mat[2][0])
                      + mat[0][2]*(mat[1][0]*mat[2][1]-mat[1][1]*mat[2][0]))
            total += sign * m[0][col] * det3(minor)
            sign = -sign
        return total
    
    return det4([r1, r2, r3, r4]) == 0

def construct():
    # Search for 6 integer points with integer pairwise distances
    # No three collinear, no four concyclic
    
    # Generate candidate points in a bounded box
    bound = 150
    candidates = []
    for x in range(-bound, bound + 1):
        for y in range(-bound, bound + 1):
            candidates.append((x, y))
    
    # Build adjacency: edge if squared distance is a perfect square
    # Precompute for efficiency
    n = len(candidates)
    # We'll use a greedy + backtracking approach
    
    # For speed, let's try a randomized greedy approach with multiple restarts
    import random
    random.seed(42)
    
    best_set = None
    
    for attempt in range(500):
        # Shuffle candidates
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        # Greedy: pick points that are "compatible" with all previously picked
        selected = []
        for p in shuffled:
            if len(selected) >= 6:
                break
            # Check if p has integer distance to all selected points
            compatible = True
            for q in selected:
                if not is_perfect_square(dist_sq(p, q)):
                    compatible = False
                    break
            if compatible:
                # Check no three collinear
                collinear_ok = True
                for i in range(len(selected) - 1):
                    for j in range(i + 1, len(selected)):
                        if are_collinear(selected[i], selected[j], p):
                            collinear_ok = False
                            break
                    if not collinear_ok:
                        break
                if not collinear_ok:
                    continue
                # Check no four concyclic
                concyclic_ok = True
                if len(selected) >= 3:
                    for i in range(len(selected) - 2):
                        for j in range(i + 1, len(selected) - 1):
                            for k in range(j + 1, len(selected)):
                                if are_concyclic(selected[i], selected[j], selected[k], p):
                                    concyclic_ok = False
                                    break
                            if not concyclic_ok:
                                break
                        if not concyclic_ok:
                            break
                if concyclic_ok:
                    selected.append(p)
        
        if len(selected) == 6:
            best_set = selected
            break
    
    if best_set is None:
        # Fallback to 5-point configuration
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
    
    points = best_set
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
