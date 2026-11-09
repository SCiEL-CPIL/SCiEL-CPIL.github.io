"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_valid(points, D):
    """Check all constraints: distinct, no 3 collinear, no 4 concyclic, all pairwise distances rational."""
    n = len(points)
    if n < 3:
        return False
    
    # Check distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    
    # Check pairwise distances are rational (for D=1, integer distances)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product for collinearity
                cross = (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - \
                        (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                if cross == 0:
                    return False
    
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points are concyclic iff the determinant is zero
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    
                    # Circle equation: x^2 + y^2 + Ax + By + C = 0
                    # Solve for A, B, C using first 3 points, check 4th
                    # Using the determinant form:
                    # |x1^2+y1^2, x1, y1, 1|
                    # |x2^2+y2^2, x2, y2, 1| = 0
                    # |x3^2+y3^2, x3, y3, 1|
                    # |x4^2+y4^2, x4, y4, 1|
                    
                    a1 = x1*x1 + y1*y1
                    a2 = x2*x2 + y2*y2
                    a3 = x3*x3 + y3*y3
                    a4 = x4*x4 + y4*y4
                    
                    # Compute 4x4 determinant
                    # Using cofactor expansion along last column
                    det = (a1 * (x2*(y3*1 - y4*1) - y2*(x3*1 - x4*1) + 1*(x3*y4 - x4*y3)) -
                           x1 * (a2*(y3*1 - y4*1) - y2*(a3*1 - a4*1) + 1*(a3*y4 - a4*y3)) +
                           y1 * (a2*(x3*1 - x4*1) - x2*(a3*1 - a4*1) + 1*(a3*x4 - a4*x3)) -
                           1 * (a2*(x3*y4 - x4*y3) - y2*(x3*a4 - x4*a3) + y3*(x3*a4 - x4*a3)))
                    # Let me use a cleaner determinant computation
                    
                    # Actually, let me compute it properly
                    # det of matrix:
                    # [a1, x1, y1, 1]
                    # [a2, x2, y2, 1]
                    # [a3, x3, y3, 1]
                    # [a4, x4, y4, 1]
                    
                    # Subtract row1 from rows 2,3,4
                    b2 = a2 - a1
                    c2 = x2 - x1
                    d2_ = y2 - y1
                    b3 = a3 - a1
                    c3 = x3 - x1
                    d3_ = y3 - y1
                    b4 = a4 - a1
                    c4 = x4 - x1
                    d4_ = y4 - y1
                    
                    # Now det = det of:
                    # [a1, x1, y1, 1]
                    # [b2, c2, d2_, 0]
                    # [b3, c3, d3_, 0]
                    # [b4, c4, d4_, 0]
                    # Expand along last column:
                    # det = 1 * det of 3x3:
                    # [b2, c2, d2_]
                    # [b3, c3, d3_]
                    # [b4, c4, d4_]
                    
                    det3 = b2 * (c3 * d4_ - d3_ * c4) - c2 * (b3 * d4_ - d3_ * b4) + d2_ * (b3 * c4 - c3 * b4)
                    if det3 == 0:
                        return False
    
    return True

def _search_incremental(max_points, deadline, initial_points=None):
    """Search for larger integral point sets by incremental addition."""
    if initial_points is None:
        initial_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best = list(initial_points)
    best_size = len(best)
    
    # Precompute valid distances (integer distances up to some bound)
    # For a new point (x,y) to be at integer distance from existing points,
    # we need sqrt((x-xi)^2 + (y-yi)^2) to be integer
    
    # Search in a bounded region around existing points
    x_min = min(p[0] for p in best) - 100
    x_max = max(p[0] for p in best) + 100
    y_min = min(p[1] for p in best) - 100
    y_max = max(p[1] for p in best) + 100
    
    # Try to add points one at a time
    for target_size in range(best_size + 1, max_points + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Try positions in a grid
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                
                # Quick check: distance to all existing points must be integer
                valid = True
                for px, py in best:
                    dx = x - px
                    dy = y - py
                    d2 = dx * dx + dy * dy
                    if not _is_square(d2):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check distinctness
                if (x, y) in best:
                    continue
                
                # Check no three collinear with new point
                collinear = False
                for i in range(len(best)):
                    for j in range(i + 1, len(best)):
                        cross = (best[j][0] - best[i][0]) * (y - best[i][1]) - \
                                (best[j][1] - best[i][1]) * (x - best[i][0])
                        if cross == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check no four concyclic with new point
                concyclic = False
                new_list = best + [(x, y)]
                n = len(new_list)
                for i in range(n):
                    if i == n - 1:
                        continue  # Skip if new point is in the triple, handled below
                    for j in range(i + 1, n):
                        if j == n - 1:
                            continue
                        for k in range(j + 1, n):
                            if k == n - 1:
                                continue
                            # Check if points i, j, k, n-1 are concyclic
                            x1, y1 = new_list[i]
                            x2, y2 = new_list[j]
                            x3, y3 = new_list[k]
                            x4, y4 = new_list[n - 1]
                            
                            a1 = x1*x1 + y1*y1
                            a2 = x2*x2 + y2*y2
                            a3 = x3*x3 + y3*y3
                            a4 = x4*x4 + y4*y4
                            
                            b2 = a2 - a1
                            c2 = x2 - x1
                            d2_ = y2 - y1
                            b3 = a3 - a1
                            c3 = x3 - x1
                            d3_ = y3 - y1
                            b4 = a4 - a1
                            c4 = x4 - x1
                            d4_ = y4 - y1
                            
                            det3 = b2 * (c3 * d4_ - d3_ * c4) - c2 * (b3 * d4_ - d3_ * b4) + d2_ * (b3 * c4 - c3 * b4)
                            if det3 == 0:
                                concyclic = True
                                break
                        if concyclic:
                            break
                    if concyclic:
                        break
                if concyclic:
                    continue
                
                # Valid! Add it
                best.append((x, y))
                found = True
                best_size = target_size
                break
            if found:
                break
        
        if not found:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with known 5-point set
    initial = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify initial set
    if not _check_valid(initial, 1):
        # Fallback to a simpler valid set
        initial = [(0, 0), (3, 4), (0, 5), (4, 0), (5, 0)]
    
    # Try to grow
    best = _search_incremental(12, deadline, initial)
    
    # Final validation
    if not _check_valid(best, 1):
        best = initial
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
