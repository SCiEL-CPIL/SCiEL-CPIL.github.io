import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Approach: Constructive Generation + Backtracking Search
    # 1. Define a base Heronian triangle to anchor the coordinate system.
    # 2. Generate a pool of integer points that are at integer distance from all base points.
    # 3. Use backtracking to find the largest subset of the pool (including base points)
    #    that satisfies: no 3 collinear, no 4 concyclic.
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def are_concyclic(p1, p2, p3, p4):
        # Exact rational check for concyclicity
        # Circle equation: x^2 + y^2 + Dx + Ey + F = 0
        # Solve for D, E, F using p1, p2, p3. Check p4.
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # System:
        # D*x1 + E*y1 + F = -(x1^2+y1^2)
        # D*x2 + E*y2 + F = -(x2^2+y2^2)
        # D*x3 + E*y3 + F = -(x3^2+y3^2)
        
        # Subtract 1 from 2 and 3 to eliminate F:
        # D(x2-x1) + E(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
        # D(x3-x1) + E(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
        
        a11 = x2 - x1
        a12 = y2 - y1
        b1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
        
        a21 = x3 - x1
        a22 = y3 - y1
        b2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            return False # Collinear, not concyclic (infinite radius)
            
        # Use Cramer's rule with Fractions for exactness
        D_frac = Fraction(b1 * a22 - a12 * b2, det)
        E_frac = Fraction(a11 * b2 - b1 * a21, det)
        F_frac = Fraction(-(x1*x1 + y1*y1) - D_frac*x1 - E_frac*y1, 1)
        
        # Check p4
        val = x4*x4 + y4*y4 + D_frac*x4 + E_frac*y4 + F_frac
        return val == 0

    def check_general_position(points):
        n = len(points)
        # No 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        # No 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # 1. Base Points: A right triangle with integer sides (Heronian)
    # (0,0), (9,0), (0,12) -> Sides 9, 12, 15.
    # This ensures distances between base points are integers.
    base_points = [(0, 0), (9, 0), (0, 12)]
    
    # 2. Generate Candidates
    # We look for points (x,y) with integer coords such that dist to each base point is integer.
    # Let P1=(0,0), P2=(9,0), P3=(0,12).
    # d1^2 = x^2 + y^2
    # d2^2 = (x-9)^2 + y^2
    # d3^2 = x^2 + (y-12)^2
    
    # d1^2 - d2^2 = x^2 + y^2 - (x^2 - 18x + 81 + y^2) = 18x - 81
    # So 18x = d1^2 - d2^2 + 81.
    # For x to be integer, d1^2 - d2^2 + 81 must be divisible by 18.
    
    # d1^2 - d3^2 = x^2 + y^2 - (x^2 + y^2 - 24y + 144) = 24y - 144
    # So 24y = d1^2 - d3^2 + 144.
    # For y to be integer, d1^2 - d3^2 + 144 must be divisible by 24.
    
    candidates = set()
    # Search range for distances. Max coord ~ 100, so max dist ~ 150.
    # We can iterate d1, d2, d3 up to 200.
    max_d = 200
    for d1 in range(1, max_d + 1):
        d1_sq = d1 * d1
        for d2 in range(1, max_d + 1):
            d2_sq = d2 * d2
            # Check x condition
            num_x = d1_sq - d2_sq + 81
            if num_x % 18 != 0:
                continue
            x = num_x // 18
            
            # Check if x^2 <= d1^2 (so y^2 is non-negative)
            if x * x > d1_sq:
                continue
            y_sq = d1_sq - x * x
            y = math.isqrt(y_sq)
            if y * y != y_sq:
                continue
            
            # Now we have candidate (x,y). Check d3.
            d3_sq = x * x + (y - 12) * (y - 12)
            d3 = math.isqrt(d3_sq)
            if d3 * d3 != d3_sq:
                continue
            
            # Also check if this point is distinct from base points
            if (x, y) in base_points:
                continue
                
            candidates.add((x, y))
            
    # Also add negative coordinates? The equations derived assume specific base points.
    # The symmetry of the triangle (0,0), (9,0), (0,12) is not fully rotational, but we can reflect.
    # However, the derivation $18x = ...$ and $24y = ...$ is specific to this setup.
    # To get more points, we should probably search a larger grid or use a different base.
    # Let's stick to this pool for now and see how many we can pick.
    # If the pool is small, we might need to expand the search or use a different base.
    
    # Let's also generate candidates by scanning a grid, but filter by distance to base points.
    # This is safer to catch points missed by the algebraic derivation if I made a mistake.
    # But the algebraic derivation is exact.
    # Let's expand the range slightly.
    for d1 in range(max_d + 1, 300):
        d1_sq = d1 * d1
        for d2 in range(1, 300):
            d2_sq = d2 * d2
            num_x = d1_sq - d2_sq + 81
            if num_x % 18 != 0:
                continue
            x = num_x // 18
            if x * x > d1_sq:
                continue
            y_sq = d1_sq - x * x
            y = math.isqrt(y_sq)
            if y * y != y_sq:
                continue
            d3_sq = x * x + (y - 12) * (y - 12)
            d3 = math.isqrt(d3_sq)
            if d3 * d3 != d3_sq:
                continue
            if (x, y) in base_points:
                continue
            candidates.add((x, y))

    # Convert to list for search
    cand_list = list(candidates)
    
    # We want to select a subset of cand_list to add to base_points.
    # Total set = base_points + selected.
    # Maximize size.
    
    # Backtracking Search
    best_size = len(base_points)
    best_set = base_points[:]
    
    # Sort candidates by some heuristic? Maybe distance from origin?
    # Or just random order with shuffling for variety.
    random.seed(42)
    random.shuffle(cand_list)
    
    # To make search faster, we can pre-filter candidates that are collinear or concyclic
    # with pairs/triples of base points.
    # But base points are only 3.
    # Any candidate is collinear with 2 base points if it lies on the line through them.
    # Lines:
    # y=0 (through (0,0), (9,0)) -> y must be 0.
    # x=0 (through (0,0), (0,12)) -> x must be 0.
    # Line through (9,0) and (0,12): y - 0 = (12-0)/(0-9) * (x-9) => y = -4/3(x-9) => 3y = -4x + 36 => 4x+3y=36.
    
    filtered_cands = []
    for p in cand_list:
        x, y = p
        if y == 0: continue
        if x == 0: continue
        if 4*x + 3*y == 36: continue
        filtered_cands.append(p)
        
    # Now, we need to check concyclicity with base points.
    # Any 4 points concyclic.
    # If we add 1 point, we have 4 points (3 base + 1 cand).
    # Check if cand is concyclic with any 3 base points.
    # Since base points are (0,0), (9,0), (0,12), they define a unique circle.
    # Center?
    # Midpoint of (0,0)-(9,0) is (4.5, 0). Perp bisector x=4.5.
    # Midpoint of (0,0)-(0,12) is (0, 6). Perp bisector y=6.
    # Center (4.5, 6).
    # Radius squared = 4.5^2 + 6^2 = 20.25 + 36 = 56.25 = 225/4.
    # Circle: (x-4.5)^2 + (y-6)^2 = 56.25
    # x^2 - 9x + 20.25 + y^2 - 12y + 36 = 56.25
    # x^2 + y^2 - 9x - 12y = 0
    # Check if candidate (x,y) satisfies x^2 + y^2 - 9x - 12y == 0.
    
    final_cands = []
    for p in filtered_cands:
        x, y = p
        if x*x + y*y - 9*x - 12*y != 0:
            final_cands.append(p)
            
    # Now we have a pool of candidates that are:
    # 1. Integer distance to all 3 base points.
    # 2. Not collinear with any pair of base points.
    # 3. Not concyclic with the 3 base points.
    
    # We can now greedily add points from this pool, checking constraints among themselves.
    # Since the pool is pre-filtered against the base, we only need to check new-new and new-base (already done).
    # Wait, we checked new-base collinearity and concyclicity.
    # We need to check:
    # - New-New collinearity (with 1 base? No, 3 collinear total. If 2 are new, 1 is base, check. If 3 are new, check.)
    # - New-New concyclicity (with 2 base? No, 4 concyclic total. If 2 new, 2 base, check. If 3 new, 1 base, check. If 4 new, check.)
    
    # This is still a complex constraint satisfaction.
    # Let's use a simple Greedy with Backtracking on this smaller pool.
    
    current_set = base_points[:]
    
    def try_add(p):
        nonlocal current_set, best_size, best_set
        # Check p against all in current_set
        # 1. Collinearity: For any q, r in current_set, are p, q, r collinear?
        #    Since p is not collinear with any pair of base points, we only need to check pairs involving at least one non-base point?
        #    Actually, if q, r are base points, we already filtered that.
        #    So if q, r are both base, skip.
        #    If one is base, one is non-base, check.
        #    If both are non-base, check.
        
        n = len(current_set)
        for i in range(n):
            q = current_set[i]
            # If q is a base point, we don't need to check collinearity with other base points for p
            # because p was filtered against all base pairs.
            # But we DO need to check if p, q, r are collinear where r is another point.
            
            for j in range(n):
                if i == j: continue
                r = current_set[j]
                # Optimization: If q and r are both base points, skip (already handled)
                # How to know if base? Check if in base_points list.
                if q in base_points and r in base_points:
                    continue
                if are_collinear(p, q, r):
                    return False
        
        # 2. Concyclicity: For any q, r, s in current_set, are p, q, r, s concyclic?
        #    If q, r, s are all base points, skip (already handled).
        #    Otherwise, check.
        
        for i in range(n):
            q = current_set[i]
            for j in range(i+1, n):
                r = current_set[j]
                for k in range(j+1, n):
                    s = current_set[k]
                    if q in base_points and r in base_points and s in base_points:
                        continue
                    if are_concyclic(p, q, r, s):
                        return False
        return True

    # Greedy add
    for p in final_cands:
        if try_add(p):
            current_set.append(p)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                
    # If greedy gets stuck, we might miss larger sets.
    # But given the pre-filtering, the remaining constraints are only between the added points.
    # This is much faster.
    
    # Format output
    points_out = []
    for x, y in best_set:
        points_out.append([[x, 1], [y, 1]])
        
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
