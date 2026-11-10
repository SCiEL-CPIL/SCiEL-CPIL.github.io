import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Helper functions
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def check_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        return is_square(d2)

    def check_collinear(p1, p2, p3):
        # Cross product
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

    def check_concyclic(p1, p2, p3, p4):
        # Determinant of the matrix:
        # | x1 y1 x1^2+y1^2 1 |
        # | x2 y2 x2^2+y2^2 1 |
        # | x3 y3 x3^2+y3^2 1 |
        # | x4 y4 x4^2+y4^2 1 |
        # is zero.
        # We can compute this using fractions or integers.
        # Since coords are integers, we can use integer arithmetic.
        # Let's use the property that 4 points are concyclic iff the determinant is 0.
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        # Subtract row 4 from rows 1,2,3 to simplify
        # | x1-x4 y1-y4 (x1^2-x4^2)+(y1^2-y4^2) 0 |
        # | x2-x4 y2-y4 (x2^2-x4^2)+(y2^2-y4^2) 0 |
        # | x3-x4 y3-y4 (x3^2-x4^2)+(y3^2-y4^2) 0 |
        # | x4   y4   x4^2+y4^2                 1 |
        
        # Determinant is 1 * det of 3x3 matrix
        a1 = x1 - x4
        b1 = y1 - y4
        c1 = (x1*x1 + y1*y1) - (x4*x4 + y4*y4)
        
        a2 = x2 - x4
        b2 = y2 - y4
        c2 = (x2*x2 + y2*y2) - (x4*x4 + y4*y4)
        
        a3 = x3 - x4
        b3 = y3 - y4
        c3 = (x3*x3 + y3*y3) - (x4*x4 + y4*y4)
        
        det = a1 * (b2*c3 - c2*b3) - b1 * (a2*c3 - c2*a3) + c1 * (a2*b3 - b2*a3)
        return det == 0

    def is_valid_set(points):
        n = len(points)
        # Check distinctness
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                if not check_dist(points[i], points[j]):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for combo in combinations(range(n), 4):
            if check_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]):
                return False
        return True

    # Initial set from seed
    current_set = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify initial set (should be valid)
    if not is_valid_set(current_set):
        # Fallback to a smaller known valid set if seed is broken (unlikely)
        current_set = [(0,0), (3,0), (0,4)]
        if not is_valid_set(current_set):
             return {'D': 1, 'points': [[[0,1],[0,1]]]} # Invalid fallback, but shouldn't happen

    best_set = current_set[:]

    # Try to add a 6th point
    # Generate candidates in a grid around the existing points
    # Existing x range: -56 to 16, y range: 0 to 63
    # Expand range
    
    # To speed up, we can pre-filter candidates that have integer distance to at least one point in current_set
    # But checking all pairs is necessary for the final set.
    
    # We will iterate through a grid of potential new points.
    # Range: x in [-100, 100], y in [-50, 100]
    
    # Optimization: Only consider points that form integer distance with all points in current_set?
    # That's too restrictive. We want to find a point P such that {current_set + P} is valid.
    # So P must have integer distance to all points in current_set.
    
    # Let's find all integer points P in a reasonable range that have integer distance to all points in current_set.
    # Then check if adding P breaks concyclicity/collinearity.
    
    # Since current_set has 5 points, P must be in the intersection of 5 "distance loci".
    # This is a system of quadratic equations.
    # (x-xi)^2 + (y-yi)^2 = di^2
    # Subtracting equations gives linear equations.
    
    # Let's just do a grid search for P.
    # To make it faster, we can limit the search space.
    
    # If we don't find a 6th point, we return the 5-point set.
    # Is it possible to find 6?
    # Known results: Max number of points in general position with rational distances is 5 for D=1?
    # Actually, there are known sets of 5. 6 is open or very difficult.
    # However, the task says "8 points would be new", implying 5-7 are possible or expected to be searched for.
    # I will stick to D=1 and try hard to find a 6th.
    
    # Grid search for 6th point
    # We can prune the grid by checking distance to the first point (0,0).
    # If dist(P, (0,0)) is not integer, skip.
    
    x_min, x_max = -100, 100
    y_min, y_max = -50, 100
    
    # To be efficient, iterate over possible squared distances from (0,0)
    # d2 = x^2 + y^2 must be a square.
    
    # We'll just loop x, y.
    # 200 * 150 = 30,000 candidates.
    # For each candidate, check distance to 5 points.
    # 30,000 * 5 = 150,000 checks. Very fast.
    # Then check collinearity/concyclicity with the new set.
    # 150,000 * (5 collinear checks + 5 concyclic checks) ~ 1,500,000 ops.
    # This will finish in < 1 second.
    
    found_six = False
    for x in range(x_min, x_max + 1):
        if time.time() > deadline - 5.0:
            break
        for y in range(y_min, y_max + 1):
            if time.time() > deadline - 5.0:
                break
            p = (x, y)
            if p in current_set:
                continue
            
            # Quick check: distance to all points in current_set must be integer
            valid_dist = True
            for cp in current_set:
                if not check_dist(p, cp):
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Check if adding p creates collinearity
            valid_geo = True
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    if check_collinear(current_set[i], current_set[j], p):
                        valid_geo = False
                        break
                if not valid_geo:
                    break
            
            if not valid_geo:
                continue
            
            # Check if adding p creates concyclicity
            for combo in combinations(range(len(current_set)), 3):
                p1, p2, p3 = current_set[combo[0]], current_set[combo[1]], current_set[combo[2]]
                if check_concyclic(p1, p2, p3, p):
                    valid_geo = False
                    break
            
            if valid_geo:
                # Found a valid 6th point!
                best_set = current_set[:] + [p]
                found_six = True
                break
        if found_six:
            break
    
    # If we found a 6th point, we could try to find a 7th, but it's unlikely.
    # We will return the best set found.
    
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_set]
    
    return {'D': D, 'points': points_out}

# EVOLVE-BLOCK-END
