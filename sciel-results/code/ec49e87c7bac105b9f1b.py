import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    def is_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = int(math.isqrt(n))
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def is_integral_dist(p1, p2):
        return is_square(dist_sq(p1, p2))

    def no_three_collinear(points):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    # 2 * Area
                    area2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if area2 == 0:
                        return False
        return True

    def no_four_concyclic(points):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                for k in range(j + 1, len(points)):
                    for l in range(k + 1, len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        
                        # Determinant of circle matrix
                        # | x^2+y^2  x  y  1 |
                        # ...
                        d1 = x1*x1 + y1*y1
                        d2 = x2*x2 + y2*y2
                        d3 = x3*x3 + y3*y3
                        d4 = x4*x4 + y4*y4
                        
                        # We can use the property that 4 points are concyclic iff the determinant is 0.
                        # To avoid large numbers, we can check if the circle through 1,2,3 passes through 4.
                        # Equation of circle through (x1,y1), (x2,y2), (x3,y3):
                        # x^2 + y^2 + Dx + Ey + F = 0
                        # Solve for D, E, F.
                        
                        # Use Cramer's rule or direct substitution.
                        # Let's use the determinant method but with careful integer arithmetic.
                        # M = [[d1, x1, y1, 1], [d2, x2, y2, 1], [d3, x3, y3, 1], [d4, x4, y4, 1]]
                        
                        # Expansion:
                        # d1*(x2*y3 - y2*x3 + x3*y4 - y3*x4 + x4*y2 - y4*x2) 
                        # - x1*(d2*y3 - y2*d3 + d3*y4 - y3*d4 + d4*y2 - y4*d2)
                        # + y1*(d2*x3 - x2*d3 + d3*x4 - x3*d4 + d4*x2 - x4*d2)
                        # - 1*(d2*x3*y4 - y2*d3*x4 + y2*d4*x3 - d2*x4*y3 + d4*x4*y3 - d4*x3*y4) -- Wait, signs are tricky.
                        
                        # Let's just compute the 4x4 determinant directly.
                        # It's O(1) for fixed size.
                        
                        def det4(m):
                            # m is 4x4 list of lists
                            return (
                                m[0][0] * (m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1])
                                - m[0][1] * (m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0])
                                + m[0][2] * (m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0])
                                - m[0][3] * (m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0])
                            )
                            
                        m = [
                            [d1, x1, y1, 1],
                            [d2, x2, y2, 1],
                            [d3, x3, y3, 1],
                            [d4, x4, y4, 1]
                        ]
                        if det4(m) == 0:
                            return False
        return True

    def is_valid(points):
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if not is_integral_dist(points[i], points[j]):
                    return False
        return True

    # Seed points
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify seed
    if not is_valid(seed_points):
        # Fallback to a simple valid set if seed fails (shouldn't happen)
        seed_points = [(0,0), (3,0), (0,4)]
        
    best_points = seed_points[:]
    best_size = len(seed_points)
    
    # Function to generate candidates given two points A, B
    # Returns list of integer points P such that dist(P,A) and dist(P,B) are integers
    def generate_candidates(A, B, max_radius=500):
        # Translate A to origin for simplicity? No, just use formulas.
        # Let A=(x1,y1), B=(x2,y2). d = dist(A,B).
        # We assume d is integer.
        d_sq = dist_sq(A, B)
        d = int(math.isqrt(d_sq))
        if d * d != d_sq or d == 0:
            return []
            
        # Vector AB
        vx = B[0] - A[0]
        vy = B[1] - A[1]
        
        candidates = []
        # r1, r2 are integer distances from A and B
        # x = (r1^2 - r2^2 + d^2) / (2 * vx) -- wait, this is projection.
        # Let's use the standard circle intersection.
        # We want integer (x,y).
        # Iterate r1, r2
        # To limit search, we can bound r1, r2.
        # But we don't know the bound.
        # However, if we are extending a set, the new point is likely close.
        # Let's try a range of radii.
        
        # A more robust way:
        # Iterate x in a range around the midpoint?
        # No, iterate r1, r2 is safer for integer solutions.
        
        # Note: x = (r1^2 - r2^2 + d^2) / (2 * (B[0]-A[0])) is only valid if vx != 0.
        # If vx == 0, use y.
        
        # Let's just iterate x and y in a box?
        # The previous program did this. It was slow.
        # Let's stick to the radius iteration but be smart.
        
        # Actually, for a given pair A, B, the set of integer points P with integer distances to A and B is finite if we bound the radius.
        # But we don't have a bound.
        # However, we can generate candidates for the *next* point by looking at the current set.
        
        # Let's try a different approach:
        # Instead of generating from a pair, generate from the *entire* current set?
        # No, that's too restrictive.
        
        # Let's go back to the box search but make it smarter.
        # We will search in a box centered at the centroid of the current points.
        
        return [] # Placeholder, we will use box search in the main loop

    # Backtracking search
    def backtrack(points):
        nonlocal best_points, best_size
        
        if time.time() > deadline:
            return
            
        if len(points) > best_size:
            best_points = points[:]
            best_size = len(points)
            
        # If we found 8, we can stop early? Or try for 9?
        # Let's keep going if time permits.
        
        # Generate candidates
        # We will search in a box around the current points.
        # Determine bounds
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Expand box
        margin = 100
        search_min_x = min_x - margin
        search_max_x = max_x + margin
        search_min_y = min_y - margin
        search_max_y = max_y + margin
        
        # To avoid re-checking, we can sort candidates.
        # But we need to find *any* valid point.
        
        # Heuristic: Try points closer to the centroid first?
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        candidates = []
        # Limit box size to avoid explosion
        if search_max_x - search_min_x > 400:
            search_min_x = int(cx) - 200
            search_max_x = int(cx) + 200
        if search_max_y - search_min_y > 400:
            search_min_y = int(cy) - 200
            search_max_y = int(cy) + 200
            
        for x in range(search_min_x, search_max_x + 1):
            if time.time() > deadline:
                return
            for y in range(search_min_y, search_max_y + 1):
                cand = (x, y)
                if cand in points:
                    continue
                
                # Check distances to all points in current set
                valid_dist = True
                for p in points:
                    if not is_integral_dist(p, cand):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                
                # Check collinearity
                valid_coll = True
                for i in range(len(points)):
                    for j in range(i + 1, len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        # area2
                        area2 = abs(x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2))
                        if area2 == 0:
                            valid_coll = False
                            break
                    if not valid_coll:
                        break
                if not valid_coll:
                    continue
                
                # Check concyclicity
                valid_conc = True
                if len(points) >= 3:
                    for i in range(len(points)):
                        for j in range(i + 1, len(points)):
                            for k in range(j + 1, len(points)):
                                x1, y1 = points[i]
                                x2, y2 = points[j]
                                x3, y3 = points[k]
                                
                                # Check if (x,y) is on circle through (x1,y1), (x2,y2), (x3,y3)
                                # Use determinant
                                d1 = x1*x1 + y1*y1
                                d2 = x2*x2 + y2*y2
                                d3 = x3*x3 + y3*y3
                                d4 = x*x + y*y
                                
                                m = [
                                    [d1, x1, y1, 1],
                                    [d2, x2, y2, 1],
                                    [d3, x3, y3, 1],
                                    [d4, x, y, 1]
                                ]
                                
                                # det4
                                val = (
                                    m[0][0] * (m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1])
                                    - m[0][1] * (m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0])
                                    + m[0][2] * (m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0])
                                    - m[0][3] * (m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0])
                                )
                                if val == 0:
                                    valid_conc = False
                                    break
                            if not valid_conc:
                                break
                        if not valid_conc:
                            break
                if not valid_conc:
                    continue
                
                # Valid candidate found!
                backtrack(points + [cand])
                
                # If we found a large set, we might want to stop exploring this branch?
                # No, we want the MAX size.
                # But we can prune if we've already found the max possible?
                # We don't know the max.
                
                # To avoid symmetry issues (same set in different order), we can enforce an ordering.
                # But since we are adding points, and the set is a set, order doesn't matter for validity.
                # However, backtracking will explore all permutations.
                # To mitigate, we can enforce that new points have coordinates greater than some previous point?
                # This is tricky with 2D.
                # For now, we rely on the time limit.

    backtrack(seed_points)
    
    # Final check
    if not is_valid(best_points):
        best_points = seed_points
        
    # Format output
    D = 1
    points_out = []
    for x, y in best_points:
        points_out.append([ [x, 1], [y, 1] ])
        
    return {'D': D, 'points': points_out}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
