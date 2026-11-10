import math
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Helper functions for exact arithmetic checks
    
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
                    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
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
                        
                        # Circle equation: x^2 + y^2 + Dx + Ey + F = 0
                        # Determinant must be non-zero for no unique circle
                        # | x1^2+y1^2  x1  y1  1 |
                        # | x2^2+y2^2  x2  y2  1 | = 0
                        # | x3^2+y3^2  x3  y3  1 |
                        # | x4^2+y4^2  x4  y4  1 |
                        
                        d1 = x1*x1 + y1*y1
                        d2 = x2*x2 + y2*y2
                        d3 = x3*x3 + y3*y3
                        d4 = x4*x4 + y4*y4
                        
                        # Calculate determinant of 4x4 matrix
                        # M = [[d1, x1, y1, 1], [d2, x2, y2, 1], [d3, x3, y3, 1], [d4, x4, y4, 1]]
                        
                        # Expansion along first row
                        # d1 * det([[x2,y2,1],[x3,y3,1],[x4,y4,1]]) - x1 * det([[d2,y2,1],[d3,y3,1],[d4,y4,1]]) 
                        # + y1 * det([[d2,x2,1],[d3,x3,1],[d4,x4,1]]) - 1 * det([[d2,x2,y2],[d3,x3,y3],[d4,x4,y4]])
                        
                        def det3(a, b, c, d, e, f, g, h, i):
                            return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
                            
                        term1 = d1 * det3(x2, y2, 1, x3, y3, 1, x4, y4, 1)
                        term2 = -x1 * det3(d2, y2, 1, d3, y3, 1, d4, y4, 1)
                        term3 = y1 * det3(d2, x2, 1, d3, x3, 1, d4, x4, 1)
                        term4 = -det3(d2, x2, y2, d3, x3, y3, d4, x4, y4)
                        
                        det_val = term1 + term2 + term3 + term4
                        
                        if det_val == 0:
                            return False
        return True

    def is_valid(points):
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        # Check all pairwise distances are integers
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if not is_integral_dist(points[i], points[j]):
                    return False
        return True

    # Seed points from the prompt (known valid set of 5)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = seed_points
    best_size = len(seed_points)
    
    # If seed is not valid (it should be), we start empty
    if not is_valid(seed_points):
        best_points = []
        best_size = 0

    # Search for more points
    # We will try to add points from a candidate pool.
    # To make it faster, we can restrict the search space.
    # Let's try a grid search for the next point, then backtrack.
    
    # Candidate generation: points with small coordinates
    # We want to find 6, 7, 8 points.
    # Let's try a local search / random restart approach.
    
    max_coord = 200
    
    # Generate candidate pool
    # We prioritize points that are likely to form integer distances with existing points.
    # But for simplicity, let's just use a grid and filter.
    
    # To speed up, we can pre-filter candidates against the current set.
    
    def try_add_point(current_points, candidate):
        new_points = current_points + [candidate]
        # Quick checks before full validation
        # 1. Distances to all existing points
        for p in current_points:
            if not is_integral_dist(p, candidate):
                return False
        # 2. Collinearity with any pair
        for i in range(len(current_points)):
            for j in range(i + 1, len(current_points)):
                x1, y1 = current_points[i]
                x2, y2 = current_points[j]
                x3, y3 = candidate
                area2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if area2 == 0:
                    return False
        # 3. Concyclicity with any triple
        if len(current_points) >= 3:
             for i in range(len(current_points)):
                 for j in range(i + 1, len(current_points)):
                     for k in range(j + 1, len(current_points)):
                         x1, y1 = current_points[i]
                         x2, y2 = current_points[j]
                         x3, y3 = current_points[k]
                         x4, y4 = candidate
                         
                         d1 = x1*x1 + y1*y1
                         d2 = x2*x2 + y2*y2
                         d3 = x3*x3 + y3*y3
                         d4 = x4*x4 + y4*y4
                         
                         def det3(a, b, c, d, e, f, g, h, i):
                             return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
                             
                         term1 = d1 * det3(x2, y2, 1, x3, y3, 1, x4, y4, 1)
                         term2 = -x1 * det3(d2, y2, 1, d3, y3, 1, d4, y4, 1)
                         term3 = y1 * det3(d2, x2, 1, d3, x3, 1, d4, x4, 1)
                         term4 = -det3(d2, x2, y2, d3, x3, y3, d4, x4, y4)
                         
                         if term1 + term2 + term3 + term4 == 0:
                             return False
        return True

    # Backtracking search
    # We'll use a simple recursive backtracking with a time limit.
    
    def backtrack(points, candidates, depth):
        nonlocal best_points, best_size
        
        if time.time() > deadline:
            return
        
        if len(points) > best_size:
            best_points = points[:]
            best_size = len(points)
            
        if depth == 0 or not candidates:
            return
            
        # Try each candidate
        for i in range(len(candidates)):
            if time.time() > deadline:
                return
                
            cand = candidates[i]
            
            # Check if cand can be added
            if try_add_point(points, cand):
                # Recurse
                new_points = points + [cand]
                # Remove cand from candidates for next level to avoid duplicates? 
                # Since we are building a set, order doesn't matter for validity, 
                # but we want to avoid checking same set multiple times.
                # However, simple backtracking without pruning might be slow.
                # Let's just pass the remaining candidates.
                backtrack(new_points, candidates[i+1:], depth - 1)
                
            # If we found a large set, we might prune? 
            # But we want max size, so we continue.

    # Initial candidates
    # We'll generate a pool of points. 
    # To be efficient, we can sort candidates by some heuristic.
    # Let's use a small grid first.
    
    # We already have 5 points. Let's try to add 1, 2, 3 more.
    # We'll generate candidates in a box around the existing points or just a global box.
    
    # To avoid huge search space, we'll use a limited pool.
    # Let's try to find 1 more point first.
    
    # Generate candidates
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            candidates.append((x, y))
            
    # Filter candidates that are not already in seed
    existing = set(seed_points)
    candidates = [c for c in candidates if c not in existing]
    
    # Sort candidates? Maybe by distance to origin?
    # Let's just try a random subset or the first few.
    # Given time limit, we can't try all.
    # Let's try a random restart with a limited candidate pool size.
    
    random.seed(42)
    
    # We'll run multiple iterations of backtracking with different random subsets of candidates
    for _ in range(20):
        if time.time() > deadline:
            break
            
        # Shuffle candidates and take a subset
        random.shuffle(candidates)
        subset = candidates[:50]
        
        # Sort subset by some metric?
        # Let's just try to add points one by one.
        current = seed_points[:]
        
        for cand in subset:
            if time.time() > deadline:
                break
            if try_add_point(current, cand):
                current.append(cand)
                if len(current) > best_size:
                    best_points = current[:]
                    best_size = len(current)
        
        # If we found more than 5, we can try to add more from the remaining candidates
        if best_size > 5:
            # Try to add more from the rest of the candidates
            for cand in candidates[50:]:
                if time.time() > deadline:
                    break
                if try_add_point(best_points, cand):
                    # Check if it improves
                    new_pts = best_points + [cand]
                    if len(new_pts) > best_size:
                        best_points = new_pts
                        best_size = len(best_points)

    # Final validation
    if not is_valid(best_points):
        # Fallback to seed
        best_points = seed_points
        best_size = 5

    # Format output
    D = 1
    points_out = []
    for x, y in best_points:
        points_out.append([ [x, 1], [y, 1] ])
        
    return {'D': D, 'points': points_out}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We just need to expose it.
# EVOLVE-BLOCK-END
