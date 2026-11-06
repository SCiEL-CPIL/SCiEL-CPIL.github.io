import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def check_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) != 0

    def check_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        dx2 = x2-x1; dy2 = y2-y1; da2 = a2-a1
        dx3 = x3-x1; dy3 = y3-y1; da3 = a3-a1
        dx4 = x4-x1; dy4 = y4-y1; da4 = a4-a1
        
        return (dx2*(dy3*da4 - da3*dy4) - dy2*(dx3*da4 - da3*dx4) + da2*(dx3*dy4 - dy3*dx4)) != 0

    def is_valid_set(points):
        n = len(points)
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_square(dx*dx + dy*dy):
                    return False
        
        # Check collinearity
        for i, j, k in combinations(range(n), 3):
            if not check_collinear(points[i], points[j], points[k]):
                return False
        
        # Check concyclicity
        for i, j, k, l in combinations(range(n), 4):
            if not check_concyclic(points[i], points[j], points[k], points[l]):
                return False
        
        return True

    def try_extend(points, limit):
        # Find a point P in [-limit, limit]^2 such that dist(P, Q) is integer for all Q in points
        # and P doesn't create collinearity/concyclicity
        current_set = list(points)
        
        # Pre-filter candidates
        # For efficiency, we can check against the last few points first?
        # No, just check all.
        
        # We iterate through a box
        # To avoid checking duplicates, we can use a set of points already in current_set
        existing = set(current_set)
        
        # We will try to find ONE valid extension
        # To increase chance, we can shuffle the order of candidates?
        # For now, simple scan.
        
        # Optimization: Only consider points that satisfy distance to all current points
        # This is the bottleneck.
        
        # We can break early if time is running out
        # But this function is called multiple times.
        
        # Let's just scan
        for x in range(-limit, limit + 1):
            if time.time() > deadline:
                return None
            for y in range(-limit, limit + 1):
                p = (x, y)
                if p in existing:
                    continue
                
                # Check distances to all points in current_set
                valid_dist = True
                for q in current_set:
                    dx = p[0] - q[0]
                    dy = p[1] - q[1]
                    if not is_square(dx*dx + dy*dy):
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                # Check collinearity with pairs in current_set
                valid_geo = True
                for i in range(len(current_set)):
                    for j in range(i+1, len(current_set)):
                        if not check_collinear(p, current_set[i], current_set[j]):
                            valid_geo = False
                            break
                    if not valid_geo:
                        break
                
                if not valid_geo:
                    continue
                
                # Check concyclicity with triples in current_set
                # Only needed if len(current_set) >= 3
                if len(current_set) >= 3:
                    for i, j, k in combinations(range(len(current_set)), 3):
                        if not check_concyclic(p, current_set[i], current_set[j], current_set[k]):
                            valid_geo = False
                            break
                
                if valid_geo:
                    return current_set + [p]
        
        return None

    # Initial seed (5 points)
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = seed_points
    
    # Try to extend the seed
    # Use a moderate limit first
    limit = 100
    
    # Attempt 1: Extend seed
    if time.time() < deadline:
        extended = try_extend(seed_points, limit)
        if extended:
            best_set = extended
            # Try to extend further
            if len(best_set) < 12:
                extended2 = try_extend(best_set, limit)
                if extended2:
                    best_set = extended2
                    # Try again
                    if len(best_set) < 12:
                        extended3 = try_extend(best_set, limit)
                        if extended3:
                            best_set = extended3

    # If we didn't get 6+ points, try a different strategy
    # Greedy construction from scratch
    if len(best_set) < 6 and time.time() < deadline:
        # Start with a small valid set
        # (0,0), (3,4)
        curr = [(0, 0), (3, 4)]
        
        # Try to grow it
        # We'll try to add points in a loop
        for _ in range(10): # Max 10 additions
            if time.time() > deadline:
                break
            # Try to add a point
            # We need to search for a point that fits curr
            # Use a smaller limit for speed, or random sampling?
            # Let's try scanning a small box around the centroid?
            # Or just a fixed box.
            
            # To be faster, we can pre-compute candidates for the *last* added point?
            # No, it must fit ALL.
            
            # Let's just call try_extend with a small limit
            # But try_extend scans a square.
            
            # To speed up, we can randomize the order of points in curr?
            # No, the constraint is symmetric.
            
            # Let's try a few different limits
            found = None
            for lim in [50, 100, 200]:
                if time.time() > deadline:
                    break
                found = try_extend(curr, lim)
                if found:
                    curr = found
                    break
            
            if not found:
                # If stuck, try a different seed?
                # Or break
                break
        
        if len(curr) > len(best_set):
            best_set = curr

    # Final verification and formatting
    # best_set is a list of (x, y)
    # D = 1
    
    return {
        'D': 1,
        'points': [[[x, 1], [y, 1]] for x, y in best_set]
    }
