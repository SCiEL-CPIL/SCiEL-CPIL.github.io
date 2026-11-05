import math
import time

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance pairs (edges).
    The optimal number of edges is 76.
    """
    # Deadline guard
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def is_adjacent(p, q):
        return dist_sq(p, q) == 1

    def count_edges(points):
        count = 0
        n = len(points)
        # Using set for O(1) neighbor lookup? No, just O(n^2) is fine for n=32
        for i in range(n):
            for j in range(i+1, n):
                if is_adjacent(points[i], points[j]):
                    count += 1
        return count

    # 1. Generate initial configuration: Spiral
    # Spiral order of lattice points
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1), (2,1)
    # ...
    
    def spiral_points(count):
        points = [(0,0)]
        if count <= 1:
            return points
        
        # Directions in order: E, NE, N, NW, W, SW? 
        # Let's use the standard 6 directions:
        # (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
        # Actually, let's just iterate by "rings" or use a known spiral generator.
        
        # A simpler way: generate points in increasing max(abs(a), abs(b), abs(a+b))
        # This creates a hexagonal spiral.
        r = 0
        while len(points) < count:
            r += 1
            # Generate all points on the hexagon of radius r
            # The hexagon of radius r consists of points where max(abs(a), abs(b), abs(a+b)) == r
            # We can iterate a from -r to r, and for each a, determine valid b range.
            # For a given a, b must satisfy:
            # -r <= b <= r
            # -r <= a+b <= r  =>  -r-a <= b <= r-a
            # So b is in [max(-r, -r-a), min(r, r-a)]
            for a in range(-r, r+1):
                b_min = max(-r, -r - a)
                b_max = min(r, r - a)
                for b in range(b_min, b_max + 1):
                    if max(abs(a), abs(b), abs(a+b)) == r:
                        points.append((a, b))
                        if len(points) >= count:
                            return points[:count]
        return points[:count]

    initial_points = spiral_points(32)
    current_points = [list(p) for p in initial_points]
    current_set = set(map(tuple, current_points))
    current_edges = count_edges(current_points)
    
    best_points = [list(p) for p in current_points]
    best_edges = current_edges

    # 2. Local Search
    # Candidate pool: Points within a reasonable distance
    # Let's use a pool of points with max(abs(a), abs(b), abs(a+b)) <= 6
    # This is 1 + 6*3*5 = 91 points? 
    # Number of points in hexagon of radius R is 1 + 6*R*(R+1)/2 = 3R^2 + 3R + 1.
    # For R=5: 75+15+1 = 91.
    # For R=6: 108+18+1 = 127.
    # 127 candidates is small enough.
    
    R_POOL = 6
    pool = []
    for a in range(-R_POOL, R_POOL + 1):
        for b in range(-R_POOL, R_POOL + 1):
            if max(abs(a), abs(b), abs(a+b)) <= R_POOL:
                pool.append((a, b))
    
    # Pre-compute adjacency within pool? 
    # Or just calculate on the fly. 
    # To speed up, let's pre-compute neighbors for each pool point within the pool.
    # But we need neighbors in the CURRENT SET, which changes.
    
    # Optimization: 
    # For each swap (remove p, add c):
    # Delta = (neighbors of c in current_set) - (neighbors of p in current_set)
    # Note: p is in current_set, c is not.
    # Neighbors of p in current_set = degree(p)
    # Neighbors of c in current_set = count of q in current_set such that is_adjacent(c, q)
    
    # We can maintain a list of current points and their degrees.
    # But degrees change when we swap.
    # Since n=32, recalculating degrees is cheap.
    
    improved = True
    iterations = 0
    while improved and current_edges < 76 and time.time() < deadline - 10:
        improved = False
        best_swap_delta = 0
        best_p_idx = -1
        best_c_pt = None
        
        # Calculate degrees of all current points
        degrees = [0] * len(current_points)
        for i in range(len(current_points)):
            p = current_points[i]
            for j in range(len(current_points)):
                if i != j and is_adjacent(p, current_points[j]):
                    degrees[i] += 1
        
        # Try all swaps
        for i in range(len(current_points)):
            p = current_points[i]
            deg_p = degrees[i]
            
            for c in pool:
                if tuple(c) in current_set:
                    continue
                
                # Calculate potential degree of c
                deg_c = 0
                for j in range(len(current_points)):
                    if is_adjacent(c, current_points[j]):
                        deg_c += 1
                
                delta = deg_c - deg_p
                if delta > best_swap_delta:
                    best_swap_delta = delta
                    best_p_idx = i
                    best_c_pt = c
        
        if best_swap_delta > 0:
            # Perform swap
            old_p = current_points[best_p_idx]
            current_set.remove(tuple(old_p))
            current_set.add(tuple(best_c_pt))
            current_points[best_p_idx] = list(best_c_pt)
            
            # Recalculate edges
            current_edges = count_edges(current_points)
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = [list(p) for p in current_points]
                
            improved = True
            iterations += 1
            
            # Early exit if we hit the theoretical max
            if current_edges == 76:
                break

    # Final check
    if best_edges > current_edges:
        final_points = best_points
    else:
        final_points = current_points

    # Ensure distinctness and validity (though logic above ensures it)
    # And return list of lists
    return [list(p) for p in final_points]
