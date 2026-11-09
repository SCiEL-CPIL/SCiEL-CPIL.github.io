import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds is plenty for 32 points
    
    # 1. Generate initial compact shape
    # Hexagon of side 3 (19 points)
    points = set()
    # Center at (0,0)
    # Hexagon points (a,b) such that max(|a|, |b|, |a+b|) <= k? 
    # Actually, in axial coordinates for triangular lattice, a hexagon of radius R 
    # includes points where max(|a|, |b|, |a-b|) <= R? No.
    # Let's use a simple spiral or just enumerate a bounding box and filter.
    
    # Better: Generate a hexagon of side 4 (37 points) and remove 5.
    # Hexagon of side k has points (a,b) with |a|<=k, |b|<=k, |a+b|<=k? 
    # Let's check the shape.
    # Standard hexagon in axial coords (a,b) with c = -a-b:
    # |a| <= R, |b| <= R, |c| <= R.
    # For R=3:
    # a in [-3,3], b in [-3,3], |a+b| <= 3.
    # Count: 
    # a=0: b in [-3,3] (7)
    # a=1: b in [-3,2] (6)
    # a=2: b in [-3,1] (5)
    # a=3: b in [-3,0] (4)
    # Symmetric for negative a.
    # Total = 7 + 2*(6+5+4) = 7 + 30 = 37. Correct.
    
    R = 3
    candidates = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                candidates.append((a,b))
                
    # We have 37 points. We need 32.
    # We will remove 5 points to maximize the remaining edges.
    # This is equivalent to finding a subset of 32 points with max edges.
    # Let's just use local search on a set of 32 points.
    
    # Initial set: First 32 points from the list (arbitrary order)
    # To be smarter, let's start with the 37 and remove 5.
    current_points = set(candidates)
    
    # Function to count edges
    def count_edges(pts):
        count = 0
        pts_list = list(pts)
        for i in range(len(pts_list)):
            a1, b1 = pts_list[i]
            for j in range(i+1, len(pts_list)):
                a2, b2 = pts_list[j]
                da = a1 - a2
                db = b1 - b2
                # dist^2 = da^2 + da*db + db^2
                if da*da + da*db + db*db == 1:
                    count += 1
        return count
        
    # Greedy removal of 5 points
    # Calculate degree of each point within the set
    # Remove point with lowest degree
    for _ in range(5):
        if time.time() > deadline - 5: break
        degrees = {}
        pts_list = list(current_points)
        for p in pts_list:
            degrees[p] = 0
        for i in range(len(pts_list)):
            a1, b1 = pts_list[i]
            for j in range(len(pts_list)):
                if i == j: continue
                a2, b2 = pts_list[j]
                da = a1 - a2
                db = b1 - b2
                if da*da + da*db + db*db == 1:
                    degrees[pts_list[i]] += 1
        # Find point with min degree
        min_deg_point = min(degrees, key=degrees.get)
        current_points.remove(min_deg_point)
        
    # Now we have 32 points.
    # Perform local search to improve
    # Moves: Pick a point, move it to a nearby random empty spot, or swap with another point.
    # Given the small size, we can try swapping any two points with any two other points? 
    # That's expensive. 
    # Let's try single point moves.
    # Pick a point p, remove it. Pick a new location q (random nearby), if q not in set, add q.
    # If score improves, keep.
    
    best_score = count_edges(current_points)
    best_set = set(current_points)
    
    # Local Search
    attempts = 0
    while time.time() < deadline - 1:
        attempts += 1
        # Pick a random point to move
        p = random.choice(list(best_set))
        # Remove it
        trial_set = set(best_set)
        trial_set.remove(p)
        
        # Generate random nearby location
        # Neighbors of p are at distance 1. 
        # Let's pick a random offset (da, db) with small magnitude
        # Or just pick a random point from a larger bounding box around the centroid
        # Centroid
        sum_a = sum(x[0] for x in best_set)
        sum_b = sum(x[1] for x in best_set)
        ca = sum_a // 32
        cb = sum_b // 32
        
        # Try a few random moves
        improved = False
        for _ in range(10):
            da = random.randint(-5, 5)
            db = random.randint(-5, 5)
            new_p = (ca + da, cb + db)
            if new_p in trial_set:
                continue
            # Check bounds
            if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                continue
            trial_set.add(new_p)
            score = count_edges(trial_set)
            if score > best_score:
                best_score = score
                best_set = trial_set
                improved = True
                break
            trial_set.remove(new_p)
            
        if not improved and attempts % 1000 == 0:
            # Maybe restart with a different initial shape?
            # Or just keep going.
            pass
            
    return [list(p) for p in best_set]
