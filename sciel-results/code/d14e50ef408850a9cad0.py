import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    def hex_dist(a, b, c, d):
        # Calculate squared distance to avoid sqrt
        # dist^2 = (a-c)^2 + (b-d)^2 + (a-c)(b-d)
        da = a - c
        db = b - d
        return da * da + db * db + da * db

    def is_valid_point(p):
        a, b = p
        return abs(a) <= 64 and abs(b) <= 64

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if hex_dist(points[i][0], points[i][1], points[j][0], points[j][1]) == 1:
                    count += 1
        return count

    # 1. Generate initial static configuration (same as current program)
    points = []
    
    def hex_dist_center(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
        
    # Add all points with hex_dist <= 2
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist_center(a, b) <= 2:
                points.append([a, b])
                
    # Generate points with hex_dist == 3
    shell3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist_center(a, b) == 3:
                shell3.append([a, b])
                
    def angle(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    shell3.sort(key=angle)
    selected_shell3 = shell3[:13]
    points.extend(selected_shell3)
    
    # Ensure distinctness (should be distinct by construction)
    # Convert to set for fast lookup
    point_set = set(map(tuple, points))
    
    best_points = [p[:] for p in points]
    best_score = count_edges(best_points)
    
    # 2. Local Search / Simulated Annealing
    # We only have 32 points, so search space is manageable.
    # Move: Pick a random point, move it to a random nearby empty location.
    
    random.seed(42)
    
    # Precompute neighbors for fast distance checks? 
    # Or just brute force count_edges if n is small.
    # n=32, count_edges is 32*31/2 = 496 checks. Very fast.
    
    current_points = [p[:] for p in points]
    current_score = best_score
    
    # Temperature for simulated annealing
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, len(current_points) - 1)
        old_p = current_points[idx][:]
        
        # Try a few random moves
        improved = False
        for _ in range(10):
            # Generate a new position nearby
            # Random step in a, b
            da = random.randint(-2, 2)
            db = random.randint(-2, 2)
            new_p = [old_p[0] + da, old_p[1] + db]
            
            if not is_valid_point(new_p):
                continue
                
            # Check if new_p is already occupied
            new_p_tuple = tuple(new_p)
            if new_p_tuple in point_set:
                if new_p_tuple == tuple(old_p):
                    continue
                # If it's a different point, we can't just place it there unless we swap
                # For simplicity, only allow moves to empty spots
                continue
            
            # Calculate score change
            # Remove old_p, add new_p
            # This is expensive to do by full recount.
            # Let's just do a full recount if the move looks promising?
            # Or just do full recount for every move?
            # 496 checks * 10 moves * many iterations.
            # 140s is plenty for ~100k-1M simple checks.
            
            # Swap in
            point_set.remove(tuple(old_p))
            point_set.add(new_p_tuple)
            current_points[idx] = new_p
            
            new_score = count_edges(current_points)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
                improved = True
                break # Accept move and restart search for this point? Or continue?
            else:
                # Revert
                point_set.remove(new_p_tuple)
                point_set.add(tuple(old_p))
                current_points[idx] = old_p[:]
                
            # If we found an improvement, we break the inner loop and restart outer
            # If not, we continue trying other moves for this point?
            # Actually, if we break on improvement, we accept it.
            
        # If no improvement in 10 tries, maybe lower temperature?
        # Simple hill climbing with random restarts is often better.
        # Let's add a random restart mechanism if stuck.
        
        # Check if we've reached the theoretical max
        if best_score >= 76:
            break
            
        # Simple cooling
        temp *= cooling
        
    return best_points
