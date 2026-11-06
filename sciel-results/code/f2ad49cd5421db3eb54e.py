import random
import time
import math

def count_collinear_triples(points):
    """Count the number of triples of points that are collinear."""
    n = len(points)
    triples = 0
    # We can just check all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Check if (p1, p2, p3) are collinear
                # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
                if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0]):
                    triples += 1
    return triples

def has_four_collinear(points):
    """Check if there are 4 or more collinear points."""
    n = len(points)
    # Use a set of lines represented by normalized (A, B, C) for Ax+By+C=0
    # Or simpler: for each pair, count points on the line.
    # To avoid duplicates, we can just iterate pairs and count, but we need to be careful not to overcount lines.
    # Actually, we just need to know if ANY line has >= 4 points.
    
    # For each point i, and each other point j > i, define line.
    # This is O(N^3) if we count for every pair.
    # N=20, N^3 = 8000. Very fast.
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction? Not strictly necessary for counting on the line, 
            # but we must ensure we are checking the same line.
            # The line is defined by the pair (i, j).
            # We check how many other points k lie on this line.
            count = 2 # i and j
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Check if (x1,y1), (x2,y2), (x3,y3) are collinear
                if dx * (y3 - y1) == dy * (x3 - x1):
                    count += 1
                    if count >= 4:
                        return True
    return False

def generate_valid_initial_points():
    """Generate 20 random points with no 4 collinear."""
    while True:
        points = []
        for _ in range(20):
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            points.append([x, y])
        
        # Check for duplicates
        if len(set(map(tuple, points))) < 20:
            continue
            
        if not has_four_collinear(points):
            return points

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial solution
    best_points = generate_valid_initial_points()
    best_score = count_collinear_triples(best_points)
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # New random position
        new_x = random.randint(-100, 100)
        new_y = random.randint(-100, 100)
        
        # Check if new point is same as old
        if new_x == old_x and new_y == old_y:
            continue
            
        # Check for duplicates with other points
        is_dup = False
        for i in range(20):
            if i == idx:
                continue
            if current_points[i][0] == new_x and current_points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Create new configuration
        new_points = [p[:] for p in current_points]
        new_points[idx] = [new_x, new_y]
        
        # Check constraint: no 4 collinear
        if has_four_collinear(new_points):
            continue
            
        # Calculate new score
        new_score = count_collinear_triples(new_points)
        
        # Acceptance
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_points
