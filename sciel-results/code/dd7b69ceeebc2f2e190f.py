import random
import math
import time

def construct():
    # Lattice neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db

    def count_edges(points):
        edges = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    edges += 1
        return edges

    def is_valid(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True

    # Generate hexagon of radius 3
    # Center (0,0)
    # Radius k points are at distance k from center in lattice steps?
    # Actually, let's just generate a box and filter by hexagon shape or just use a large box.
    # Hexagon of radius 3 has 37 points.
    # Coordinates:
    # a from -3 to 3
    # b from -3 to 3
    # Condition for hexagon radius 3: max(|a|, |b|, |a+b|) <= 3 ? 
    # Let's check: (3,0) -> max(3,0,3)=3. (0,3) -> 3. (3,-3) -> max(3,-3,0)=3.
    # Yes, the hexagon of radius R is { (a,b) | max(|a|, |b|, |a+b|) <= R }.
    
    points_hex = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                points_hex.append((a,b))
                
    assert len(points_hex) == 37, f"Expected 37, got {len(points_hex)}"
    
    # We need 32 points.
    # Strategy: Start with all 37, remove 5 random points, then local search.
    # To be deterministic and fast, we can try removing points from the boundary first.
    # Or just run local search on the 37 points, but we must have exactly 32.
    
    # Let's use a local search that allows adding/removing? No, fixed N.
    # Let's start with a random subset of 32 from the 37.
    # Better: Start with the 37 points, pick 5 to remove.
    # The points with the fewest neighbors in the full hexagon are the corners.
    # Corners: (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3). There are 6 corners.
    # Removing a corner loses 2 edges. Removing an edge point (non-corner) loses 3 edges.
    # We need to remove 5 points.
    # If we remove 5 corners, we lose 5*2 = 10 edges?
    # Total edges in hexagon 3:
    # Formula: 3n - perimeter? 
    # For n=37, max edges is 3*37 - sqrt(12*37-3) = 111 - sqrt(441) = 111 - 21 = 90.
    # If we remove 5 corners, we lose 10 edges. 90 - 10 = 80.
    # But the remaining 32 points might rearrange? No, we are just removing.
    # However, the optimal 32-point shape might not be a subset of the 37-point hexagon.
    # But it's a very good candidate.
    
    # Let's try a local search that can swap points or move points.
    # Since we are constrained to 32 points, we can't just move them freely if we start with a fixed set.
    # We can start with the 32 points (37 - 5 corners) and then try to move points to improve edge count.
    
    # Identify corners of the hexagon radius 3
    corners = [(3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)]
    
    # Remove 5 corners
    initial_points = [p for p in points_hex if p not in corners[:5]]
    
    # Local Search
    # Objective: maximize edges
    # Move: pick a point, try to move to one of 6 neighbors.
    # Constraint: new position must not overlap with any other point.
    
    best_points = initial_points[:]
    best_edges = count_edges(best_points)
    
    # Simple Hill Climbing with random restarts
    # Since N=32 is small, this will be fast.
    
    for restart in range(100):
        if restart == 0:
            current_points = initial_points[:]
        else:
            # Perturb current best
            current_points = best_points[:]
            # Randomly move a few points to random valid nearby positions?
            # Or just start from a random subset of the hexagon?
            # Let's stick to hill climbing from the best found so far, with random initialization for the first few restarts.
            if restart < 10:
                # Random subset of 32 from hexagon
                random.shuffle(points_hex)
                current_points = points_hex[:32]
            else:
                # Perturb best
                current_points = best_points[:]
                # Move 1-3 points to random neighbors
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 31)
                    a, b = current_points[idx]
                    na, nb = random.choice(neighbors)
                    new_p = (a+na, b+nb)
                    # Check if valid
                    valid = True
                    for j, p in enumerate(current_points):
                        if j == idx: continue
                        if dist_sq(new_p, p) < 1:
                            valid = False
                            break
                    if valid:
                        current_points[idx] = new_p
        
        improved = True
        while improved:
            improved = False
            current_edges = count_edges(current_points)
            # Try moving each point
            for i in range(32):
                a, b = current_points[i]
                for da, db in neighbors:
                    new_a, new_b = a + da, b + db
                    new_p = (new_a, new_b)
                    # Check bounds? The problem says |a|,|b| <= 64. Our hexagon is within 3, so moves are safe.
                    # Check collision
                    valid = True
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(new_p, current_points[j]) < 1:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate change in edges
                    # Remove edges involving i
                    old_edges_i = 0
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(current_points[i], current_points[j]) == 1:
                            old_edges_i += 1
                    
                    # Add edges involving new_p
                    new_edges_i = 0
                    for j in range(32):
                        if j == i: continue
                        if dist_sq(new_p, current_points[j]) == 1:
                            new_edges_i += 1
                    
                    delta = new_edges_i - old_edges_i
                    if delta > 0:
                        current_points[i] = new_p
                        current_edges += delta
                        improved = True
                        break # Restart loop for this point? Or just continue?
                    # If delta == 0, we could accept to escape local minima, but let's stick to strict improvement for simplicity first.
                    # If we get stuck, the restarts will help.
                
                if improved:
                    break
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_points = current_points[:]
        
        if best_edges == 76:
            break
            
    # Convert to list of lists
    return [[a, b] for a, b in best_points]
