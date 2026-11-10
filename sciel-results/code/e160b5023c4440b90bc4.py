import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 30
    MIN_COORD = -30
    MAX_COORD = 30
    
    def doubled_area(p1, p2, p3):
        return abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

    # Initialize points
    # Use a grid start to have many small areas
    # Grid 6x5 = 30 points
    # x in 0..5, y in 0..4 -> scale to [-30, 30]?
    # The seed program uses [(x,y) for x in range(6) for y in range(5)]
    # This is in [0,5]x[0,4].
    # Let's start with a random distribution or a scaled grid.
    
    # Random start is often better for SA to explore.
    points = []
    while len(points) < N:
        x = random.randint(MIN_COORD, MAX_COORD)
        y = random.randint(MIN_COORD, MAX_COORD)
        if (x, y) not in points:
            points.append((x, y))
            
    # Initialize counts
    counts = {}
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                d = doubled_area(points[i], points[j], points[k])
                if d > 0:
                    counts[d] = counts.get(d, 0) + 1
                    
    current_max_mult = max(counts.values()) if counts else 0
    best_points = points.copy()
    best_max_mult = current_max_mult
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        i = random.randint(0, N-1)
        old_p = points[i]
        
        # Generate new point
        # Random in box
        new_p = (random.randint(MIN_COORD, MAX_COORD), random.randint(MIN_COORD, MAX_COORD))
        
        if new_p == old_p:
            continue
        if new_p in points:
            continue
            
        # Calculate old areas involving i
        # And new areas involving i
        
        # We will update the counts dict.
        # To avoid modifying the dict while iterating or losing track, 
        # we can just do it directly.
        
        # Remove old contributions
        for j in range(N):
            if j == i: continue
            for k in range(j+1, N):
                if k == i: continue
                # Triangle (i, j, k)
                d = doubled_area(old_p, points[j], points[k])
                if d > 0:
                    counts[d] -= 1
                    if counts[d] == 0:
                        del counts[d]
                        
        # Update point
        points[i] = new_p
        
        # Add new contributions
        for j in range(N):
            if j == i: continue
            for k in range(j+1, N):
                if k == i: continue
                # Triangle (i, j, k)
                d = doubled_area(new_p, points[j], points[k])
                if d > 0:
                    counts[d] = counts.get(d, 0) + 1
                    
        new_max_mult = max(counts.values()) if counts else 0
        
        # Acceptance criterion
        delta = new_max_mult - current_max_mult
        if delta > 0:
            current_max_mult = new_max_mult
            if current_max_mult > best_max_mult:
                best_max_mult = current_max_mult
                best_points = points.copy()
        else:
            # Simulated annealing acceptance
            # If delta is negative, accept with probability exp(delta / T)
            # delta is integer, so exp(-1/T) etc.
            if random.random() < math.exp(delta / T):
                current_max_mult = new_max_mult
            else:
                # Reject: revert
                # Revert point
                points[i] = old_p
                # Revert counts
                # Remove new contributions
                for j in range(N):
                    if j == i: continue
                    for k in range(j+1, N):
                        if k == i: continue
                        d = doubled_area(new_p, points[j], points[k])
                        if d > 0:
                            counts[d] -= 1
                            if counts[d] == 0:
                                del counts[d]
                # Add old contributions
                for j in range(N):
                    if j == i: continue
                    for k in range(j+1, N):
                        if k == i: continue
                        d = doubled_area(old_p, points[j], points[k])
                        if d > 0:
                            counts[d] = counts.get(d, 0) + 1
                            
        T *= cooling
        
    return best_points
