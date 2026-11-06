"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Start with an empty set and greedily add points
    selected = []
    selected_set = set()
    
    # Try each point in a randomized order to get different results
    import random
    random.seed(42)
    point_order = all_points[:]
    random.shuffle(point_order)
    
    for p in point_order:
        if time.time() > deadline:
            break
        
        if p in selected_set:
            continue
            
        # Check if adding p creates any isosceles triangle
        valid = True
        if len(selected) >= 2:
            # For each pair of existing points, check if p forms an isosceles triangle
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                if not valid:
                    break
                for j in range(i + 1, len(selected)):
                    # Check if (selected[i], selected[j], p) forms an isosceles triangle
                    d1 = (selected[i][0] - selected[j][0])**2 + (selected[i][1] - selected[j][1])**2
                    d2 = (selected[i][0] - p[0])**2 + (selected[i][1] - p[1])**2
                    d3 = (selected[j][0] - p[0])**2 + (selected[j][1] - p[1])**2
                    
                    # Isosceles if any two distances are equal
                    if d1 == d2 or d1 == d3 or d2 == d3:
                        valid = False
                        break
        
        if valid:
            selected.append(p)
            selected_set.add(p)
    
    return selected
# EVOLVE-BLOCK-END
