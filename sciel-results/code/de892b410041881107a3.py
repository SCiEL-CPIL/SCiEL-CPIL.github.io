import math
import time
import random
import numpy as np

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START

def _fast_greedy_dist_set_optimized(seed, deadline):
    """
    Fast greedy algorithm using a global distance set to check for isosceles triangles.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create list of all points and shuffle
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    points = []
    points_array = np.array([]).reshape(0, 2)
    dist_set = set()
    
    for p in all_points:
        if time.time() > deadline:
            break
            
        if len(points_array) == 0:
            points.append(p)
            points_array = np.array([p])
            continue

        px, py = p
        # Vectorized distance calculation
        dx = points_array[:, 0] - px
        dy = points_array[:, 1] - py
        d2 = dx * dx + dy * dy
        
        # Check if distances from p to existing points are distinct
        # Using sort is O(K log K). K is small.
        # Alternatively, use a set: if len(set(d2)) != len(d2): continue
        # set(d2) is O(K). Faster for small K.
        if len(set(d2.tolist())) != len(d2):
            continue

        # Check if any distance is already in dist_set
        valid = True
        for d in d2:
            if int(d) in dist_set:
                valid = False
                break
        
        if valid:
            points.append(p)
            # Update points_array
            points_array = np.vstack([points_array, np.array(p)])
            # Update dist_set
            for d in d2:
                dist_set.add(int(d))
            
    return points

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    
    # Run many random restarts
    seed = 0
    while time.time() < deadline - 2:
        points = _fast_greedy_dist_set_optimized(seed, deadline)
        if len(points) > len(best_points):
            best_points = points
        seed += 1
        
    return best_points

# EVOLVE-BLOCK-END
