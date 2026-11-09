# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _incidence_count(points, lines):
    # points: (N, 2), lines: (M, 3)
    px = points[:, 0:1].T
    py = points[:, 1:2].T
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    total = A * px + B * py + C
    return np.sum(total == 0)

def _get_best_lines(points):
    n = len(points)
    pairs = []
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A != 0 or B != 0:
                pairs.append((A, B, C))
    
    seen = set()
    unique_lines = []
    for A, B, C in pairs:
        g = math.gcd(abs(A), abs(B))
        g = math.gcd(g, abs(C))
        if g == 0: continue
        A, B, C = A // g, B // g, C // g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        key = (A, B, C)
        if key not in seen:
            seen.add(key)
            unique_lines.append(key)
            
    if len(unique_lines) < 27:
         for c in range(-20, 21):
             for A, B in [(1,0), (0,1), (1,1), (1,-1)]:
                 key = _normalize_line(A, B, -c)
                 if key not in seen:
                     seen.add(key)
                     unique_lines.append(key)
                     
    lines_arr = np.array(unique_lines, dtype=np.int64)
    
    px = points[:, 0:1].T
    py = points[:, 1:2].T
    A = lines_arr[:, 0:1]
    B = lines_arr[:, 1:2]
    C = lines_arr[:, 2:3]
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    if len(counts) > 27:
         idx = np.argsort(-counts)
         top_idx = idx[:27]
    else:
         top_idx = np.arange(len(counts))
         
    return lines_arr[top_idx]

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 590
    
    # Initial points: Grid
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = _get_best_lines(points)
    current_score = _incidence_count(points, lines)
    
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = current_score
    
    n_points = len(points)
    
    # Hill Climbing Loop
    # We will try to move points.
    # To allow exploration, we accept equal scores with prob 0.5.
    
    # To speed up, we can pre-allocate some arrays?
    # No, numpy handles it.
    
    # Let's run the loop.
    # We need to be careful with the time limit.
    
    # Check time every 1000 iterations.
    
    # To improve the search, we can also try to "swap" points?
    # No, moving one point is sufficient to reach any configuration.
    
    # Let's add a "restart" mechanism if we get stuck for a long time?
    # Or just keep going.
    
    # One issue: The points might drift far away.
    # We can constrain the search space?
    # Or just let it drift.
    
    # Let's try to bias the search towards the center?
    # No, just random walk.
    
    # To make the search more effective, we can try multiple random points per iteration?
    # No, one at a time is standard.
    
    # Let's run.
    
    # We will use a counter to check time.
    iter_count = 0
    # Use a smaller step size (1) to allow more frequent updates and finer tuning,
    # or vary it. Step 1 is safer for integer grid.
    while time.time() < deadline - 5:
        iter_count += 1
        
        # Pick a random point
        i = random.randint(0, n_points - 1)
        
        # Pick a new position
        # Try small steps first
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        
        new_x = points[i, 0] + dx
        new_y = points[i, 1] + dy
        
        # Check if new position is valid (not occupied)
        # Vectorized check
        if np.any((points[:, 0] == new_x) & (points[:, 1] == new_y) & (np.arange(n_points) != i)):
            continue
            
        # Create new points array
        new_points = points.copy()
        new_points[i, 0] = new_x
        new_points[i, 1] = new_y
        
        # Compute new best lines and score
        new_lines = _get_best_lines(new_points)
        new_score = _incidence_count(new_points, new_lines)
        
        # Accept/Reject
        if new_score > current_score:
            points = new_points
            lines = new_lines
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                best_lines = lines.copy()
                
        elif new_score == current_score:
            if random.random() < 0.5:
                points = new_points
                lines = new_lines
                # current_score stays same
                
        else:
            # Reject
            pass
            
    # Return best found
    return best_points, best_lines


# EVOLVE-BLOCK-END
