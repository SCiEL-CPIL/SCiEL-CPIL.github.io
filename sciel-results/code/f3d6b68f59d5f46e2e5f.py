import numpy as np
import time
import random

N = 64
NUM_POINTS = N * N
DEADLINE = time.time() + 290

# Pre-compute squared distances
# Using a flat list or list of lists? List of lists is fine.
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    x1, y1 = i % N, i // N
    for j in range(i + 1, NUM_POINTS):
        x2, y2 = j % N, j // N
        d2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d2
        dist[j][i] = d2

def construct():
    start_time = time.time()
    best_set = []
    
    # Local variables for speed
    dist_local = dist
    num_points = NUM_POINTS
    
    while time.time() < DEADLINE - 2:
        order = list(range(num_points))
        random.shuffle(order)
        
        S = []
        dist_sets = [None] * num_points
        
        for p in order:
            if not S:
                S.append(p)
                dist_sets[p] = set()
                continue
            
            valid = True
            seen_dists = set()
            updates = []
            
            for q in S:
                d = dist_local[p][q]
                if d in seen_dists:
                    valid = False
                    break
                seen_dists.add(d)
                if d in dist_sets[q]:
                    valid = False
                    break
                updates.append((q, d))
            
            if valid:
                S.append(p)
                dist_sets[p] = seen_dists
                for q, d in updates:
                    dist_sets[q].add(d)
        
        if len(S) > len(best_set):
            best_set = S
            # Optional: early stop if we reach a very high number?
            # But we don't know the max.
            
    # Convert indices to (x, y)
    result = []
    for idx in best_set:
        x, y = idx % N, idx // N
        result.append((x, y))
        
    return result
