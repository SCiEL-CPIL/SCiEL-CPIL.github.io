import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Generate points of Projective Plane PG(2, 7)
    # Points are 1D subspaces of F_7^3
    q = 7
    points = []
    seen = set()
    
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Canonical representation: first non-zero coordinate is 1
                if x != 0:
                    canon = (1, y * pow(x, q-2, q), z * pow(x, q-2, q))
                elif y != 0:
                    canon = (0, 1, z * pow(y, q-2, q))
                else:
                    canon = (0, 0, 1)
                
                if canon not in seen:
                    seen.add(canon)
                    points.append(canon)
    
    n_total = len(points) # Should be 57
    if n_total != 57:
        # Fallback if generation logic has a bug, though math says 57
        pass

    # 2. Build adjacency for the Polarity Graph
    # Adjacency: P . Q == 0 (mod q)
    # We will store edges for the full graph first, then filter
    # Or just build the subset directly.
    
    # We want n=43. Let's select the first 43 points.
    target_n = 43
    selected_points = points[:target_n]
    
    edges = []
    
    # Map point index to canonical tuple for lookup? 
    # We can just compute dot products directly.
    
    for i in range(target_n):
        p1 = selected_points[i]
        for j in range(i + 1, target_n):
            p2 = selected_points[j]
            # Dot product mod 7
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if dot == 0:
                edges.append([i, j])
                
    # Verification (Optional but good for safety, skip if time is tight)
    # The graph is C4-free by construction.
    # Independence number is small.
    
    return {'n': target_n, 'edges': edges}
# EVOLVE-BLOCK-END
