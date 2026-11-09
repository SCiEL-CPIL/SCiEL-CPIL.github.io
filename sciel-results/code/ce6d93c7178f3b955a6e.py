import random
import time

N = 64

def construct():
    # Precompute all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    best_size = 0
    best_set = []
    
    start_time = time.time()
    deadline = start_time + 290  # 10 second safety margin
    
    seed = 0
    while time.time() < deadline:
        random.seed(seed)
        candidates = all_points[:]
        random.shuffle(candidates)
        
        S = []
        # dist_sets[a] is a set of squared distances from a to other points in S
        dist_sets = {}
        
        for p in candidates:
            valid = True
            dists_set = set()
            px, py = p
            
            # Check against all existing points
            for a in S:
                ax, ay = a
                dx = px - ax
                dy = py - ay
                d2 = dx*dx + dy*dy
                
                # Check if distance from P to A is same as P to some other point (Apex at P)
                if d2 in dists_set:
                    valid = False
                    break
                
                # Check if distance from P to A is same as distance from B to A for some B in S (Apex at A)
                if d2 in dist_sets[a]:
                    valid = False
                    break
                    
                dists_set.add(d2)
            
            if valid:
                # Add P to S
                S.append(p)
                dist_sets[p] = dists_set
                
                # Update dist_sets for all existing points
                for a in S[:-1]:
                    ax, ay = a
                    dx = px - ax
                    dy = py - ay
                    d2 = dx*dx + dy*dy
                    dist_sets[a].add(d2)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        seed += 1
        
    return best_set
