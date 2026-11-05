import numpy as np
import time
import random

N = 64
TIME_LIMIT = 320  # seconds, safety margin before 330s kill

def construct():
    start_time = time.time()
    
    # Precompute all grid points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    n_points = len(all_points)
    
    xs = all_points[:, 0].astype(np.int32)
    ys = all_points[:, 1].astype(np.int32)
    
    # Compute squared distance matrix
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = (dx * dx + dy * dy).astype(np.int32)
    del dx, dy
    
    best_set = []
    best_score = 0
    
    np.random.seed(42)
    random.seed(42)
    
    n_restarts = 0
    while time.time() - start_time < TIME_LIMIT:
        n_restarts += 1
        # Create random order
        order = np.random.permutation(n_points)
        
        # Greedy construction
        S = []  # list of point indices
        dist_from = {}  # A -> set of dist_sq[A,B] for B in S, B != A
        
        for P in order:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            # Check if P can be added to S
            valid = True
            d_PA_values = []
            for A in S:
                d_PA = int(dist_sq[P, A])
                d_PA_values.append(d_PA)
                # Step 2: check if d_PA is in dist_from[A]
                if d_PA in dist_from.get(A, set()):
                    valid = False
                    break
            
            if valid:
                # Step 1: check if all d_PA_values are distinct
                if len(set(d_PA_values)) != len(d_PA_values):
                    valid = False
            
            if valid:
                # Add P to S
                S.append(P)
                dist_from[P] = set()
                for A in S:
                    if A == P:
                        continue
                    d = int(dist_sq[P, A])
                    dist_from[P].add(d)
                    dist_from[A].add(d)
        
        if len(S) > best_score:
            best_score = len(S)
            best_set = S.copy()
        
        # Early exit if we've used most of the time
        if time.time() - start_time > TIME_LIMIT - 10:
            break
    
    # Convert back to (x,y) pairs
    result = [(int(all_points[i][0]), int(all_points[i][1])) for i in best_set]
    return result
