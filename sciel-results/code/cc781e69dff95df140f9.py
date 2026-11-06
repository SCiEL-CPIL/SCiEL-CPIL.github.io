import numpy as np
import random
import time

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Generate all grid points
    points = np.array([(x, y) for x in range(N) for y in range(N)])
    
    best_set = []
    best_size = 0
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.seed(num_restarts)
        
        # Shuffle points
        indices = np.arange(len(points))
        random.shuffle(indices)
        candidates = points[indices]
        
        S = []
        # D_A: list of sets of squared distances from A to other points in S
        # When S is empty, D_A is empty.
        # We need to update D_A when adding a point.
        
        # For efficiency, we might just recompute checks?
        # No, incremental is better.
        
        # Actually, let's just use the "All Distances Distinct" first to see if it's fast enough and gives a good score.
        # If "All Distances Distinct" gives ~100, and relaxed gives ~200, the relaxed is better.
        # But "All Distances Distinct" is much simpler to check vectorized.
        # Let's try "All Distances Distinct" first. It's a safe bet for a jump from 8 to 100+.
        
        used_dists = set()
        current_set = []
        
        for p in candidates:
            # Check if time is up
            if time.time() > deadline - 5:
                break
            
            if not current_set:
                current_set.append(p)
                continue
                
            # Calculate squared distances to all points in current_set
            # p is shape (2,), current_set is list of (2,)
            # Convert to numpy for speed
            ps = np.array(current_set)
            diffs = ps - p
            d_sq = np.sum(diffs**2, axis=1)
            
            # Check if any d_sq is in used_dists
            # And check if d_sq has duplicates
            if len(set(d_sq)) < len(d_sq):
                continue # Duplicate distances among new edges
                
            valid = True
            for d in d_sq:
                if d in used_dists:
                    valid = False
                    break
            
            if valid:
                current_set.append(p)
                used_dists.update(d_sq)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
        # If we found a very large set, we might want to stop or continue
        # Let's continue to try to beat it.
        
    return [(int(x), int(y)) for x, y in best_set]
