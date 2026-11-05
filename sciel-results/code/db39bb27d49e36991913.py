import time
import random
import numpy as np

N = 64

def _get_seed_set():
    """Replicate the seed program's logic to ensure a baseline."""
    xs, used = [], set()
    x = 0
    while x < N:
        diffs = {abs(x - y) for y in xs}
        if not (diffs & used):
            used |= diffs
            xs.append(x)
        x += 1
    return [(x, 0) for x in xs]

def construct():
    start_time = time.time()
    deadline = 280.0  # Safety margin
    
    # Initialize with seed set
    best_set = _get_seed_set()
    best_size = len(best_set)
    
    # List of all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    # Try different random seeds and orderings
    # We will run multiple rounds of randomized greedy
    
    attempts = 0
    while time.time() - start_time < deadline:
        attempts += 1
        
        # Create a random permutation of points
        # Using random.shuffle on a list of indices or points
        points = all_points.copy()
        random.shuffle(points)
        
        current_set = []
        # dist_sets[i] contains the set of squared distances from current_set[i] to all other points in current_set
        dist_sets = []
        
        for P in points:
            x, y = P
            valid = True
            new_dists = []
            
            # Check against all existing points
            for i, Q in enumerate(current_set):
                dx = x - Q[0]
                dy = y - Q[1]
                d2 = dx*dx + dy*dy
                
                # If this distance is already used by Q for another point, Q would be an apex
                if d2 in dist_sets[i]:
                    valid = False
                    break
                
                new_dists.append(d2)
            
            if not valid:
                continue
            
            # Check if P would be an apex: all distances from P to existing points must be distinct
            if len(set(new_dists)) != len(new_dists):
                continue
            
            # Valid, add P to current_set
            current_set.append(P)
            
            # Update dist_sets for existing points
            for i in range(len(current_set) - 1):
                dist_sets[i].add(new_dists[i])
            
            # Add dist set for the new point P
            dist_sets.append(set(new_dists))
        
        # Update best if we found a larger set
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
            # If we found a very good set, we might want to try to extend it?
            # For now, just restart.
            
        # Early exit if we hit a theoretical limit? No, we don't know it.
        # But if we find 50+ points, it's very good.
        if best_size > 50:
            break

    return best_set
