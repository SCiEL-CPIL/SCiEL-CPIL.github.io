import time
import random
import sys
from collections import Counter

N = 64
TIME_LIMIT = 290.0  # Safety margin

# Precompute grid points
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def construct():
    start_time = time.time()
    
    # State
    S = []
    dist_counts = Counter()
    
    # Helper to check if a point can be added to current S
    def can_add(point):
        # 1. Check distances to all points in S
        dists = []
        for p in S:
            d = dist2(point, p)
            dists.append(d)
            # Early exit: if this distance already exists in S's internal distances
            # Note: This checks d(C,P) != d(A,B) for any A,B in S
            if dist_counts.get(d, 0) > 0:
                return False
        
        # 2. Check if all distances from C to S are distinct
        # This ensures d(C, P1) != d(C, P2)
        if len(dists) != len(set(dists)):
            return False
            
        return True

    def add_point(point):
        for p in S:
            d = dist2(point, p)
            dist_counts[d] += 1
        S.append(point)

    def remove_point(point):
        # Remove point from S
        S.remove(point)
        # Update dist_counts
        for p in S:
            d = dist2(point, p)
            dist_counts[d] -= 1
            if dist_counts[d] == 0:
                del dist_counts[d]

    # 1. Greedy Initialization
    random.shuffle(GRID_POINTS)
    for p in GRID_POINTS:
        if time.time() - start_time > TIME_LIMIT * 0.1:
            break
        if can_add(p):
            add_point(p)
            
    best_size = len(S)
    
    # 2. Local Search
    # Strategy: Remove a random point, try to add a new one, then try to add another.
    # This allows "swapping" to unlock better configurations.
    
    while time.time() - start_time < TIME_LIMIT:
        # Try a move
        if not S:
            break
            
        # Pick a random point to remove
        idx = random.randint(0, len(S) - 1)
        r_point = S[idx]
        
        # Save state
        S_backup = S.copy()
        dc_backup = dist_counts.copy()
        
        # Remove r_point
        remove_point(r_point)
        
        # Try to find a replacement point C
        found_c = False
        c_point = None
        
        # Try random candidates
        # To speed up, we can try a few random points
        for _ in range(50):
            if time.time() - start_time > TIME_LIMIT:
                break
            cand = random.choice(GRID_POINTS)
            if cand in S:
                continue
            if can_add(cand):
                found_c = True
                c_point = cand
                break
        
        if found_c:
            # Add C
            add_point(c_point)
            
            # Now try to add another point D to increase size
            found_d = False
            d_point = None
            
            for _ in range(20):
                if time.time() - start_time > TIME_LIMIT:
                    break
                cand = random.choice(GRID_POINTS)
                if cand in S:
                    continue
                if can_add(cand):
                    found_d = True
                    d_point = cand
                    break
            
            if found_d:
                add_point(d_point)
                if len(S) > best_size:
                    best_size = len(S)
            else:
                # Size same, but configuration changed
                pass
        else:
            # Restore state
            S[:] = S_backup
            dist_counts.update(dc_backup)
            # Note: dist_counts is a Counter, update might not clear zeros if not careful, 
            # but we deleted zeros in remove_point. 
            # Safer to just reset from backup
            dist_counts = dc_backup

    return [(p[0], p[1]) for p in S]

if __name__ == "__main__":
    # Just to allow local testing
    pts = construct()
    print(f"Found {len(pts)} points")
