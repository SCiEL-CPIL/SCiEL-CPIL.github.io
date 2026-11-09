"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 64
TIME_LIMIT = 280  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_pts = []
    best_len = 0
    rng = random.Random(42)
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def dist2(a, b):
        return (a[0] - b[0])**2 + (a[1] - b[1])**2
    
    def try_add(p, pts, dist_cache):
        """Try to add point p to pts. Returns True if valid, updates dist_cache."""
        k = len(pts)
        # Distances from p to all existing points
        dp = []
        for q in pts:
            dp.append(dist2(p, q))
        
        # Check p as apex: all distances from p must be distinct
        if len(set(dp)) != k:
            return False
        
        # For each existing point A_i as apex: dist(A_i, p) must not equal dist(A_i, A_j) for j != i
        for i in range(k):
            di = dp[i]
            Ai = pts[i]
            for j in range(k):
                if j == i:
                    continue
                if dist2(Ai, pts[j]) == di:
                    return False
        
        return True
    
    def greedy_construct(order, start_time, time_limit):
        """Greedy construction following a given order of candidate points."""
        pts = []
        # For efficiency, we'll just recompute distances each time
        for p in order:
            if time.time() - start_time > time_limit:
                break
            if try_add(p, pts, None):
                pts.append(p)
        return pts
    
    # Strategy 1: Multiple random restarts with shuffled orders
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Create a shuffled order
        order = all_points[:]
        rng.shuffle(order)
        
        pts = greedy_construct(order, start_time, TIME_LIMIT - 10)
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
    
    # Strategy 2: Local improvement - try to add more points by trying all candidates
    # on the best solution found
    if best_pts and time.time() - start_time < TIME_LIMIT - 5:
        improved = best_pts[:]
        # Try adding each candidate point not already in the set
        existing_set = set(improved)
        candidates = [p for p in all_points if p not in existing_set]
        rng.shuffle(candidates)
        
        for p in candidates:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            if try_add(p, improved, None):
                improved.append(p)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_pts = improved
    
    # Strategy 3: Another round of local improvement with different shuffle
    if best_pts and time.time() - start_time < TIME_LIMIT - 3:
        improved = best_pts[:]
        existing_set = set(improved)
        candidates = [p for p in all_points if p not in existing_set]
        rng.shuffle(candidates)
        
        for p in candidates:
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            if try_add(p, improved, None):
                improved.append(p)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_pts = improved
    
    # Ensure we have at least the baseline
    if not best_pts:
        # Fallback: simple 1D Sidon-like construction
        xs, used = [], set()
        x = 0
        while x < N:
            diffs = {abs(x - y) for y in xs}
            if not (diffs & used):
                used |= diffs
                xs.append(x)
            x += 1
        best_pts = [(x, 0) for x in xs]
    
    return best_pts
# EVOLVE-BLOCK-END
