import random
import time
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute all points as a numpy array
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)

# Pre-compute squared distances as a numpy array
# DIST[i, j] = squared distance between point i and point j
# We compute this vectorized for speed
xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# Convert to a list of tuples for the return value
ALL_POINTS = [tuple(p) for p in points_arr]

def construct():
    start_time = time.time()
    deadline = start_time + 260.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # We use a randomized greedy approach with local search (swap improvements)
    # The key insight is that we need to check isosceles constraints efficiently.
    
    # For a set S, adding point k is invalid if:
    # 1. There exist a, b in S such that d(k,a) == d(k,b)  [k is apex]
    # 2. There exist a, b in S such that d(k,a) == d(a,b)  [a is apex, k and b are equal distance from a]
    # 3. There exist a, b in S such that d(k,b) == d(a,b)  [b is apex, k and a are equal distance from b]
    
    # Note: cases 2 and 3 are symmetric. Case 1 means k sees two points at the same distance.
    
    def is_valid_addition(k, S_indices):
        m = len(S_indices)
        if m == 0:
            return True
        if m == 1:
            return True  # Two points never form a triangle
        
        # Check 1: duplicate distances from k
        seen = set()
        for a in S_indices:
            d = DIST[k, a]
            if d in seen:
                return False
            seen.add(d)
        
        # Check 2 & 3: for each pair (a, b) in S, check if d(k,a)==d(a,b) or d(k,b)==d(a,b)
        # We can restructure: for each a in S, for each b in S where b != a:
        #   if d(k,a) == d(a,b): invalid
        # This covers both cases since we iterate all ordered pairs.
        
        # More efficient: precompute dists from k
        # Then for each pair (a,b), we need d_ab, d_ka, d_kb
        
        # Let's just do the double loop but with numpy-friendly access
        for i in range(m):
            a = S_indices[i]
            d_ka = DIST[k, a]
            for j in range(i + 1, m):
                b = S_indices[j]
                d_ab = DIST[a, b]
                d_kb = DIST[k, b]
                if d_ka == d_ab or d_kb == d_ab:
                    return False
        return True
    
    # We'll do multiple random restarts with greedy construction + local search
    indices = list(range(GRID_SIZE))
    
    while time.time() < deadline:
        random.shuffle(indices)
        current_set = []
        
        for k in indices:
            if time.time() > deadline:
                break
            if is_valid_addition(k, current_set):
                current_set.append(k)
        
        # Local search: try to replace one point with another to gain size
        # Actually, local search for "no isosceles" is tricky because removing and re-adding
        # might not help directly. Let's try a simple improvement:
        # Try to add points that were rejected, or swap.
        
        # For now, just track best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # If we have time, try local search on current_set
        if time.time() < deadline - 10:
            # Try to improve by attempting to insert unselected points
            unselected = [k for k in indices if k not in set(current_set)]
            random.shuffle(unselected)
            
            improved = False
            for k in unselected:
                if time.time() > deadline - 5:
                    break
                if is_valid_addition(k, current_set):
                    current_set.append(k)
                    improved = True
            
            if improved and len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
    
    return [ALL_POINTS[i] for i in best_set]
