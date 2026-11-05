import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Define lattice operations
    def dist_sq(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        da = a1 - a2
        db = b1 - b2
        return da*da + db*db + da*db

    def neighbors(p):
        a, b = p
        return [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]

    # Generate H_2 (19 points)
    def in_hex(a, b, r):
        return max(abs(a), abs(b), abs(a + b)) <= r

    core = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if in_hex(a, b, 2):
                core.append((a, b))
    
    # Generate candidate pool: H_4 points (or H_3 + some H_4)
    # Let's use H_4 to be safe, but prioritize H_3
    pool = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if in_hex(a, b, 4):
                pool.append((a, b))
                
    pool_set = set(pool)
    core_set = set(core)
    
    # Initial set: Core + 13 points from H_3 boundary (H_3 \ H_2)
    # H_3 points
    h3_points = [p for p in pool if in_hex(p[0], p[1], 3) and p not in core_set]
    
    # Greedily add 13 points from h3_points to core to maximize edges
    current_set = set(core)
    
    # Count edges in current_set
    def count_edges(s):
        edges = 0
        pts = list(s)
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if dist_sq(pts[i], pts[j]) == 1:
                    edges += 1
        return edges

    best_edges = count_edges(current_set)
    
    # Add 13 points
    candidates = h3_points[:]
    random.shuffle(candidates) # Randomize to break symmetry
    
    for _ in range(13):
        if time.time() > deadline - 5:
            break
        best_candidate = None
        best_gain = -1
        # Try all remaining candidates
        for p in candidates:
            # Calculate gain
            gain = 0
            for q in neighbors(p):
                if q in current_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = p
        
        if best_candidate:
            current_set.add(best_candidate)
            candidates.remove(best_candidate)
            best_edges += best_gain
        else:
            break

    # Local Search: Swap points
    # Candidate pool for swaps: H_4 points not in current_set
    # To keep it fast, restrict to H_3 and H_4 boundary
    swap_pool = [p for p in pool if p not in current_set]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Iterate over points in current_set
        for p in list(current_set):
            if time.time() > deadline - 10:
                break
            # Try swapping p with each q in swap_pool
            # To speed up, only consider q near p? Or all?
            # swap_pool is ~61 - 32 = 29 points. Small.
            for q in swap_pool:
                # Check if swap improves edges
                # Edges lost: neighbors of p in current_set (excluding q if q was in set, but q is not)
                lost = 0
                for r in neighbors(p):
                    if r in current_set:
                        lost += 1
                
                # Edges gained: neighbors of q in current_set (excluding p if p was in set, but p is being removed)
                # Note: current_set still contains p for the "gained" calculation?
                # We should calculate gain based on (current_set - {p})
                gained = 0
                for r in neighbors(q):
                    if r in current_set and r != p:
                        gained += 1
                
                if gained > lost:
                    current_set.remove(p)
                    current_set.add(q)
                    swap_pool.remove(q)
                    swap_pool.append(p)
                    improved = True
                    break # Restart loop for this p? Or just continue?
                    # Breaking inner loop and restarting outer loop is safer for consistency
                    # But we can just continue to next p if we don't restart
                    # Let's restart the while loop to be safe
                    break
            
            if improved:
                break

    # Convert to list of [a, b]
    result = [[a, b] for a, b in current_set]
    return result
