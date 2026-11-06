"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _precompute_dist2():
    """Precompute squared distances between all pairs of grid points."""
    points = list(itertools.product(range(N), repeat=2))
    V = len(points)
    dist2 = np.zeros((V, V), dtype=np.int64)
    for i in range(V):
        xi, yi = points[i]
        for j in range(i + 1, V):
            xj, yj = points[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist2[i, j] = d
            dist2[j, i] = d
    return points, dist2

# Precompute at module load time to save time during construct()
POINTS, DIST2 = _precompute_dist2()
V = len(POINTS)

def _check_isosceles_with_new_point(new_idx, selected_indices):
    """
    Check if adding new_idx to selected_indices creates any isosceles triangle.
    An isosceles triangle is formed if any two distances from the new point to 
    existing points are equal, or if the distance between two existing points 
    equals the distance from one of them to the new point.
    
    Actually, simpler: For any pair (a, b) in selected, the triangle (new, a, b) is isosceles if:
    d(new, a) == d(new, b)
    d(new, a) == d(a, b)
    d(new, b) == d(a, b)
    """
    # Optimization: If we maintain the property that no isosceles triangles exist in 'selected',
    # we only need to check triangles involving the new point.
    # Let S be the set of selected indices.
    # For each a in S, d(new, a) must be distinct from d(new, b) for all b in S, b != a.
    # Also, for each a in S, d(new, a) must not equal d(a, b) for any b in S, b != a.
    
    # Let's check the first condition: all distances from new to S must be distinct.
    # And second condition: d(new, a) != d(a, b) for all a, b in S, a != b.
    
    # We can do this efficiently.
    # 1. Check if any d(new, a) == d(new, b) for a != b in S.
    # 2. Check if any d(new, a) == d(a, b) for a != b in S.
    
    # Note: If S is small, this is fast. If S grows, it becomes O(|S|^2).
    # But we are trying to keep S small (minimal maximal).
    
    if len(selected_indices) < 2:
        return True # No triangle possible yet
    
    # Get indices
    S = list(selected_indices)
    
    # Check condition 1: d(new, a) distinct for all a in S
    # We can just compute distances and check for duplicates.
    d_new_to_S = {}
    for a in S:
        d = DIST2[new_idx, a]
        if d in d_new_to_S:
            return False
        d_new_to_S[d] = a
        
    # Check condition 2: d(new, a) != d(a, b) for all a, b in S, a != b
    # This is equivalent to: for each a in S, d(new, a) is not in the set of distances 
    # from a to any other point in S.
    # Pre-computing distances within S is expensive if done from scratch.
    # However, since we add points one by one, we can maintain a set of "forbidden distances" 
    # or just check on the fly.
    # Given |S| is expected to be small (maybe < 20), O(|S|^2) is fine.
    
    for a in S:
        d_na = DIST2[new_idx, a]
        for b in S:
            if a == b:
                continue
            d_ab = DIST2[a, b]
            if d_na == d_ab:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # 10s buffer for safety
    
    # Start with a small valid set. 
    # Any 2 points form a valid set (no triangle).
    # We want a MAXIMAL set. 
    # Strategy: Randomized greedy construction with backtracking or restarts?
    # Or: Start with 2 points, try to add random points. If stuck, restart.
    # To find a *small* maximal set, we want to get stuck early.
    # The greedy row-major method gets stuck at size ~20-30.
    # Can we get smaller?
    # A maximal set means every other point forms an isosceles triangle with some pair in the set.
    
    best_size = V # Initialize with worst case
    best_set = []
    
    # We will run multiple randomized restarts.
    # In each restart, we build a set greedily but with random order or random choices.
    # Actually, a better approach for "minimal maximal" is often to use a heuristic 
    # that prioritizes points that "block" many others, but here we want to ADD points 
    # such that the set becomes maximal as quickly as possible.
    
    # Let's try a different angle: 
    # Start with empty set.
    # At each step, pick a random candidate from the remaining valid points.
    # If no valid points remain, the set is maximal.
    # Record the size.
    # Repeat many times.
    
    # Optimization: Precompute for each point, the set of points that form an isosceles 
    # triangle with it and any other point? No, that's complex.
    
    # Let's stick to the random greedy approach but optimize the check.
    
    for restart in range(5000):
        if time.time() > deadline:
            break
            
        # Random permutation of points
        order = list(range(V))
        random.shuffle(order)
        
        current_set = []
        current_set_set = set()
        
        for idx in order:
            # Check if idx can be added
            if _check_isosceles_with_new_point(idx, current_set_set):
                current_set.append(idx)
                current_set_set.add(idx)
                
                # Early termination check: 
                # If the set is already large, it's unlikely to beat the best.
                # But we don't know the max size until we finish.
                # However, if len(current_set) > best_size, we can break early 
                # because adding more points only increases the size.
                # Wait, we want MINIMAL maximal. 
                # So if we already have more points than the best known minimal maximal set, 
                # we can abort this restart.
                if len(current_set) > best_size:
                    break
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we find a very small set, we might want to stop early?
            # Theoretical lower bound? 
            # For N=16, area is 256. 
            # If we get down to 10 or 12, that's a big improvement over ~25.
            if best_size <= 8:
                break
                
    # Convert indices back to points
    return [POINTS[i] for i in best_set]

# EVOLVE-BLOCK-END
