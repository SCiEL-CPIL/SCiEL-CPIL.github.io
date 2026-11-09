import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    n_points_total = 2 ** d
    # Time limit check
    start_time = time.time()
    timeout = 590  # 10s margin
    
    # Precompute conflicts for each point in {0,1}^11
    # Conflict if Hamming distance is 1 or 2.
    # We represent points as integers 0..2047.
    
    # Bit operations for Hamming distance
    def hamming_dist(a, b):
        return bin(a ^ b).count('1')
        
    # Precompute conflict lists
    # This takes O(N^2) which is 4M operations. Fast enough.
    conflicts = [[] for _ in range(n_points_total)]
    for i in range(n_points_total):
        for j in range(i + 1, n_points_total):
            if hamming_dist(i, j) <= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    # Function to check if a point can be added to a set
    # S is a set of ints
    def can_add(point, S):
        for c in conflicts[point]:
            if c in S:
                return False
        return True

    # Initial set: Shortened Hamming Code
    # We can generate a random maximal independent set and then try to improve
    # Or start with a known good set.
    # Let's try a greedy construction with random ordering, repeated.
    
    best_set = set()
    best_size = 0
    
    # Helper to get a greedy independent set given an order
    def greedy_independent_set(order):
        S = set()
        for p in order:
            if can_add(p, S):
                S.add(p)
        return S

    # Run multiple greedy passes with different random seeds
    # And then try to improve with local search (add/remove/swap)
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        # Random permutation
        order = list(range(n_points_total))
        random.shuffle(order)
        
        # Greedy phase
        current_set = greedy_independent_set(order)
        
        # Local search phase: try to add points, if fail, try to swap
        # Simple local search: 
        # 1. Try to add any point not in set.
        # 2. If none can be added, try to swap a point in set with a point not in set 
        #    such that the size doesn't decrease (impossible for swap) 
        #    or we just try to find a larger set by restarting.
        
        # Actually, a better local search for Max Independent Set:
        # Try to add points. If stuck, remove a random point from the set 
        # and try to add 2 points that were blocked by it?
        
        # Let's use a simpler strategy: 
        # If current_size > best_size, update best.
        # Then try to improve current_set by trying to add all non-members.
        # If we can add any, we do it.
        # If we can't add any, we are in a local maximum.
        # We can try to "kick" by removing a random element and re-greedy-ing the rest?
        
        # For now, just collect best from many greedy runs.
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # If we hit 128 or more, we are doing well.
        # We can stop early if we find a very large set?
        # Theoretical max is ~140-170.
        
        # To speed up, we can break if we've done enough iterations
        # and aren't improving.
        if best_size >= 140:
             break

    # Convert best_set to points
    # Map integer to vector in {1, -1}
    # 0 -> 1, 1 -> -1 (or vice versa, doesn't matter for distance)
    
    points = np.zeros((best_size, d), dtype=np.int64)
    for i, p in enumerate(best_set):
        for j in range(d):
            # Check bit j
            if (p >> j) & 1:
                points[i, j] = -1
            else:
                points[i, j] = 1
                
    # Verify validity
    # Max norm squared
    max_norm_sq = 0
    for p in points:
        ns = np.sum(p * p)
        if ns > max_norm_sq:
            max_norm_sq = ns
            
    # Min dist squared
    min_dist_sq = float('inf')
    k = best_size
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            ds = np.sum(diff * diff)
            if ds < min_dist_sq:
                min_dist_sq = ds
                
    if min_dist_sq <= max_norm_sq:
        # Fallback to seed if invalid (should not happen)
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        
    return points
