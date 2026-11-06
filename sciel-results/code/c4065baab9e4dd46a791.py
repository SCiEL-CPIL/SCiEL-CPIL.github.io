import numpy as np
import time
import random

N = 64
TIME_LIMIT = 280.0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5.0
    
    # Grid points
    xs, ys = np.meshgrid(np.arange(N), np.arange(N))
    points = np.column_stack((xs.ravel(), ys.ravel()))
    total_points = N * N
    
    # Precompute all pairwise squared distances
    # diff[i, j] = points[i] - points[j]
    # dist_sq[i, j] = ||points[i] - points[j]||^2
    # This is (64*64)^2 = 16,777,216 floats. ~134 MB. Acceptable.
    # But computing it naively might be slow or memory heavy.
    # Let's compute distance matrix D where D[i,j] is squared dist.
    
    # Vectorized distance calculation
    # points: (4096, 2)
    # We want D[i,j] = (xi-xj)^2 + (yi-yj)^2
    # D = sum(points^2, axis=1)[:,None] + sum(points^2, axis=1)[None,:] - 2 * points @ points.T
    pts_sq_norms = np.sum(points**2, axis=1)
    D = pts_sq_norms[:, None] + pts_sq_norms[None, :] - 2 * (points @ points.T)
    # Ensure non-negative due to floating point errors (though integers are exact here)
    # Since coords are int, D should be integer. But float64 is used.
    # For exactness, we can cast to int if we are sure, but float is safer for general.
    # However, isosceles check is D[i,j] == D[i,k] or D[i,j] == D[j,k] or D[i,k] == D[j,k].
    
    # Precompute "Isosceles Pairs" matrix?
    # Two points i, j form a potential isosceles apex for a third point k if:
    # 1. D[i,k] == D[j,k] (k is on perpendicular bisector)
    # 2. D[i,j] == D[i,k] (k is on circle centered at i)
    # 3. D[i,j] == D[j,k] (k is on circle centered at j)
    
    # Strategy: Simulated Annealing
    # State: binary vector S of length total_points (1 if point is in set)
    # Initial state: Greedy construction using vectorized checks?
    # Or just random sparse start?
    
    # Let's do a fast greedy initialization first to get a good starting point.
    # Greedy: Iterate points in random order. If adding point p doesn't create isosceles, add it.
    # Check for new point p against existing set E:
    # For each e in E:
    #   For each e2 in E:
    #     Check if (p, e, e2) is isosceles.
    # This is still O(K^2) per insertion.
    
    # Alternative: Start with empty set, use SA to add points?
    # SA moves:
    # 1. Add a random point p not in set. Accept if valid.
    # 2. Remove a random point p in set. Always accept? No, might lower score.
    # 3. Swap: Remove p_in, Add p_out. Accept if valid (or with probability if invalid? No, must be valid).
    
    # Since validity is hard constraint, we can only move to valid states.
    # So it's a random walk on the graph of valid sets.
    # To increase size, we need to find a valid set of size K+1 from size K.
    # This is hard if K is maximal.
    
    # Better approach: 
    # Use the "No-Isosceles" constraint to precompute forbidden pairs for each point?
    # No, the constraint is on triples.
    
    # Let's try a different vectorized heuristic:
    # Randomly select a subset of points, check validity, keep if valid.
    # Optimize size via hill climbing on swaps.
    
    # Hill Climbing with Swaps:
    # Start with a valid set S of size K.
    # Try to swap p_in (in S) with p_out (not in S) to get S' = S \ {p_in} U {p_out}.
    # Check if S' is valid.
    # If valid, accept.
    # This maintains size K. It doesn't increase size.
    
    # To increase size, we need to find a "hole" where a new point can be added.
    # Or, we can try to add a point to S \ {p_in}. If S \ {p_in} U {p_new} is valid, size is K.
    # Wait, if we remove 1 and add 1, size is constant.
    
    # How to grow?
    # Start with small set.
    # Try to add a point. If fail, try removing one point from current set, then adding two?
    # Or just restart greedy with different random orders?
    
    # The previous program did 100 greedy restarts.
    # Let's do a smarter initialization:
    # 1. Generate a random permutation.
    # 2. Greedily add points.
    # 3. Once maximal, try "local search" to swap points to free up space for larger points?
    # No, swapping keeps size constant.
    
    # Key Insight from Inspiration: "Vectorized algorithm... precomputes distances... enabling much faster local search"
    # And "structured initialization and aggressive swap-based local search may escape local optima better"
    
    # Let's implement a valid-swap hill climber that also tries to grow.
    # Actually, if we are at a local maximum of size K, we can't just add.
    # We need to find a set of size K+1.
    
    # Algorithm:
    # 1. Fast Greedy Init (vectorized check for new point against existing set).
    #    Check: For new point p, and existing set E (indices).
    #    D[p, E] gives distances.
    #    For any e1, e2 in E:
    #       Is D[p, e1] == D[p, e2]? (Apex p)
    #       Is D[p, e1] == D[e1, e2]? (Apex e1)
    #       Is D[p, e2] == D[e1, e2]? (Apex e2)
    #    
    #    Vectorized check for adding p to E:
    #    d_pe = D[p, E]  # (K,)
    #    D_EE = D[np.ix_(E, E)] # (K, K) - this is expensive to slice every time.
    #    
    #    Let's precompute D as a full matrix.
    #    To check if p can be added to E:
    #    1. d_pe = D[p, E]
    #    2. Check if any d_pe[i] == d_pe[j] for i!=j. (Apex p)
    #       This is equivalent to len(set(d_pe)) < K.
    #    3. Check if any d_pe[i] == D[E[i], E[j]] for i!=j. (Apex E[i] or E[j])
    #       We need to check if the set of distances from p to E intersects with the set of distances within E?
    #       No, specific pairs.
    #       For each i in E:
    #          Does d_pe[i] exist in D[E[i], E] (excluding i)?
    #          Or does d_pe[i] exist in D[E, E[i]] (excluding i)?
    #          Actually, D is symmetric.
    #          So check if d_pe[i] is in D[E[i], E] (excluding self).
    
    #    This is still O(K^2) or O(K * log K) per insertion if done carefully.
    
    #    Let's stick to the restart strategy but make the greedy check faster using NumPy.
    
    best_size = 0
    best_set_indices = np.array([], dtype=int)
    
    # Precompute D
    # D is 4096x4096 float64.
    
    def greedy_construct(random_order):
        current_indices = []
        current_indices_set = set()
        
        for p_idx in random_order:
            if time.time() > deadline:
                break
            
            # Check if p_idx can be added
            if len(current_indices) < 2:
                current_indices.append(p_idx)
                current_indices_set.add(p_idx)
                continue
            
            curr_arr = np.array(current_indices)
            
            # Distances from p_idx to all in current
            d_p_curr = D[p_idx, curr_arr]
            
            # Check 1: Duplicate distances from p_idx (Apex at p_idx)
            if len(np.unique(d_p_curr)) < len(curr_arr):
                continue
            
            # Check 2 & 3: 
            # For each i in current, check if d_p_curr[i] matches any distance from curr_arr[i] to other points in curr_arr.
            # D[curr_arr, curr_arr] is KxK.
            # We need to check if d_p_curr[i] is in D[curr_arr[i], curr_arr] (excluding i).
            
            # This is a bit tricky to vectorize fully without loops over K.
            # But K is small (maybe 50-100?).
            # Let's try a loop over current points, using vectorized inner check.
            
            valid = True
            for i, c_idx in enumerate(current_indices):
                d_pc = d_p_curr[i]
                # Distances from c_idx to all other points in current
                # Exclude c_idx itself (distance 0)
                # d_c_others = D[c_idx, curr_arr]
                # We need to check if d_pc is in d_c_others, but not at the position corresponding to c_idx.
                
                # Since we already checked d_p_curr for duplicates, d_pc is unique in d_p_curr.
                # But d_pc might match a distance between two existing points.
                
                # Let's get all pairwise distances in current set.
                # This is O(K^2) to compute if not cached.
                
                # Optimization: Maintain a set of all pairwise squared distances in the current set.
                # If d_pc is in that set, it might be invalid?
                # No, it's invalid only if the pair forming that distance is (p, c) or (c, c2) or (p, c2).
                # We already know d_p_curr has no duplicates. So d_pc != d_p_c2.
                # So we only need to check if d_pc == d_c_c2 for some c2 in current.
                # This is equivalent to: is d_pc in the set of distances from c to other points in current?
                
                # Let's maintain a list of points.
                # For each c in current, we need to check if d_pc is in {D[c, c2] for c2 in current, c2 != c}.
                
                # We can precompute the set of all pairwise distances in the current set?
                # If d_pc is in the global set of pairwise distances of current, it's a candidate.
                # But we need to ensure the pair is (c, c2) or (c2, c).
                # Actually, if d_pc == d_c_c2, then (p, c, c2) is isosceles with apex c.
                # If d_pc == d_c2_c, same thing.
                # If d_pc == d_c1_c2 where c1, c2 != c, that doesn't directly involve p and c?
                # Wait. The triangle is (p, c, c2).
                # Sides: d_pc, d_c_c2, d_p_c2.
                # We know d_pc != d_p_c2 (from check 1).
                # So we only need to check if d_pc == d_c_c2 OR d_p_c2 == d_c_c2.
                # We are iterating c. So we check d_pc == d_c_c2.
                # What about d_p_c2 == d_c_c2?
                # That would be caught when we iterate c2?
                # If we iterate all c in current, and for each c we check if d_pc == d_c_c2 for any c2, 
                # do we cover the case where apex is c2?
                # Triangle (p, c, c2).
                # If apex is c2, then d_p_c2 == d_c_c2.
                # When we iterate c, we check d_pc == d_c_c2.
                # When we iterate c2, we check d_p_c2 == d_c2_c.
                # So yes, iterating all points in current and checking against distances to other points covers all cases.
                
                # So:
                # For each c in current:
                #   Let d_target = D[p, c]
                #   Let d_others = D[c, current] (excluding c)
                #   If d_target is in d_others, invalid.
                
                # To speed this up, we can precompute the set of all pairwise distances in the current set?
                # No, because we need to know if the distance originates from c.
                
                # Let's just do the loop. K is likely < 200.
                # 200 * 200 = 40,000 ops. Fast in NumPy?
                # D[c, curr_arr] is a vector of length K.
                # Checking if d_target is in that vector (excluding self) is O(K) or O(1) with set.
                
                # Let's maintain a list of current indices.
                # And maybe a cached D_curr?
                
                # For now, simple loop:
                c_idx = c_idx
                d_c_to_all = D[c_idx, curr_arr]
                # Exclude self distance (which is 0). 
                # Since d_pc > 0 (p not in set), 0 is not a match.
                # But we must ensure we don't match d_pc with D[c, c] = 0.
                # Since d_pc > 0, it won't match 0.
                # So we can just check if d_pc is in d_c_to_all.
                
                # However, d_c_to_all contains D[c, c] = 0.
                # It also contains D[c, c2] for all c2.
                # If d_pc == D[c, c2], invalid.
                
                # Use np.isin? Or set?
                # Set is faster for small K.
                # But creating a set every time is O(K).
                # np.isin is O(K log K) or similar.
                
                # Let's try:
                if np.any(d_c_to_all == d_pc):
                    valid = False
                    break
            
            if valid:
                current_indices.append(p_idx)
                current_indices_set.add(p_idx)
                
        return current_indices

    # Run multiple greedy passes
    # To make it faster, we can try to limit the number of passes or optimize the inner loop.
    # The inner loop is the bottleneck.
    
    # Alternative: Use the "Swap" strategy to improve a greedy result.
    # But first, let's see if we can get a good greedy result.
    
    # Let's try 20 greedy passes.
    for i in range(20):
        if time.time() > deadline:
            break
            
        random_order = np.random.permutation(total_points)
        curr = greedy_construct(random_order)
        
        if len(curr) > best_size:
            best_size = len(curr)
            best_set_indices = np.array(curr)
            
    # Local Search: Swaps
    # Try to swap points in best_set_indices with points not in the set.
    # This keeps size constant.
    # To increase size, we need a different move.
    
    # Let's try a "Grow" move:
    # Pick a point p_in to remove.
    # Try to add TWO points p1, p2 such that (S \ {p_in}) U {p1, p2} is valid.
    # This is complex.
    
    # Instead, let's just run more greedy passes if time permits.
    # The randomization is key.
    
    # Convert best_set_indices to list of tuples
    if len(best_set_indices) == 0:
        return []
        
    best_points = points[best_set_indices]
    return [(int(x), int(y)) for x, y in best_points]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# Ensure it is the entry point.
# EVOLVE-BLOCK-END

# Re-define construct to be the one above? 
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `def construct():`.
# I have defined `def construct():` above.
# I should make sure there are no duplicate definitions.

# I will wrap the logic in a single construct function as requested by the block structure, 
# or just ensure the global construct is the one used.

# The code above defines construct globally.
# I will remove the duplicate definition if any.

# Final check:
# 1. N=64
# 2. D matrix precomputed.
# 3. Greedy with vectorized inner checks (semi-vectorized).
# 4. Multiple restarts.

# One optimization: 
# In greedy_construct, `curr_arr` is created in every iteration of the outer loop.
# This is O(K) allocation. 
# We can maintain `curr_indices` as a list and only convert to array when needed for D slicing.
# D[p_idx, curr_arr] requires curr_arr to be an array.
# So we need the array.
# But we can update the array incrementally? No, numpy arrays are fixed size.
# We can use a list for indices, and only convert to array when checking.
# This is what I did.

# Is there a faster way to check?
# Precompute for each point p, the set of points q that form an isosceles triangle with p and ANY other point?
# No, depends on the set.

# Let's stick to this. It should be faster than pure Python loops.

# One more thing: The `np.any(d_c_to_all == d_pc)` check includes the distance to itself (0).
# Since d_pc > 0, it's safe.
# But what if d_pc is 0? p is not in set, so p != c, so d_pc > 0.

# Let's add a time check inside the inner loop?
# No, K is small.

# Let's refine the greedy function to be slightly more efficient by reusing the array where possible?
# Not easily.

# I will output the code now.
