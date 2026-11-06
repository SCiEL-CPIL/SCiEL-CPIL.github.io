"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _fast_greedy(arr, limit, deadline):
    """
    Greedy construction: iterate through candidates in order 'arr'.
    Maintain a count array of sums.
    If adding a candidate keeps all sum counts <= limit, accept it.
    """
    n = 1000
    max_sum = 2000
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    chosen = []
    
    # Precompute indices for speed? 
    # We need to check if adding x causes any count to exceed limit.
    # New sums are x + y for y in chosen, and 2*x.
    # We can just compute the max count in the new sums.
    
    # Optimization: 
    # If we store chosen in a list, iterating over it is O(k).
    # Total O(N * K).
    
    for x in arr:
        if time.time() > deadline:
            break
        
        # Calculate new counts
        # Sums: x + y for y in chosen, and 2*x
        
        # We need to check:
        # 1. Is count[2*x] + 1 <= limit? (Actually count[2*x] + 1, since 2*x is a new sum)
        # 2. For each y in chosen, is count[x+y] + 1 <= limit?
        
        # Note: if x is already in chosen, 2*x is counted. 
        # But we are adding x, so 2*x is a new representation.
        # Wait, the representation a+b with a<=b.
        # If we add x, the new pairs are (x, y) for y in chosen where y >= x? 
        # No, the set is just a set of integers. 
        # The condition is on the set S. 
        # If S' = S U {x}, the new pairs are (x, y) for y in S such that x <= y? 
        # No, the pair is (a, b) with a <= b.
        # If we add x, the new pairs involving x are (x, y) where x <= y and y in S.
        # AND (x, x) which is 2x.
        
        # However, the greedy algorithm in the seed simply checks:
        # new = [x + y for y in chosen] + [2 * x]
        # This implies it considers ALL y in chosen, regardless of order.
        # But wait, if y < x, then (y, x) is the pair. 
        # Since the set is unordered, adding x creates pairs (x, y) for ALL y in S.
        # The representation count for sum s increases by 1 for each y in S such that x+y = s.
        # Plus 1 for 2x.
        
        # So yes, we just need to check counts[x+y] and counts[2*x].
        
        # Efficient check:
        # max_new_count = 1 # for 2x
        # if counts[2*x] + 1 > limit: reject
        # for y in chosen:
        #     if counts[x+y] + 1 > limit: reject
        
        # This is O(K) per candidate.
        
        # Can we speed this up?
        # If we use FFT, we can compute all counts at once, but we only add one element.
        # The change in counts is sparse (K+1 positions).
        # So O(K) is optimal for a single step.
        
        # Let's implement the O(K) check carefully.
        
        # 1. Check 2x
        if counts[2 * x] + 1 > limit:
            continue
            
        # 2. Check x + y
        # We need to check if any counts[x+y] + 1 > limit.
        # i.e., counts[x+y] >= limit.
        
        # Vectorized check:
        if chosen:
            # chosen is a list of ints. Convert to numpy array for fast addition?
            # Or just loop. 
            # K is up to ~1000. 1000 * 1000 = 1e6 ops.
            # Numpy overhead might be higher than a tight Python loop for small K?
            # Let's try Numpy.
            c_arr = np.array(chosen, dtype=np.int32)
            sums = x + c_arr
            # Check if any count at these sums is >= limit
            # Note: sums might have duplicates? No, y are distinct.
            # But x+y could be same for different y? No, y distinct -> x+y distinct.
            # So we just check the counts at these indices.
            if np.any(counts[sums] >= limit):
                continue
        
        # Accept
        chosen.append(x)
        counts[2 * x] += 1
        if chosen:
            # Update counts for x+y
            # We already have c_arr? 
            # c_arr was computed from chosen BEFORE appending x.
            # So c_arr contains the old chosen.
            # We need to update counts[x+y] for y in old chosen.
            # Also, we need to update counts for future steps.
            # Wait, the loop above used `chosen` (old).
            # So `sums` are correct for the update.
            counts[sums] += 1
            
        # Note: if chosen was empty, c_arr is empty, loop skipped.
        # But we still need to update counts[2*x]. Done above.
        # And for future, chosen has x.
        
    return chosen

def _local_search(arr, limit, deadline, time_per_step=0.01):
    """
    Take a valid set, try to improve by swapping or moving elements.
    Simplified: Try to add elements that were rejected, or swap existing with rejected.
    """
    n = 1000
    max_sum = 2000
    chosen = list(arr)
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Rebuild counts
    for x in chosen:
        for y in chosen:
            if x <= y:
                counts[x + y] += 1
        # Wait, this is O(K^2). 
        # Better to rebuild from scratch using the same logic as greedy?
        # Or just use the counts from the greedy run if passed?
        # For now, rebuild.
        
    # Actually, let's just use the greedy result and try to "fix" or "expand" it.
    # A simple local search:
    # 1. Try to add any element from 1..1000 not in chosen.
    # 2. If success, keep.
    # 3. If fail, try swapping an existing element with a new one.
    
    # This is complex. Let's stick to multiple greedy restarts with different orders.
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_size = 0
    best_set = []
    
    # Strategy: Multiple greedy runs with different candidate orders.
    # 1. Ascending
    # 2. Descending
    # 3. Random shuffles
    # 4. Specific patterns (primes, etc.)
    
    orders = []
    
    # 1. Ascending
    orders.append(list(range(1, 1001)))
    
    # 2. Descending
    orders.append(list(range(1000, 0, -1)))
    
    # 3. Random shuffles
    # How many? Depends on speed.
    # Greedy is ~0.1s per run (estimated). 
    # 110s / 0.1s = 1100 runs.
    # Let's do as many as possible.
    
    # Pre-generate a few random orders
    rng = np.random.default_rng(42)
    
    # We will run in a loop
    run_count = 0
    
    while time.time() < deadline:
        # Determine order
        if run_count == 0:
            order = orders[0]
        elif run_count == 1:
            order = orders[1]
        else:
            # Random shuffle
            order = rng.permutation(range(1, 1001)).tolist()
            
        result = _fast_greedy(order, 2, deadline)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
        run_count += 1
        
        # If we've done a lot of random runs, maybe break?
        # No, just keep going until deadline.
        
        # Optimization: if best_size is very high, maybe stop?
        # But we don't know the max.
        
    return best_set

# EVOLVE-BLOCK-END
