"""Optimized construction for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _max_ap_discrepancy(signs):
    """
    Compute the maximum absolute signed sum over any arithmetic progression 
    contained in [1, n].
    signs: list of +1/-1, length n. Index i corresponds to position i+1.
    Returns: max absolute sum.
    """
    n = len(signs)
    # For each step d, we can iterate through the sub-sequences defined by that step.
    # An AP is defined by start s, step d, length L.
    # Sum = sum(signs[s-1 + k*d] for k in range(L))
    # We need max |Sum| over all valid s, d, L.
    
    max_abs = 0
    # d ranges from 1 to n-1 (step 0 is not a valid AP step in this context usually, 
    # but "arithmetic progression" usually implies distinct terms, so d>=1. 
    # If d=0, it's just a single element repeated, which is just the element itself. 
    # The prompt says "any start, positive step", so d >= 1.
    # Length L >= 1.
    
    # Optimization: 
    # For a fixed step d, the indices form chains: 
    # 0, d, 2d, ...
    # 1, 1+d, 1+2d, ...
    # ...
    # d-1, d-1+d, ...
    # Within each chain, we just need to find the max absolute subarray sum? 
    # No, the AP must be contiguous in the chain. Yes, an AP with step d is a contiguous 
    # subsequence of one of these chains.
    # So for each d, we split indices 0..n-1 into d chains. 
    # For each chain, compute max absolute subarray sum.
    # The global max is the max over all d and all chains.
    
    # However, iterating d from 1 to n-1 and then processing chains is O(n^2) roughly.
    # n=256, so n^2 = 65536. This is very fast.
    
    for d in range(1, n):
        # Process d chains
        # Chain j starts at j, j+d, j+2d...
        for j in range(d):
            # Extract chain values
            # We can just iterate and maintain a running sum for Kadane's algorithm 
            # modified for absolute max?
            # Actually, max absolute subarray sum is max(max_subarray_sum, -min_subarray_sum).
            
            # Let's just extract the list for this chain if it's short, or process on the fly.
            # Chain length is at most ceil(n/d).
            
            # To avoid creating lists, we can just iterate.
            # But Kadane's is easier with a list or careful in-place tracking.
            # Given n=256, creating small lists is negligible overhead.
            
            chain = [signs[i] for i in range(j, n, d)]
            
            # Find max subarray sum (Kadane)
            max_ending_here = 0
            max_so_far = 0
            min_ending_here = 0
            min_so_far = 0
            
            for val in chain:
                max_ending_here = max(val, max_ending_here + val)
                max_so_far = max(max_so_far, max_ending_here)
                
                min_ending_here = min(val, min_ending_here + val)
                min_so_far = min(min_so_far, min_ending_here)
            
            current_max_abs = max(max_so_far, -min_so_far)
            if current_max_abs > max_abs:
                max_abs = current_max_abs
                # Early exit? If max_abs reaches n, it can't get worse. 
                # But we are minimizing, so we just track the max.
                
    return max_abs

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # Evaluate initial discrepancy
    current_disc = _max_ap_discrepancy(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    # Local Search: Simulated Annealing or Hill Climbing
    # We will use a simple hill-climbing with random restarts if stuck, 
    # but given the time budget, a single long run with occasional jumps is better.
    # Let's use a temperature-based approach or just greedy flips with random tie-breaking.
    
    # Greedy Hill Climbing: 
    # 1. Pick a random index i.
    # 2. Flip signs[i].
    # 3. Calculate new discrepancy.
    # 4. If new_disc < current_disc, accept.
    # 5. If new_disc == current_disc, accept with some probability (to avoid local minima).
    # 6. If new_disc > current_disc, reject (or accept with very low prob).
    
    # To speed up, we can try to be smarter about which index to flip.
    # But random is simple and robust.
    
    t = 10.0  # Initial temperature
    t_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randint(0, n - 1)
        
        # Flip
        old_val = signs[i]
        signs[i] = -old_val
        
        # Evaluate
        new_disc = _max_ap_discrepancy(signs)
        
        if new_disc < current_disc:
            # Improvement
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        elif new_disc == current_disc:
            # Neutral move, accept to explore
            current_disc = new_disc
        else:
            # Worse, accept with probability exp(-delta / T)
            delta = new_disc - current_disc
            # Since discrepancy is integer, delta >= 1
            prob = math.exp(-delta / t)
            if rng.random() < prob:
                current_disc = new_disc
            else:
                # Reject, revert
                signs[i] = old_val
        
        # Cool down
        t *= cooling
        if t < t_min:
            t = t_min
            
    return best_signs
# EVOLVE-BLOCK-END
