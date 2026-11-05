"""Greedy local search for low discrepancy on arithmetic progressions."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    # Initialize with a pseudo-random sequence
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    # Helper to compute the max discrepancy and identify a "worst" AP to fix
    # We don't compute ALL APs every time (too slow). We sample or check specific problematic ones.
    # However, for a robust greedy step, we need to know which sign flip reduces the global max.
    # A full check of all APs is O(n^2 * log n) roughly? No, number of APs is ~ n^2/2.
    # Summing each is O(n/d). Total work ~ n^2/2 * n/d? No.
    # Let's estimate cost of evaluating discrepancy for one configuration.
    # Number of APs: sum_{d=1}^{n} (n-d) * (1) starts? 
    # For step d, starts 1..n-d. Lengths 1..(n-s)/d.
    # Total number of distinct APs is roughly n^2/2.
    # If we compute sum for each, it's too slow for many iterations.
    
    # Strategy: 
    # 1. Precompute all APs as index lists? Memory heavy but n=256 is small.
    #    Max APs ~ 256^2/2 = 32768.
    #    Storing them as arrays of indices.
    # 2. For each AP, maintain current sum.
    # 3. When flipping sign at index i, update sums of all APs containing i.
    #    How many APs contain i?
    #    For each step d, i is part of APs starting at i-k*d for k>=0 such that i-k*d >= 1.
    #    And ending at i+j*d for j>=0 such that i+j*d <= n.
    #    Actually, an AP is defined by (start, step). 
    #    Does i belong to AP (s, d)? Yes if i = s + k*d for some k>=0.
    #    So s = i - k*d.
    #    For a fixed i and d, the possible starts s are i, i-d, i-2d, ... >= 1.
    #    So there are roughly i/d APs with step d containing i.
    #    Sum over d=1..n of i/d ~ i * log n.
    #    So flipping one sign requires updating ~ 256 * log(256) ~ 2000 AP sums.
    #    This is fast enough for many iterations.
    
    # Precomputation:
    # Generate all APs.
    # ap_indices: list of lists of indices (0-based)
    # ap_sums: np.array of current sums for each AP
    # ap_len: lengths (not strictly needed for sum, but good for debug)
    
    # To speed up updates, we need to know which APs contain index i.
    # ap_to_indices: list of sets or lists of AP IDs for each index i.
    
    ap_list = []
    index_to_aps = [set() for _ in range(n)]
    
    # Generate all APs
    # Start s (0-based: 0 to n-1), Step d (1 to n-s)
    # Wait, the problem says "any start, positive step and length including one".
    # An AP is defined by start s, step d, and length L.
    # Usually "arithmetic progression" in discrepancy problems refers to the set of points {s, s+d, ...}.
    # The sum is over the points in the AP.
    # Does "length" imply we consider sub-APs? 
    # "Max absolute signed sum on ANY arithmetic progression contained in [1,256]"
    # Usually this means the sum over the entire defined progression.
    # If the progression is defined by (s, d), it usually extends until it hits the boundary.
    # OR does it mean ANY contiguous segment of an AP?
    # "contained in [1,256]" usually means the AP itself must be within bounds.
    # "any start, positive step and length" suggests we can choose the length.
    # This implies we must check ALL sub-sequences of APs? 
    # i.e. For any s, d, and any k1 < k2, the sum over {s+k1*d, ..., s+k2*d}?
    # If so, the number of such "progressions" is much larger.
    # Let's re-read carefully: "minimize the maximum absolute sum on any arithmetic progression... including arbitrary starts and singletons".
    # Standard Erdos discrepancy problem D(n) usually considers homogeneous progressions {x, x+d, ...} fully contained.
    # The prompt says "including arbitrary starts and singletons". Singletons are APs of length 1.
    # If we consider *any* length, we are looking at max subarray sum on each "diagonal" (fixed step d).
    # For a fixed step d, the indices form chains: 1, 1+d, 1+2d...; 2, 2+d...; etc.
    # On each chain, we want the max absolute subarray sum.
    # This is equivalent to: For each d, for each chain c, compute max subarray sum.
    # 
    # Let's verify the count.
    # For d=1, one chain: 1..256. Max subarray sum.
    # For d=2, two chains: 1,3..255 and 2,4..256. Max subarray sum for each.
    # Total work to evaluate discrepancy:
    # Sum over d=1..n of (number of chains for d) * (length of chain).
    # Number of chains for d is d (starts 1..d).
    # Lengths are roughly n/d.
    # So work per d is d * (n/d) = n.
    # Total work is sum_{d=1}^n n = n^2 = 256^2 = 65536 operations.
    # This is VERY fast. We can evaluate the full discrepancy in < 1ms.
    #
    # So we can do a simple local search:
    # 1. Compute current max discrepancy D.
    # 2. For each index i, try flipping signs[i].
    # 3. Compute new discrepancy D'.
    # 4. If D' < D, accept flip.
    # 5. Repeat until no improvement or time limit.
    #
    # Optimization: We don't need to recompute ALL D' from scratch.
    # But since D computation is O(n^2) = 65k ops, and we have 256 indices,
    # one full pass of trying all flips is 256 * 65k = 16.6 million ops.
    # Python is slow, ~10M ops/sec? Maybe 1-2 seconds per full pass.
    # We have 85 seconds. So we can do ~40-80 full passes.
    # This should be sufficient to get a very low discrepancy.
    
    def compute_max_discrepancy(signs):
        """
        Computes max absolute sum over any contiguous subsequence of any AP.
        Equivalent to: for each step d, for each start s in 1..d,
        find max absolute subarray sum in the sequence signs[s-1], signs[s-1+d], ...
        """
        max_disc = 0
        n = len(signs)
        for d in range(1, n + 1):
            # For this step d, we have d chains
            for s in range(1, d + 1):
                # Chain starts at index s-1 (0-based)
                # Elements: s-1, s-1+d, s-1+2d, ...
                # We need max subarray sum on this 1D array.
                # We can extract the subarray or process on the fly.
                # On the fly is faster.
                curr_sum = 0
                max_ending_here = 0
                max_so_far = 0
                # Also need min for negative side? 
                # "Max absolute sum" means max(|sum|).
                # So we need max(subarray_sum) and min(subarray_sum).
                # max_abs = max(max_sum, -min_sum)
                
                # Kadane's for max
                curr_max = 0
                max_max = 0
                # Kadane's for min
                curr_min = 0
                max_min = 0 # Actually we want min_min (most negative)
                min_min = 0
                
                idx = s - 1
                while idx < n:
                    val = signs[idx]
                    # Max update
                    curr_max = max(val, curr_max + val)
                    if curr_max > max_max:
                        max_max = curr_max
                    # Min update
                    curr_min = min(val, curr_min + val)
                    if curr_min < min_min:
                        min_min = curr_min
                    idx += d
                
                chain_max_abs = max(max_max, -min_min)
                if chain_max_abs > max_disc:
                    max_disc = chain_max_abs
        return max_disc

    # Initial evaluation
    current_disc = compute_max_discrepancy(signs)
    
    # Local search loop
    # We will try to flip signs.
    # To avoid oscillation, we can keep track of best state.
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Shuffle indices to avoid bias
    indices = list(range(n))
    rng.shuffle(indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in indices:
            # Try flipping i
            old_val = signs[i]
            signs[i] = -old_val
            
            new_disc = compute_max_discrepancy(signs)
            
            if new_disc < current_disc:
                current_disc = new_disc
                if new_disc < best_disc:
                    best_disc = new_disc
                    best_signs = signs.copy()
                improved = True
            else:
                # Revert
                signs[i] = old_val
        
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # We can try a few random restarts or random perturbations if time permits.
        if not improved:
            # Try a random perturbation: flip 5 random signs and continue
            if time.time() < deadline - 5.0:
                flip_indices = rng.sample(range(n), 5)
                for i in flip_indices:
                    signs[i] = -signs[i]
                current_disc = compute_max_discrepancy(signs)
                if current_disc < best_disc:
                    best_disc = current_disc
                    best_signs = signs.copy()
                # Continue searching from this new state
                improved = True # Force another pass
            else:
                break

    # Return the best signs found
    return list(best_signs)

# EVOLVE-BLOCK-END
