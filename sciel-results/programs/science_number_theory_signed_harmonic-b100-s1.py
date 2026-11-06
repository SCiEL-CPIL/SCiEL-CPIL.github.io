"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
def construct():
    """
    Find 30 coefficients in {-1, 0, 1} such that sum(c[i]/(i+1)) is nonzero
    and has minimal absolute value.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # We represent the sum as an integer S = sum(c[i] * L / (i+1))
    # We want to minimize |S| for S != 0.
    # The actual value is S / L.
    
    # Precompute weights
    weights = np.array([L // (i + 1) for i in range(n)], dtype=np.int64)
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_sums(weights_half):
        """
        Generate all possible sums for a half using coefficients in {-1, 0, 1}.
        Returns a numpy array of sums.
        """
        # Number of combinations is 3^len(weights_half)
        # We can generate them iteratively.
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        
        for w in weights_half:
            # For each existing sum, we can add -w, 0, or +w
            # sums_new = [s-w for s in sums] + [s for s in sums] + [s+w for s in sums]
            # This triples the size.
            sums = np.concatenate([sums - w, sums, sums + w])
            
        return sums

    # Generate sums for both halves
    # 3^15 is ~14.3 million. This might use significant memory (14.3M * 8 bytes * 3 arrays during concat ~ 350MB peak?)
    # Let's be careful. 
    # 14.3M int64 is ~115MB.
    # Concatenating creates new arrays.
    # It should be fine on most evaluators.
    
    sums1 = generate_sums(w1)
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)
    
    best_abs_S = np.iinfo(np.int64).max
    best_coeff = None
    
    # We need to find pairs (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # Iterate through sums1.
    # To speed up, we can iterate through unique sums1 if there are duplicates, but there are likely few.
    # However, 14.3M iterations in Python loop is slow.
    # We can vectorize the search?
    # np.searchsorted can take an array of values.
    
    # Let's try vectorized search.
    # For each s1 in sums1, we want to find s2 in sums2_sorted close to -s1.
    # target = -sums1
    # indices = np.searchsorted(sums2_sorted, -sums1)
    
    # But 14.3M elements in sums1.
    # searchsorted is O(N log M). 14.3M * log(14.3M) ~ 14.3M * 24 ~ 340M ops.
    # NumPy handles this fast.
    
    # However, we need to check neighbors.
    # Let's do it in chunks to avoid memory issues or just let NumPy do it.
    
    targets = -sums1
    indices = np.searchsorted(sums2_sorted, targets)
    
    # We need to check indices and indices-1 (and maybe indices+1 if index is at boundary or value matches exactly?)
    # searchsorted returns the index where the value would be inserted to maintain order.
    # If the value exists, it returns the leftmost index.
    # So we should check:
    # 1. The element at `indices` (if valid)
    # 2. The element at `indices - 1` (if valid)
    
    # We also need to handle the case where s1 + s2 == 0. We want the smallest NON-ZERO.
    # So if s1 + s2 == 0, we should look for the next closest non-zero.
    # This complicates the vectorized approach slightly.
    
    # Let's calculate candidates:
    # cand_idx_right = indices
    # cand_idx_left = indices - 1
    
    # We must clip indices to valid range [0, len(sums2_sorted)-1]
    
    len_s2 = len(sums2_sorted)
    
    # Right candidates
    idx_r = np.clip(indices, 0, len_s2 - 1)
    s2_r = sums2_sorted[idx_r]
    total_r = sums1 + s2_r
    
    # Left candidates
    idx_l = np.clip(indices - 1, 0, len_s2 - 1)
    s2_l = sums2_sorted[idx_l]
    total_l = sums1 + s2_l
    
    # We have two candidate totals for each s1: total_r and total_l.
    # We want the min |total| for total != 0.
    
    # Mask out zeros
    mask_r = total_r != 0
    mask_l = total_l != 0
    
    # If both are zero, we need to look further?
    # If s1 + s2 == 0, it means there is a zero sum.
    # The next best is the closest non-zero.
    # This requires checking idx_r + 1 or idx_l - 1 if the primary candidate is 0.
    
    # Let's create an array of all candidate totals and their corresponding coefficients?
    # No, we just need the value first. Then we can reconstruct.
    
    # Let's combine candidates.
    # To be safe and simple, let's just evaluate the min absolute value among non-zero candidates.
    
    abs_total_r = np.abs(total_r)
    abs_total_l = np.abs(total_l)
    
    # Set zero candidates to infinity
    abs_total_r[~mask_r] = np.iinfo(np.int64).max
    abs_total_l[~mask_l] = np.iinfo(np.int64).max
    
    # Find the global minimum
    min_abs_r = np.min(abs_total_r)
    min_abs_l = np.min(abs_total_l)
    
    if min_abs_r < min_abs_l:
        best_abs_S = min_abs_r
        # Find the index in sums1 that produced this
        idx_best = np.argmin(abs_total_r)
        s1_val = sums1[idx_best]
        s2_val = s2_r[idx_best]
        # Reconstruct coefficients
        # We need to find which combination of coeffs in w1 sums to s1_val and w2 to s2_val.
        # This is a subset sum reconstruction problem.
        # We can do this by backtracking.
        
        # However, if both min_abs_r and min_abs_l are inf, we have a problem (all sums are 0? Impossible).
        
        # What if the true minimum was skipped because we only checked immediate neighbors?
        # searchsorted finds the insertion point. The closest value is either at idx or idx-1.
        # If the value at idx is 0, the next closest is either idx-1 or idx+1.
        # We checked idx-1 (as left candidate). But if idx-1 is also 0? Unlikely for distinct sums, but possible if duplicates.
        # If both idx and idx-1 give 0, we need to check idx+1.
        
        # Let's add a check for "if best is 0 or very large, check further neighbors".
        # Actually, if min_abs is inf, we failed.
        
    else:
        best_abs_S = min_abs_l
        idx_best = np.argmin(abs_total_l)
        s1_val = sums1[idx_best]
        s2_val = s2_l[idx_best]
        
    # If best_abs_S is still inf, we might have missed the closest non-zero due to the zero trap.
    # Let's do a small loop around the best candidate to ensure we find the true min non-zero.
    # We know the target s1_val.
    # Let's find all s2 in sums2_sorted close to -s1_val.
    
    target = -s1_val
    # Find indices of target in sums2_sorted
    # Since sums2_sorted might have duplicates, we want the range.
    left_idx = np.searchsorted(sums2_sorted, target, side='left')
    right_idx = np.searchsorted(sums2_sorted, target, side='right')
    
    # Check neighbors in a window around [left_idx, right_idx]
    # The closest non-zero sum will be at the boundaries of the zero-sum block or just outside.
    # Let's scan a small window around the insertion point.
    
    # We already have s1_val.
    # Let's just iterate s2 candidates from the sorted list around the target.
    
    # Get a window of indices
    # We want the smallest |s1 + s2| != 0.
    # This is equivalent to smallest |s2 - (-s1)| != 0.
    
    # Let's just check a range of indices around the insertion point.
    # The number of duplicates is likely small.
    
    start_scan = max(0, left_idx - 10)
    end_scan = min(len_s2, right_idx + 10)
    
    local_sums2 = sums2_sorted[start_scan:end_scan]
    local_totals = s1_val + local_sums2
    local_abs = np.abs(local_totals)
    
    # Mask zeros
    local_mask = local_totals != 0
    local_abs[~local_mask] = np.iinfo(np.int64).max
    
    if np.min(local_abs) < best_abs_S:
        best_abs_S = np.min(local_abs)
        # Update s2_val to the one that gave this min
        local_idx_min = np.argmin(local_abs)
        s2_val = local_sums2[local_idx_min]
        
    # Now we have s1_val and s2_val.
    # We need to reconstruct the coefficients.
    
    def reconstruct_coeffs(weights, target_sum):
        """
        Find coefficients in {-1, 0, 1} that sum to target_sum given weights.
        Uses simple backtracking/DP for small N (15).
        """
        n_w = len(weights)
        # Use a set to track reachable sums and store one path?
        # Or just DFS. 3^15 is too big for DFS if not pruned.
        # But we know the target is reachable.
        # We can use DP:
        # dp[i][s] = True if sum s is reachable using first i weights.
        # Range of s is [-sum(w), sum(w)].
        # Sum(w) for 15 terms is approx 1.4 * 10^12? No, weights are L/i.
        # L is 2.3 * 10^12. Sum is huge.
        # DP array will be too large.
        
        # Alternative: Since N=15 is small, we can just generate all sums again and map back?
        # We already generated sums. We can store the path?
        # Better: Just do a recursive search with pruning.
        # Or, simpler: We can just find ANY solution.
        # Let's use a dictionary mapping sum -> coeffs tuple.
        # This will use a lot of memory for 14M entries.
        
        # Let's try a greedy/heuristic approach? No, must be exact.
        
        # Let's use a BFS/DFS that stops when target is found.
        # Since the target is "close" to 0 (small relative to range), the search space might be small?
        # Not necessarily.
        
        # Let's just re-run the generation but keep track of parents?
        # Or, simpler:
        # We have the list of all sums. We can just find the index of target_sum in the generated list?
        # No, we didn't keep the coefficients.
        
        # Let's write a fast recursive solver for N=15.
        # It will visit at most 3^15 nodes in worst case, but often prunes.
        # Actually, 3^15 is 14M. A Python recursive call for 14M nodes is too slow.
        
        # We need a better way to reconstruct.
        # We can store the coefficients during the initial generation?
        # That would require storing 14M tuples of 15 ints. 14M * 15 * 4 bytes = 840MB. Too much.
        
        # Let's use the fact that we can solve the subset sum problem for N=15 using a meet-in-the-middle again?
        # Or just use a SAT solver? No.
        
        # How about we just iterate through the 14M sums again, but this time we compute the coefficients?
        # We can generate the coefficients in parallel with the sums?
        # We can store the "index" in the 3-ary representation?
        # Each sum corresponds to a unique combination of coefficients?
        # Not unique, different coeffs can give same sum.
        # But we just need ONE valid set of coefficients.
        
        # Let's modify generate_sums to also return the coefficients?
        # No, let's just do a second pass for the specific target.
        
        # We can use a dictionary: sum -> coeffs (first found).
        # Memory: 14M entries. Key: int (8 bytes), Value: tuple (overhead).
        # This is risky.
        
        # Alternative:
        # We can just use the `itertools.product` for 15 elements?
        # 14M iterations in Python is ~5-10 seconds.
        # We have 85s.
        # Let's try it.
        
        coeffs = [0] * n_w
        # We want sum(c[i]*w[i]) = target_sum
        
        # Use a recursive function with early exit.
        # To make it fast, sort weights descending?
        # Actually, let's just use a simple DFS.
        
        # Optimization: Precompute suffix sums to prune.
        # suffix_sums[i] = max possible sum from i to end.
        
        # Let's implement a fast iterative DFS or just use the product approach if time permits.
        # Given the deadline, a Python loop of 14M is borderline.
        
        # Let's try a smarter DP for reconstruction.
        # We only care about the target.
        # We can use a set of reachable sums for the first half of the 15 (i.e., 7-8 terms) and the second half.
        # Meet-in-the-middle for reconstruction!
        
        mid_r = n_w // 2
        w_r1 = weights[:mid_r]
        w_r2 = weights[mid_r:]
        
        # Generate sums for r1 and r2 with their coeffs
        def gen_with_coeffs(w):
            res = [(0, tuple([0]*len(w)))]
            for i, wi in enumerate(w):
                new_res = []
                for s, c in res:
                    # -1
                    nc1 = list(c)
                    nc1[i] = -1
                    new_res.append((s - wi, tuple(nc1)))
                    # 0
                    nc0 = list(c)
                    nc0[i] = 0
                    new_res.append((s, tuple(nc0)))
                    # 1
                    nc2 = list(c)
                    nc2[i] = 1
                    new_res.append((s + wi, tuple(nc2)))
                res = new_res
            return res
        
        # This will generate 3^7 ~ 2000 and 3^8 ~ 6500 entries. Very fast.
        sums_r1 = gen_with_coeffs(w_r1)
        sums_r2 = gen_with_coeffs(w_r2)
        
        # Create a map from sum to coeffs for r2
        map_r2 = {}
        for s, c in sums_r2:
            if s not in map_r2:
                map_r2[s] = c
        
        # Now find s1, s2 such that s1 + s2 = target_sum
        for s1, c1 in sums_r1:
            s2_needed = target_sum - s1
            if s2_needed in map_r2:
                c2 = map_r2[s2_needed]
                return list(c1) + list(c2)
                
        return None # Should not happen

    c1 = reconstruct_coeffs(w1, s1_val)
    c2 = reconstruct_coeffs(w2, s2_val)
    
    final_coeffs = c1 + c2
    
    # Verify
    # val = sum(c[i] / (i+1) for i in range(30))
    # assert val != 0
    
    return final_coeffs

# EVOLVE-BLOCK-END
