import time
from math import lcm
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute integer weights w_i = L / i
    # Sum S = sum(c_i * 1/i) = (sum(c_i * L/i)) / L
    # We want to minimize |sum(c_i * w_i)| != 0
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    # First half: indices 0..14 (terms 1/1 to 1/15)
    # Second half: indices 15..29 (terms 1/16 to 1/30)
    half1_weights = weights[:15]
    half2_weights = weights[15:]
    
    # Generate all possible sums for the first half
    # Each coefficient is in {-1, 0, 1}
    # Total combinations: 3^15 = 14,348,907
    # This is manageable in Python if optimized, but might be slow.
    # Let's estimate: 14M iterations. In Python, ~10-20ms per million simple ops?
    # Actually, generating the list and sorting might take a few seconds.
    
    def generate_sums(w_list):
        sums = [0]
        for w in w_list:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    # Check time
    if time.time() - start_time > deadline - 5:
        # Fallback if LCM calculation or setup took too long (unlikely)
        return [0] * 28 + [1, -1]
        
    t0 = time.time()
    sums1 = generate_sums(half1_weights)
    t1 = time.time()
    sums2 = generate_sums(half2_weights)
    t2 = time.time()
    
    # Sort sums2 for binary search
    sums2.sort()
    
    best_diff = float('inf')
    best_c1 = None
    best_c2 = None
    best_s1_val = None
    best_s2_val = None
    
    # To recover coefficients, we need to store the path or re-derive it.
    # Storing paths for 14M entries is memory heavy (14M * 15 bytes ~ 210MB+).
    # Alternative: Store just the sum, and if we find a good candidate, 
    # we can reconstruct the coefficients by checking which subset of weights 
    # sums to that value? That's NP-hard again.
    # Better: Store the coefficients as a tuple or integer mask?
    # 3^15 states. We can encode the state as an integer base-3 number.
    # Or simply store the list of coefficients for each sum.
    # 14M lists of 15 ints is too much memory.
    
    # Let's use a different approach for reconstruction:
    # Since we just need *a* solution, and MITM gives us the optimal sum value,
    # we can find the coefficients by backtracking or by storing the 
    # "last choice" for each sum if we do DP, but MITM enumeration is flat.
    
    # Revised Plan:
    # 1. Enumerate (sum, coeff_tuple) for half 1. 
    #    Memory: 14M tuples of 15 ints. Too much.
    # 2. Enumerate sums only for half 2, sort them.
    # 3. Enumerate half 1 sums one by one (or in batches), for each s1, 
    #    binary search in sums2 for -s1.
    #    If we find s2 such that |s1+s2| is minimal and non-zero, we have the value.
    #    But we need the coefficients!
    
    # We must recover coefficients.
    # Strategy:
    # Keep a dictionary for half 2: sum -> one example coeff tuple.
    # 14M entries in a dict. Keys are ints, values are tuples of 15 small ints.
    # Memory: ~14M * (key overhead + value overhead). Might be ~1-2 GB. Risky.
    
    # Alternative:
    # Since we only need the GLOBAL minimum, and the distribution of sums is dense,
    # the minimum non-zero difference is likely very small (like 1/L or 2/L).
    # If we find a difference of 1 (in integer units), we stop.
    
    # Let's try to store the coefficients for half 2 in a dict, but limit the size?
    # No, we need the specific s2 that matches s1.
    
    # Let's try a more memory efficient MITM:
    # Store sums2 sorted.
    # For reconstruction, we can use a "meet in the middle" reconstruction trick:
    # If we find optimal s1 and s2, we can find the coefficients for s1 by 
    # solving the subset sum for half1 weights to equal s1? 
    # That's still hard.
    
    # Better: Store the coefficients for half 1 in a list of tuples, and sums in a list.
    # Same for half 2.
    # 14M * 15 * 28 bytes (python int overhead) is huge.
    
    # Let's use NumPy or arrays?
    # Actually, 3^15 is 14.3M.
    # Let's try to just store the sums for half 2, and for half 1, 
    # we iterate and check. If we find the best pair (s1, s2), we need to 
    # find *any* coeff vector for s1 and *any* for s2.
    
    # We can find a coeff vector for a target sum S using a simple recursive 
    # search with pruning, because 15 variables is small enough for a 
    # targeted search if we have the target.
    
    def find_coeffs_for_sum(weights, target_sum):
        # Recursive search for coefficients in {-1,0,1} summing to target_sum
        # weights length 15.
        # Use memoization or simple DFS.
        # Since 3^15 is 14M, a single DFS might be slow if it explores the whole space.
        # But we can prune: 
        # Max possible sum from remaining weights.
        
        n_w = len(weights)
        suffix_max = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + weights[i]
            
        def dfs(idx, current_sum):
            if idx == n_w:
                if current_sum == target_sum:
                    return []
                return None
            
            remaining_max = suffix_max[idx]
            # Pruning: if current_sum + remaining_max < target_sum or 
            # current_sum - remaining_max > target_sum, return None
            
            if current_sum + remaining_max < target_sum or current_sum - remaining_max > target_sum:
                return None
                
            # Try -1
            res = dfs(idx + 1, current_sum - weights[idx])
            if res is not None:
                return [-1] + res
                
            # Try 0
            res = dfs(idx + 1, current_sum)
            if res is not None:
                return [0] + res
                
            # Try 1
            res = dfs(idx + 1, current_sum + weights[idx])
            if res is not None:
                return [1] + res
                
            return None
        
        return dfs(0, 0)

    # This DFS is efficient because of the strong pruning by suffix sums.
    # The weights are distinct and large enough relative to the target usually.
    
    # Now, let's find the optimal s1, s2.
    # We generate sums1 and sums2 lists.
    # To save memory, we don't store coeffs for all 14M sums.
    # We just store the sums.
    
    # sums1: list of 14M ints
    # sums2: list of 14M ints, sorted
    
    # Iterate through sums1. For each s1, we want s2 such that s1+s2 is closest to 0 but not 0.
    # Binary search in sums2 for -s1.
    
    # Optimization:
    # We can stop early if we find a difference of 1 (in integer units).
    # The minimal non-zero integer difference is 1.
    
    best_abs_int_diff = float('inf')
    target_s1 = 0
    target_s2 = 0
    
    # To speed up, we can sample sums1? No, we need the global min.
    # But 14M * log(14M) ~ 14M * 24 ~ 300M ops. In Python this is too slow (minutes).
    
    # We need a faster way to find the closest pair between two lists.
    # Merge sort style?
    # We have sums1 (unsorted) and sums2 (sorted).
    # If we sort sums1 too, we can do a linear merge to find min |s1+s2|.
    # Sorting 14M ints in Python: ~10-20 seconds?
    # Let's try.
    
    sums1.sort()
    
    # Now we have two sorted lists.
    # We want min |a + b| for a in sums1, b in sums2, a+b != 0.
    # Standard two-pointer technique?
    # Since we want a+b close to 0, we can use a two-pointer approach if we 
    # traverse one in ascending and one in descending order.
    
    i = 0
    j = len(sums2) - 1
    found_one = False
    
    # We need to be careful: a+b can be positive or negative.
    # We want the smallest absolute value.
    
    # Actually, since the sums are dense, the minimum difference is likely 1.
    # We can just check if there exists a pair summing to 1 or -1.
    
    # Let's do a full scan with two pointers to find the global min.
    # But wait, the two-pointer technique for "closest sum to target" works 
    # if we fix the target. Here target is 0.
    
    # Pointer i starts at 0 (min sums1), Pointer j starts at len-1 (max sums2).
    # Sum = sums1[i] + sums2[j].
    # If sum > 0, we need a smaller sum, so decrement j (since sums2 is sorted asc, 
    # wait, if we want smaller sum, we should pick smaller b, so j--? 
    # No, if sum > 0, we want to reduce sum. 
    # sums1[i] is small, sums2[j] is large. 
    # To reduce sum, we can decrease sums2[j] by moving j to a smaller index? 
    # Yes, j-- gives smaller b. 
    # Or increase sums1[i]? No, i++ gives larger a, which increases sum.
    # So if sum > 0, we must decrease the total. 
    # We can either increase i (bad, increases sum) or decrease j (good, decreases sum).
    # Wait, if i increases, a increases. If j decreases, b decreases.
    # So if sum > 0, we should decrease b (j--).
    # If sum < 0, we should increase a (i++).
    
    # This finds the closest pair to 0?
    # Let's trace.
    # This standard algorithm finds the pair with sum closest to 0.
    
    min_abs = float('inf')
    best_a = 0
    best_b = 0
    
    i = 0
    j = len(sums2) - 1
    
    while i < len(sums1) and j >= 0:
        s_val = sums1[i] + sums2[j]
        if s_val != 0:
            abs_val = abs(s_val)
            if abs_val < min_abs:
                min_abs = abs_val
                best_a = sums1[i]
                best_b = sums2[j]
                if min_abs == 1:
                    # Can't get better than 1 (integer)
                    break
            
        if s_val > 0:
            j -= 1
        elif s_val < 0:
            i += 1
        else:
            # s_val == 0, invalid. Move both to avoid same pair?
            # We need to find the NEXT best.
            # If we just move one, we might miss the next best on the other side.
            # But since we are looking for non-zero, we can move both? 
            # Actually, if s_val == 0, any other combination with same i or j 
            # will have larger absolute value? Not necessarily.
            # If sums1[i] + sums2[j] == 0, then sums1[i] + sums2[j+1] > 0 and 
            # sums1[i-1] + sums2[j] < 0.
            # We should check neighbors.
            # For simplicity, if we hit 0, we move both pointers to explore new territory.
            i += 1
            j -= 1
            
    if min_abs == float('inf'):
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Now we have best_a (sum for half 1) and best_b (sum for half 2).
    # We need to find coefficients for best_a using half1_weights.
    # And coefficients for best_b using half2_weights.
    
    c1 = find_coeffs_for_sum(half1_weights, best_a)
    c2 = find_coeffs_for_sum(half2_weights, best_b)
    
    if c1 is None or c2 is None:
        # This shouldn't happen if best_a/b were generated from these weights
        return [0] * 28 + [1, -1]
        
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        # Should not happen due to s_val != 0 check
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
