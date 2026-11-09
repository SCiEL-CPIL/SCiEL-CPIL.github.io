"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 30
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # We want to minimize |sum(c_i * (L/i))| != 0
    # c_i in {-1, 0, 1}
    
    # Strategy: 3-Way Meet-in-the-Middle (MITM)
    # Split 30 terms into three sets of 10.
    # 3^10 = 59,049 combinations per set. This is very fast to generate.
    # We generate all sums for each set.
    # Then we find the triplet (s1, s2, s3) from the three sets such that
    # |s1 + s2 + s3| is minimized and non-zero.
    
    # To make the search efficient:
    # 1. Generate all sums for Set 1 and Set 2.
    # 2. Combine them into a single sorted list of sums S12 = {s1 + s2 | s1 in S1, s2 in S2}.
    #    Size of S12 is 3^10 * 3^10 = 3^20 = 3,486,784,401. 
    #    Wait, 3^20 is too large to store in memory (3.4 billion integers ~ 27GB+).
    #    
    # Let's re-evaluate the split.
    # 2-Way MITM with 15/15:
    # 3^15 = 14,348,907.
    # Storing 14M integers is ~112MB. This is feasible.
    # Storing 14M tuples of coefficients is NOT feasible.
    #
    # How to reconstruct coefficients in 2-Way MITM?
    # We can store the sums in a list.
    # When we find the best pair (s1, s2), we need to find the coefficients.
    # We can re-run the generation for the first half, but this time we can use a dictionary
    # mapping sum -> one valid coefficient tuple, BUT only for the specific target sum?
    # No, we don't know the target sum until we do the search.
    #
    # Alternative:
    # Store the sums in a list.
    # Also store a "proof" or "path" for each sum?
    # We can encode the path (coefficients) into the index or a parallel structure?
    # 
    # Let's try a different reconstruction method:
    # 1. Generate sums1 (list of 14M ints).
    # 2. Generate sums2 (list of 14M ints).
    # 3. Find best (s1, s2).
    # 4. To find coeffs for s1:
    #    We can split the first 15 terms into 7 and 8.
    #    Generate all sums for the 7 terms (3^7 = 2187).
    #    Generate all sums for the 8 terms (3^8 = 6561).
    #    Now we have a small MITM problem: find c7, c8 such that sum7 + sum8 = s1.
    #    This is very fast.
    #    We can store the coefficients for the small sets because they are small.
    #
    # This seems robust and fast.
    
    def generate_sums_with_coeffs(indices):
        """
        Generates all sums for the given indices.
        Returns a list of (sum, coeffs_tuple).
        This is only used for small sets (<= 10 terms).
        """
        sums = [(0, ())]
        for idx in indices:
            val = L // idx
            new_sums = []
            for s, c in sums:
                # -1
                new_sums.append((s - val, c + (-1,)))
                # 0
                new_sums.append((s, c + (0,)))
                # 1
                new_sums.append((s + val, c + (1,)))
            sums = new_sums
        return sums

    def generate_sums_only(indices):
        """
        Generates all sums for the given indices.
        Returns a list of sums.
        Used for larger sets (15 terms).
        """
        sums = [0]
        for idx in indices:
            val = L // idx
            # Optimization: use list comprehensions
            sums = [s - val for s in sums] + sums + [s + val for s in sums]
        return sums

    # Split indices 1..30 into two halves of 15
    half1 = list(range(1, 16))
    half2 = list(range(16, 31))
    
    # Generate sums for half 1
    sums1 = generate_sums_only(half1)
    
    if time.time() > deadline - 20:
        # Fallback if generation is too slow (unlikely)
        return [0] * 28 + [1, -1]
        
    sums1.sort()
    
    # Generate sums for half 2
    sums2 = generate_sums_only(half2)
    
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
        
    # Find best pair (s1, s2) such that |s1 + s2| is minimized and non-zero
    best_found_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    # Iterate over sums2. To speed up, we can iterate over the smaller unique values?
    # No, just iterate. 14M iterations.
    # Bisect is C-speed.
    for s2 in sums2:
        if time.time() > deadline - 5:
            break
            
        target = -s2
        pos = bisect.bisect_left(sums1, target)
        
        # Check pos and pos-1
        candidates = []
        if pos < len(sums1):
            candidates.append(sums1[pos])
        if pos > 0:
            candidates.append(sums1[pos-1])
            
        for s1 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_found_val:
                best_found_val = abs_total
                best_s1 = s1
                best_s2 = s2
                
                if best_found_val == 1:
                    break
        if best_found_val == 1:
            break
            
    if best_s1 is None:
        # Should not happen
        return [0] * 28 + [1, -1]

    # Now reconstruct coefficients for best_s1 and best_s2
    # We use the "small MITM" trick for each half.
    
    def find_coeffs_for_target(target, indices):
        """
        Finds coefficients for the given indices that sum to target.
        Uses MITM on the indices themselves.
        """
        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]
        
        # Generate sums with coeffs for left and right
        # Since len(indices) <= 15, mid <= 8.
        # 3^8 = 6561. This is small.
        left_sums = generate_sums_with_coeffs(left_idx)
        right_sums = generate_sums_with_coeffs(right_idx)
        
        # Create a dict for right sums: {sum: coeffs}
        # If multiple coeffs give the same sum, we just keep one.
        right_dict = {}
        for s, c in right_sums:
            if s not in right_dict:
                right_dict[s] = c
                
        # Search for match
        for s_l, c_l in left_sums:
            needed = target - s_l
            if needed in right_dict:
                c_r = right_dict[needed]
                return c_l + c_r
                
        return None

    coeffs1 = find_coeffs_for_target(best_s1, half1)
    if coeffs1 is None:
        # Error in logic?
        return [0] * 28 + [1, -1]
        
    coeffs2 = find_coeffs_for_target(best_s2, half2)
    if coeffs2 is None:
        return [0] * 28 + [1, -1]
        
    final_coeffs = list(coeffs1) + list(coeffs2)
    
    # Verify
    total = sum(final_coeffs[i] * (L // (i + 1)) for i in range(n))
    if total == 0:
        # This should not happen as we excluded 0
        # But just in case, flip last coeff
        if final_coeffs[-1] == 0:
            final_coeffs[-1] = 1
        else:
            final_coeffs[-1] = -final_coeffs[-1]
            
    return final_coeffs

# EVOLVE-BLOCK-END
