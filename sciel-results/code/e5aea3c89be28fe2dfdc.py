"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Weights w_i = L // i for i in 1..30
    # Sum S = sum(c_i * w_i)
    # We want S to be a small nonzero integer.
    # c_i in {-1, 0, 1}
    
    # Split indices 0..29 into two halves
    n = 30
    half = 15
    
    # Weights for first half (indices 0..14)
    w1 = [L // (i + 1) for i in range(half)]
    # Weights for second half (indices 15..29)
    w2 = [L // (i + 1) for i in range(half, n)]
    
    # Generate all possible sums for the first half
    # There are 3^15 possibilities.
    # We store the sums in a set. To reconstruct the solution, 
    # we also need to store one set of coefficients that yields that sum.
    # However, storing coefficients for 1.4M entries might be memory heavy 
    # or slow if we store lists. 
    # Alternative: Store just the sum in a set. 
    # If we find a match, we can re-search for the specific coefficients 
    # for that sum? No, that's inefficient.
    # Better: Store a mapping from sum to the index of the choice tuple?
    # Or simply: Since 3^15 is manageable, we can store the sum -> coeffs.
    # Let's optimize memory by storing only the sum in a set first to check existence,
    # and if found, we can backtrack or store a parent pointer.
    # Actually, 1.4M integers in a set is ~10-20MB. That's fine.
    # But we need the coefficients to return them.
    # Let's store: sum -> tuple of coefficients for that half.
    # Tuple of 15 ints is ~100 bytes. 1.4M * 100 bytes = 140MB. Might be tight but likely OK.
    # To be safe and fast, we can store sum -> index in a list of coeffs?
    # Or, since we just need ONE solution, we can do this:
    # 1. Generate all sums for half 1. Store in dict: sum -> coeffs_tuple.
    # 2. Iterate all sums for half 2. For each sum s2, check if (target - s2) is in dict.
    # 3. If yes, combine coeffs.
    
    # Let's try to be efficient.
    # We will search for target sums k = 1, 2, 3, ...
    # Because if sum is 1, value is 1/L. If sum is 2, value is 2/L.
    # 1/L is the minimum possible non-zero value (since numerator is integer).
    # So we primarily look for sum = 1 or -1.
    
    # Generate half 1 sums
    sums1 = {}
    
    # Recursive generation or iterative?
    # Iterative with a list of current states is faster in Python than recursion depth 15.
    # Start with [(0, (0,)*15)]
    # But 3^15 is 14M. Storing 14M tuples in a list is memory heavy.
    # Let's use a set of sums first to see if 1 is possible.
    # If 1 is possible, we then need to find the coeffs.
    # We can do a BFS/DFS to find the coeffs for the specific sum once existence is confirmed.
    
    # Step 1: Check if sum = 1 or -1 is possible.
    # We can use a bitset or set of integers.
    possible_sums1 = set()
    
    # Generate all sums for half 1
    # Using iterative deepening or simple recursion?
    # Recursion limit is usually 1000, so depth 15 is fine.
    # But 14M calls might be slow.
    # Let's use an iterative approach with a list of (sum, coeffs) but only keep the sum in the set for existence.
    # To reconstruct, we can store the coeffs in a dict only for the sums we care about? No, we don't know which ones we care about until we check half 2.
    
    # Strategy:
    # 1. Generate all (sum, coeffs) for half 1. Store in dict `sum_to_coeffs1`.
    #    Key: sum, Value: coeffs tuple.
    #    If multiple coeffs give same sum, keep one.
    # 2. Generate all (sum, coeffs) for half 2.
    # 3. For each (s2, c2) in half 2:
    #    target = 1 - s2
    #    if target in sum_to_coeffs1:
    #        c1 = sum_to_coeffs1[target]
    #        return c1 + c2
    #    target = -1 - s2
    #    if target in sum_to_coeffs1:
    #        c1 = sum_to_coeffs1[target]
    #        return c1 + c2
        
    # Memory check:
    # 14M entries in a dict.
    # Keys are large ints. Values are tuples of 15 small ints.
    # This might exceed memory or time limits in Python if not careful.
    # 14M * (key overhead + value overhead) could be > 1GB.
    # Risky.
    
    # Alternative:
    # Since we only care about small targets (1, -1, 2, -2...),
    # and the sums are distributed, we can:
    # 1. Generate all sums for half 1 and store in a `set` of ints. (Much smaller memory)
    # 2. Generate all sums for half 2.
    # 3. For each s2, check if (1 - s2) in set1.
    # 4. If found, we know a solution exists.
    # 5. Then, we need to find the specific coefficients.
    #    We can run a second pass:
    #    Generate half 1 again, but this time, when we hit a sum that is in the "required" set 
    #    (derived from the s2 that matched), we record the coeffs.
    #    But we need to know WHICH s1 is required.
    #    We can store the required s1 values in a set `required_s1`.
    #    Then do a DFS on half 1 to find coeffs for any sum in `required_s1`.
    
    # Let's implement this two-pass approach.
    
    # Pass 1: Existence check
    sums1_set = set()
    
    # Generate sums for half 1
    # Using recursion for clarity, but we must ensure it's fast.
    # Depth 15.
    def gen_sums(weights, idx, current_sum, sums_set):
        if idx == len(weights):
            sums_set.add(current_sum)
            return
        w = weights[idx]
        # Option 0
        gen_sums(weights, idx + 1, current_sum, sums_set)
        # Option +1
        gen_sums(weights, idx + 1, current_sum + w, sums_set)
        # Option -1
        gen_sums(weights, idx + 1, current_sum - w, sums_set)
        
    # Recursion might be slow for 14M calls. 
    # Let's try an iterative stack-based approach or just hope Python handles it.
    # Actually, 14M function calls in Python is ~1-2 seconds. Acceptable.
    
    # Check time
    if time.time() > deadline - 10:
        return [0] * 28 + [1, -1]
        
    gen_sums(w1, 0, 0, sums1_set)
    
    # Now check half 2 against targets 1, -1
    # We need to find if there exists s2 in S2 such that (1 - s2) in S1 OR (-1 - s2) in S1.
    # We also need to remember the s2 that worked, and its coeffs.
    # So we generate half 2 sums with coeffs.
    
    found_s2 = None
    found_c2 = None
    found_target_sign = 0 # 1 if we matched 1, -1 if we matched -1
    
    # We need to generate half 2 sums and check.
    # We can do this iteratively to avoid storing all half 2 sums.
    
    def check_half2(weights, idx, current_sum, current_coeffs):
        nonlocal found_s2, found_c2, found_target_sign
        if found_s2 is not None:
            return
        
        if idx == len(weights):
            # Check target 1
            if (1 - current_sum) in sums1_set:
                found_s2 = current_sum
                found_c2 = tuple(current_coeffs)
                found_target_sign = 1
                return
            # Check target -1
            if (-1 - current_sum) in sums1_set:
                found_s2 = current_sum
                found_c2 = tuple(current_coeffs)
                found_target_sign = -1
                return
            return
        
        w = weights[idx]
        
        # Option 0
        current_coeffs[idx] = 0
        check_half2(weights, idx + 1, current_sum, current_coeffs)
        if found_s2 is not None: return
        
        # Option +1
        current_coeffs[idx] = 1
        check_half2(weights, idx + 1, current_sum + w, current_coeffs)
        if found_s2 is not None: return
        
        # Option -1
        current_coeffs[idx] = -1
        check_half2(weights, idx + 1, current_sum - w, current_coeffs)
        if found_s2 is not None: return
        
    c2_init = [0] * half
    check_half2(w2, 0, 0, c2_init)
    
    if found_s2 is not None:
        # We found a target sign. Let's say we want sum = found_target_sign.
        # We need s1 such that s1 + s2 = found_target_sign.
        target_s1 = found_target_sign - found_s2
        
        # Now find coeffs for s1 in half 1.
        found_c1 = None
        def find_c1(weights, idx, current_sum, current_coeffs):
            nonlocal found_c1
            if found_c1 is not None:
                return
            if idx == len(weights):
                if current_sum == target_s1:
                    found_c1 = tuple(current_coeffs)
                return
            
            w = weights[idx]
            
            # Option 0
            current_coeffs[idx] = 0
            find_c1(weights, idx + 1, current_sum, current_coeffs)
            if found_c1 is not None: return
            
            # Option +1
            current_coeffs[idx] = 1
            find_c1(weights, idx + 1, current_sum + w, current_coeffs)
            if found_c1 is not None: return
            
            # Option -1
            current_coeffs[idx] = -1
            find_c1(weights, idx + 1, current_sum - w, current_coeffs)
            if found_c1 is not None: return
            
        c1_init = [0] * half
        find_c1(w1, 0, 0, c1_init)
        
        if found_c1 is not None:
            # Combine
            result = list(found_c1) + list(found_c2)
            # Verify
            s = sum(c * (L // (i+1)) for i, c in enumerate(result))
            if s != 0:
                return result
            else:
                # Fallback if verification fails (should not happen)
                return [0] * 28 + [1, -1]
        else:
            return [0] * 28 + [1, -1]
    else:
        # Sum 1 or -1 not found. Try 2, -2?
        # Given the density, 1 is very likely. 
        # If not, we can extend the target loop.
        # But for now, return seed if 1/-1 fails, or try a quick local search.
        # Let's try targets 2 and -2 as well in the check_half2 function if we refactor.
        # For this version, we stick to 1/-1.
        # If 1/-1 is not possible, the next best is 2.
        # We can just re-run with target 2.
        
        # Let's try target 2 and -2
        found_s2 = None
        found_c2 = None
        found_target_sign = 0
        
        def check_half2_t2(weights, idx, current_sum, current_coeffs):
            nonlocal found_s2, found_c2, found_target_sign
            if found_s2 is not None:
                return
            if idx == len(weights):
                if (2 - current_sum) in sums1_set:
                    found_s2 = current_sum
                    found_c2 = tuple(current_coeffs)
                    found_target_sign = 2
                    return
                if (-2 - current_sum) in sums1_set:
                    found_s2 = current_sum
                    found_c2 = tuple(current_coeffs)
                    found_target_sign = -2
                    return
                return
            
            w = weights[idx]
            
            current_coeffs[idx] = 0
            check_half2_t2(weights, idx + 1, current_sum, current_coeffs)
            if found_s2 is not None: return
            
            current_coeffs[idx] = 1
            check_half2_t2(weights, idx + 1, current_sum + w, current_coeffs)
            if found_s2 is not None: return
            
            current_coeffs[idx] = -1
            check_half2_t2(weights, idx + 1, current_sum - w, current_coeffs)
            if found_s2 is not None: return
            
        c2_init = [0] * half
        check_half2_t2(w2, 0, 0, c2_init)
        
        if found_s2 is not None:
            target_s1 = found_target_sign - found_s2
            
            found_c1 = None
            def find_c1_t2(weights, idx, current_sum, current_coeffs):
                nonlocal found_c1
                if found_c1 is not None:
                    return
                if idx == len(weights):
                    if current_sum == target_s1:
                        found_c1 = tuple(current_coeffs)
                    return
                
                w = weights[idx]
                
                current_coeffs[idx] = 0
                find_c1_t2(weights, idx + 1, current_sum, current_coeffs)
                if found_c1 is not None: return
                
                current_coeffs[idx] = 1
                find_c1_t2(weights, idx + 1, current_sum + w, current_coeffs)
                if found_c1 is not None: return
                
                current_coeffs[idx] = -1
                find_c1_t2(weights, idx + 1, current_sum - w, current_coeffs)
                if found_c1 is not None: return
                
            c1_init = [0] * half
            find_c1_t2(w1, 0, 0, c1_init)
            
            if found_c1 is not None:
                result = list(found_c1) + list(found_c2)
                s = sum(c * (L // (i+1)) for i, c in enumerate(result))
                if s != 0:
                    return result
        
        # Fallback
        return [0] * 28 + [1, -1]

# EVOLVE-BLOCK-END
