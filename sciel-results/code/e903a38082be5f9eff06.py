import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Finds 30 integer coefficients from {-1, 0, 1} such that the signed sum
    sum(c[i-1]/i) is non-zero and its absolute value is minimized.
    
    Strategy: Meet-in-the-Middle (MITM) exact search.
    1. Split the 30 coefficients into two groups of 15: indices 0-14 and 15-29.
    2. Enumerate all 3^15 = 14,348,907 possible sums for the first group.
       Store these sums in a set for O(1) existence checks and to recover coefficients.
    3. Enumerate all 3^15 possible sums for the second group.
    4. For each sum s2 in the second group, we want to find s1 in the first group
       such that |s1 + s2| is minimized and s1 + s2 != 0.
       Since the set of s1 values is finite, we can iterate through the second group's
       sums and for each, check candidates in the first group.
       
       Optimization: 
       Instead of checking all s1 for each s2 (which would be 3^30), we note that
       the optimal s1 for a given s2 is the s1 in the set closest to -s2.
       However, a simple set lookup for "is -s2 in set" only finds sum=0.
       We need the closest non-zero sum.
       
       Efficient Approach:
       1. Generate all sums for group 1 (S1). Store in a list.
       2. Sort S1.
       3. For each sum s2 in group 2 (S2):
          - Target is -s2.
          - Binary search in sorted S1 for -s2.
          - Check the element at the insertion point and its neighbors.
          - If the found s1 makes s1+s2 == 0, check the next closest elements.
          - Track the global minimum |s1+s2| != 0.
          
       Complexity:
       - Generating S1: ~14M operations.
       - Sorting S1: ~14M * log(14M) ~ 14M * 24 ~ 340M operations. This might be tight for 90s in Python if not careful, but Fraction comparison is slow.
       
       Alternative Optimization for Fractions:
       Fractions are expensive. Let's use integer arithmetic.
       Let L = lcm(1, ..., 30).
       Sum = (sum c_i * (L/i)) / L.
       Minimizing |Sum| is equivalent to minimizing |sum c_i * (L/i)|.
       Let w_i = L // i.
       We are looking for min |sum1 + sum2| where sum1 is from {sum c_i w_i | i in 1..15} and sum2 from {sum c_i w_i | i in 16..30}.
       The weights w_i are integers. The maximum sum is bounded by sum(w_i).
       L for 1..30 is large, but Python handles big integers natively.
       
       Let's refine the MITM with integers:
       1. Compute L = lcm(1..30).
       2. Compute weights w[1..30].
       3. Generate all possible integer sums for the first 15 indices. Store in a list `sums1`.
       4. Sort `sums1`.
       5. Generate all possible integer sums for the last 15 indices.
       6. For each `s2` in the second group:
          - Target `t = -s2`.
          - Binary search `t` in `sums1`.
          - Check the closest few elements in `sums1` around `t` to find `s1` such that `s1+s2 != 0` and `|s1+s2|` is minimal.
          
       This reduces the problem to two lists of size 14M. 
       Generating 14M integers takes ~1-2 seconds in optimized Python (using iterative generation).
       Sorting 14M integers takes ~2-5 seconds.
       Iterating 14M times and doing binary searches (log(14M) ~ 24 steps) takes ~14M * 24 ~ 340M steps. 
       In Python, 340M steps might take 30-60 seconds. This is risky but feasible within 90s.
       
       To speed up the search phase:
       Instead of binary searching for every s2, we can merge-sort style traversal?
       Or, since we just need the global minimum, we can iterate through the sorted sums1 and sorted sums2?
       No, the sum is s1 + s2. We want s1 + s2 close to 0.
       This is equivalent to finding two numbers, one from A, one from B, whose sum is closest to 0.
       We can sort A and B.
       Then use two pointers?
       If A is sorted ascending and B is sorted ascending:
       We want A[i] + B[j] ~ 0.
       Start i=0, j=len(B)-1.
       If A[i] + B[j] > 0, decrease j.
       If A[i] + B[j] < 0, increase i.
       If A[i] + B[j] == 0, we need to find the next best non-zero.
       
       This two-pointer approach is O(N + M) = O(14M + 14M) = 28M steps. This is MUCH faster than 340M.
       
       Let's implement the two-pointer approach on sorted integer sums.
       """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 30
    # 1. Compute LCM and weights
    import math
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    # 2. Generate sums for first half (indices 0-14)
    # Weights for first half
    w1 = weights[:15]
    # Weights for second half
    w2 = weights[15:]
    
    def generate_sums(ws):
        """
        Generate all possible sums for a list of weights using coefficients -1, 0, 1.
        Returns a list of (sum_value, coeffs_tuple).
        """
        sums = []
        k = len(ws)
        # Iterative generation to avoid recursion depth issues and overhead
        # Total combinations: 3^k
        # We can use base-3 counting or recursive generation.
        # Recursive is often faster in Python if implemented carefully, but 3^15 is deep.
        # Let's use a stack or iterative base-3.
        
        # Actually, for 3^15, a simple recursive function with memoization or just a loop
        # might be slow. Let's try an iterative approach using a list of current sums.
        # Start with [0]
        # For each weight w:
        #   new_sums = []
        #   for s in current_sums:
        #       new_sums.append(s - w)
        #       new_sums.append(s)
        #       new_sums.append(s + w)
        #   current_sums = new_sums
        # This creates 3^k lists. Memory usage is high but manageable for 14M ints?
        # 14M ints * 28 bytes ~ 400MB. This might be too high for some environments, but usually OK.
        # However, we also need to store the coefficients to reconstruct the answer.
        # Storing coefficients for 14M items is expensive.
        
        # Alternative: Store only sums in a list, sort them.
        # Then, how do we reconstruct the coefficients?
        # We can store a mapping from sum_value to a representative coefficient vector?
        # No, multiple vectors can have the same sum. We just need ANY vector that achieves the optimal sum.
        # So, we can store a dictionary: sum_value -> coeffs_tuple.
        # If a sum is encountered again, we can keep the first one or any one.
        
        # Let's try the dictionary approach.
        # dict: sum -> coeffs
        # Size: 14M entries. This is large but feasible in CPython if memory allows.
        # Let's estimate memory: 14M * (key + value + overhead). Key is int (~28 bytes), Value is tuple of 15 ints (~15*28 + overhead ~ 500 bytes).
        # 14M * 500 bytes = 7GB. This is too much.
        
        # We need a more memory-efficient way.
        # We only need the sum values for the two-pointer search.
        # After finding the optimal s1 and s2, we can reconstruct the coefficients?
        # No, we need the specific coefficients that produced s1 and s2.
        
        # Revised Plan:
        # 1. Generate list of (sum, coeffs) for first half.
        # 2. Sort by sum.
        # 3. Generate list of (sum, coeffs) for second half.
        # 4. Sort by sum.
        # 5. Two-pointer search.
        
        # Memory check:
        # List of 14M tuples. Each tuple: (int, tuple_of_15_ints).
        # This is still heavy.
        
        # Can we reduce memory?
        # We can store sums in a list, and coeffs in a separate parallel structure?
        # Or, we can note that the number of UNIQUE sums is much smaller than 3^15?
        # The sums are multiples of some gcd? No, weights are L/i.
        # The range of sums is [-sum(w), sum(w)].
        # sum(w) for 1..15 is roughly L * sum(1/i) ~ L * 3.3.
        # L is huge. So the range is huge. Unique sums are likely close to 3^15.
        
        # Let's try to be smarter.
        # We don't need to store ALL coeffs if we can reconstruct them.
        # But reconstruction is hard without storing the path.
        
        # Let's assume we have enough memory for 2 lists of 14M tuples.
        # If memory fails, we can fall back to random search.
        
        # To save time in generation, we can use a more efficient method.
        # But 14M items is the bottleneck.
        
        # Let's try to generate just the sums first, sort them, and see how long it takes.
        # If it's too slow, we abort.
        
        pass

    # Actually, let's look at the time.
    # Generating 14M items in Python:
    # Loop 15 times. Each step multiplies list size by 3.
    # Step 1: 3
    # Step 15: 14M.
    # Total operations: 3 + 9 + ... + 14M ~ 21M appends.
    # This is fast (~1-2 seconds).
    
    # Sorting 14M integers:
    # Python's Timsort is fast. ~2-5 seconds.
    
    # Two-pointer search:
    # 28M steps. ~5-10 seconds.
    
    # Total time: ~10-20 seconds. This is well within 90s.
    
    # The main risk is memory.
    # Let's try to store only the sums in a list, and use a second pass to find the coeffs?
    # No, we need the coeffs for the specific s1 and s2 found.
    
    # Let's store the coeffs in a compact way.
    # Instead of a tuple of 15 ints, we can encode the coeffs as a base-3 integer.
    # 3^15 is about 14M, which fits in a 24-bit integer.
    # So we can store (sum_value, code).
    # Then decode the code to get coeffs.
    
    def generate_sums_with_codes(ws):
        """
        Returns a list of (sum_value, code) where code is a base-3 representation of coeffs.
        """
        current = [(0, 0)] # (sum, code)
        for i, w in enumerate(ws):
            new_current = []
            # We iterate over existing sums
            # To make it faster, we can list comprehension?
            # But we need to update code.
            # code_new = code_old + coeff * (3^i)
            # Let's precompute powers of 3.
            pow3 = [3**i for i in range(len(ws))]
            
            for s, c in current:
                # coeff -1
                new_current.append((s - w, c - pow3[i]))
                # coeff 0
                new_current.append((s, c))
                # coeff 1
                new_current.append((s + w, c + pow3[i]))
            current = new_current
        return current

    # Note: The code can be negative?
    # If we use standard base-3, digits are 0,1,2.
    # Coeffs are -1,0,1.
    # Map -1->0, 0->1, 1->2.
    # Then code = sum (mapped_coeff * 3^i).
    # Range of code: 0 to 3^15 - 1.
    # This fits in an int.
    
    def generate_sums_optimized(ws):
        current = [(0, 0)]
        pow3 = [1] * len(ws)
        for i in range(1, len(ws)):
            pow3[i] = pow3[i-1] * 3
            
        for i, w in enumerate(ws):
            p = pow3[i]
            new_current = []
            # Local variable for speed
            append = new_current.append
            for s, c in current:
                # coeff -1 -> digit 0
                append((s - w, c))
                # coeff 0 -> digit 1
                append((s, c + p))
                # coeff 1 -> digit 2
                append((s + w, c + 2*p))
            current = new_current
        return current

    # Let's run this for both halves.
    sums1 = generate_sums_optimized(w1)
    sums2 = generate_sums_optimized(w2)
    
    # Sort by sum
    sums1.sort(key=lambda x: x[0])
    sums2.sort(key=lambda x: x[0])
    
    # Two-pointer search for min |s1 + s2| != 0
    n1 = len(sums1)
    n2 = len(sums2)
    
    i = 0
    j = n2 - 1
    
    best_diff = None
    best_i = None
    best_j = None
    
    # We need to be careful with the two-pointer logic when sum is 0.
    # If s1[i] + s2[j] == 0, we must move one pointer to find a non-zero sum.
    # We should check the neighbors to ensure we don't miss the smallest non-zero sum.
    
    # Standard closest pair sum to 0:
    # If sum > 0, j--
    # If sum < 0, i++
    # If sum == 0, we have a valid zero sum (invalid for us), so we need to find the next best.
    # We can try i+1, j-1, etc.
    
    # Actually, it's safer to just check the current sum. If it's 0, we record it as "invalid" but keep looking.
    # But the two-pointer method moves away from 0.
    # If we hit 0, the next moves will yield non-zero sums.
    # However, we might skip the *closest* non-zero sum if we jump over it?
    # No, the sums are sorted. The closest non-zero sum to 0 will be adjacent to 0 in the sorted order of the combined sums?
    # Not exactly. s1 + s2 = 0 is a single point.
    # The next best sum is min(|s1[i+1]+s2[j]|, |s1[i]+s2[j-1]|).
    
    # Let's modify the loop:
    while i < n1 and j >= 0:
        if time.time() > deadline:
            break
            
        s_val = sums1[i][0] + sums2[j][0]
        
        if s_val == 0:
            # We found a zero sum. We need to find the closest non-zero.
            # Candidates:
            # 1. Increase i: s1[i+1] + s2[j]
            # 2. Decrease j: s1[i] + s2[j-1]
            # We should check both and pick the one with smaller abs sum.
            # Then update best if it's better than current best.
            # Then move the pointer that gave the better result?
            # Or just move both?
            
            # Let's check the immediate neighbors.
            candidate_sums = []
            
            if i + 1 < n1:
                candidate_sums.append((abs(sums1[i+1][0] + sums2[j][0]), i+1, j))
            if j - 1 >= 0:
                candidate_sums.append((abs(sums1[i][0] + sums2[j-1][0]), i, j-1))
                
            # Also, it's possible that the global minimum non-zero sum is found elsewhere.
            # But since we are doing a sweep, we should track the global best.
            
            # Update global best with these candidates
            for diff, ci, cj in candidate_sums:
                if diff > 0: # Must be non-zero
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_i = ci
                        best_j = cj
            
            # Move pointers to continue search.
            # Since s1[i]+s2[j]=0, increasing i makes sum positive, decreasing j makes sum negative.
            # We should move both to explore new territory?
            # Actually, just moving one might be enough to find the next candidate.
            # Let's move i++ and j--.
            i += 1
            j -= 1
        elif s_val > 0:
            # Sum is too large, decrease s2
            # Check current sum as a candidate?
            # Yes, because it might be the best non-zero sum so far.
            if best_diff is None or abs(s_val) < best_diff:
                best_diff = abs(s_val)
                best_i = i
                best_j = j
            j -= 1
        else: # s_val < 0
            # Sum is too small, increase s1
            if best_diff is None or abs(s_val) < best_diff:
                best_diff = abs(s_val)
                best_i = i
                best_j = j
            i += 1

    if best_i is None:
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients
    code1 = sums1[best_i][1]
    code2 = sums2[best_j][1]
    
    coeffs = [0] * 30
    
    # Decode code1 for first 15
    for i in range(15):
        digit = code1 % 3
        code1 //= 3
        # Digit 0 -> -1, 1 -> 0, 2 -> 1
        coeffs[i] = digit - 1
        
    # Decode code2 for next 15
    for i in range(15):
        digit = code2 % 3
        code2 //= 3
        coeffs[15+i] = digit - 1
        
    return coeffs
# EVOLVE-BLOCK-END
