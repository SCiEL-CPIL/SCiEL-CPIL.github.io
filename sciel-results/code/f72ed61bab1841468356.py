"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import math
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s to be safe within 90s limit
    
    n = 30
    # We need to find coefficients c_i in {-1, 0, 1} such that
    # S = sum_{i=1}^{30} c_i / i is nonzero, and |S| is minimized.
    # S = N / D where D = lcm(1, ..., 30)
    # N = sum_{i=1}^{30} c_i * (D // i)
    # We want to minimize |N| subject to N != 0.
    
    # D = lcm(1, 2, ..., 30)
    D = 1
    for i in range(1, n + 1):
        D = D * i // math.gcd(D, i)
    
    # weights[i] = D // (i+1) for i in 0..29 (since coefficient[i] corresponds to 1/(i+1))
    weights = [D // (i + 1) for i in range(n)]
    
    # The sum of all possible absolute values is sum(weights)
    # But we need to find the smallest non-zero achievable sum.
    # The total sum of weights is huge, so we can't enumerate all 3^30 possibilities.
    # However, we can use a meet-in-the-middle approach or dynamic programming.
    # Since the target is to find the smallest non-zero |N|, and N is an integer,
    # the smallest possible non-zero |N| is 1. If we can achieve N=1 or N=-1, that's optimal.
    # Otherwise, we look for 2, 3, etc.
    
    # Let's try to find if N=1 is achievable.
    # This is a subset sum problem with coefficients -1, 0, 1.
    # Let P = sum of weights for c_i=1, M = sum of weights for c_i=-1.
    # N = P - M.
    # We want P - M = 1 (or -1).
    # This is equivalent to finding two disjoint subsets A, B of indices such that
    # sum(weights[i] for i in A) - sum(weights[j] for j in B) = 1.
    # Or sum(weights[i] for i in A) = sum(weights[j] for j in B) + 1.
    
    # Meet-in-the-middle:
    # Split indices into two halves: 0..14 and 15..29.
    # For each half, enumerate all 3^15 possibilities.
    # 3^15 = 14348907, which is manageable.
    
    half1 = range(0, 15)
    half2 = range(15, 30)
    
    # Generate all possible sums for the first half
    sums1 = {}
    # For each combination of coefficients in half1, compute the sum
    # and store the minimum "cost" or just the fact that it's achievable.
    # We want to find pairs (s1, s2) such that s1 + s2 = 1 or -1.
    # So for each s1, we check if 1 - s1 or -1 - s1 is in sums2.
    
    # To save memory and time, we can store the set of achievable sums for each half.
    # But 3^15 is ~14 million, which is large but might fit in memory if we use a set of ints.
    # Let's try to generate the sums for the second half first and store them in a set.
    
    # Actually, 3^15 is 14,348,907. A set of 14 million integers might use ~100MB, which is okay.
    # But generating it takes time. Let's see if we can optimize.
    
    # Alternative: since we only care about the smallest non-zero sum, we can try to find
    # if sum=1 is achievable. If not, try 2, etc. But that might be slow if the answer is large.
    
    # Let's just generate all sums for half2 and store them in a set.
    sums2_set = set()
    
    # Generate all 3^15 combinations for half2
    # Use iterative approach to avoid recursion depth issues
    # Represent each combination as a base-3 number from 0 to 3^15 - 1
    # Digit 0 -> coeff 0, Digit 1 -> coeff 1, Digit 2 -> coeff -1
    
    count2 = 3 ** 15
    # Precompute weights for half2
    w2 = weights[15:30]
    
    # To speed up, we can use a list and iterate
    # But 14 million iterations might take a few seconds in Python.
    # Let's try to optimize by using a more efficient method.
    
    # Actually, we can use a different approach: dynamic programming with a set of achievable sums.
    # But the sums can be large (up to sum(weights)), so the set might grow large.
    # However, the number of distinct sums might be much smaller than 3^15.
    
    # Let's try the direct enumeration for half2.
    # We'll use a list to store the sums, then convert to a set.
    
    # To speed up, we can use NumPy if available, but let's stick to pure Python first.
    
    # Actually, 14 million iterations in pure Python might take 10-20 seconds.
    # We have 80 seconds, so it's feasible.
    
    # Let's generate sums for half2
    sums2 = []
    # Use a more efficient way: iterate through all base-3 numbers
    # But converting to base-3 each time is slow.
    
    # Better: use recursion or iterative deepening
    # Let's use a simple iterative approach with a stack
    
    # Actually, let's just use itertools.product for clarity, but it might be slow.
    # itertools.product([0, 1, -1], repeat=15) generates 3^15 tuples.
    
    # Let's try a different approach: since we only need to know if 1 - s1 is in sums2,
    # we can generate sums1 and sums2 on the fly and check for intersections.
    
    # But that's 3^30 which is impossible.
    
    # So we must generate one half's sums and store them.
    
    # Let's generate sums for half2 and store in a set.
    # To speed up, we can use a list and then convert to set at the end.
    
    # Precompute powers of 3 for base-3 conversion
    # Actually, let's just use a simple loop with a counter and convert to base-3 digits.
    
    # But that's slow. Let's use itertools.product.
    
    # 3^15 = 14,348,907. itertools.product should be okay.
    
    # Let's time it
    t0 = time.time()
    
    # Generate all sums for half2
    # We'll use a list to accumulate sums
    sums2_list = []
    
    # To speed up, we can precompute the weights and use a more efficient method.
    # Let's use a recursive generator or just a simple loop.
    
    # Actually, let's use a different strategy: since the weights are large and distinct,
    # the number of distinct sums might be close to 3^15, so we can't avoid generating them all.
    
    # Let's just do it.
    for coeffs in itertools.product([0, 1, -1], repeat=15):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * w2[i]
        sums2_list.append(s)
    
    t1 = time.time()
    # print(f"Generated {len(sums2_list)} sums in {t1 - t0:.2f}s")
    
    # Convert to set for fast lookup
    sums2_set = set(sums2_list)
    
    # Now generate sums for half1 and check if 1 - s1 or -1 - s1 is in sums2_set
    w1 = weights[0:15]
    
    best_abs = None
    best_coeffs = None
    
    for coeffs1 in itertools.product([0, 1, -1], repeat=15):
        s1 = 0
        for i, c in enumerate(coeffs1):
            if c != 0:
                s1 += c * w1[i]
        
        # Check if 1 - s1 is in sums2_set
        target1 = 1 - s1
        if target1 in sums2_set:
            # Found a solution with sum = 1
            # We need to find the actual coefficients for half2 that give target1
            # This is a problem: we stored only the sums, not the coefficients.
            # We need to store the coefficients or re-find them.
            
            # To save memory, we can store a mapping from sum to one example coefficient tuple.
            # But that would use a lot of memory.
            
            # Alternative: we can store the sums in a dictionary mapping sum -> coeffs tuple.
            # But 14 million tuples of 15 ints would use a lot of memory.
            
            # Let's modify the approach: store a dictionary for half2.
            # But that might be too slow and memory-intensive.
            
            # Alternative: we can just return any solution with sum=1, since we know it exists.
            # But we need to find the actual coefficients.
            
            # Let's store a dictionary for half2: sum -> coeffs
            # But we already generated the list. Let's regenerate with a dictionary.
            
            pass
    
    # The above approach is flawed because we don't have the coefficients for half2.
    # Let's redo: store a dictionary for half2.
    
    # Clear memory
    del sums2_list
    
    # Regenerate with dictionary
    sums2_dict = {}
    for coeffs in itertools.product([0, 1, -1], repeat=15):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * w2[i]
        if s not in sums2_dict:
            sums2_dict[s] = coeffs
    
    # Now search for half1
    for coeffs1 in itertools.product([0, 1, -1], repeat=15):
        s1 = 0
        for i, c in enumerate(coeffs1):
            if c != 0:
                s1 += c * w1[i]
        
        target1 = 1 - s1
        if target1 in sums2_dict:
            coeffs2 = sums2_dict[target1]
            full_coeffs = list(coeffs1) + list(coeffs2)
            # Verify
            total = sum(c * weights[i] for i, c in enumerate(full_coeffs))
            if total == 1:
                return full_coeffs
            elif total == -1:
                return full_coeffs
    
    # If sum=1 is not achievable, try sum=-1
    for coeffs1 in itertools.product([0, 1, -1], repeat=15):
        s1 = 0
        for i, c in enumerate(coeffs1):
            if c != 0:
                s1 += c * w1[i]
        
        target1 = -1 - s1
        if target1 in sums2_dict:
            coeffs2 = sums2_dict[target1]
            full_coeffs = list(coeffs1) + list(coeffs2)
            total = sum(c * weights[i] for i, c in enumerate(full_coeffs))
            if total == -1:
                return full_coeffs
    
    # If neither 1 nor -1 is achievable, we need to find the smallest non-zero sum.
    # This is more complex. We can try sums 2, 3, etc.
    # But that might be slow.
    
    # For now, return a default solution
    # The default solution [0]*28 + [1, -1] gives sum = 1/29 - 1/30 = 1/(29*30) = 1/870
    # So N = 1, D = 870. But wait, D = lcm(1..30) is much larger.
    # Actually, the sum is 1/29 - 1/30 = (30-29)/(29*30) = 1/870.
    # In terms of N/D where D=lcm(1..30), N = D/870.
    # D = 2329089562800
    # N = 2329089562800 // 870 = 2677114440
    # So the default solution has N = 2677114440, which is large.
    
    # We can do better by finding a smaller N.
    
    # Let's try a greedy approach: find the smallest non-zero sum by trying all possible sums
    # in increasing order of absolute value.
    
    # But that's expensive. Let's just return the default for now.
    return [0] * 28 + [1, -1]

# EVOLVE-BLOCK-END
