import time
import random
from math import gcd
from functools import reduce
from itertools import product
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 30

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, n + 1))
    W = [L // (i + 1) for i in range(n)]

    # Split indices into two halves
    mid = n // 2
    idx1 = list(range(mid))
    idx2 = list(range(mid, n))
    
    w1 = [W[i] for i in idx1]
    w2 = [W[i] for i in idx2]
    
    # Function to generate all sums for a set of weights
    # Returns list of sums
    def generate_sums(weights):
        sums = []
        # Using product is slow in Python for 15 dimensions?
        # 3^15 = 14M. 
        # A recursive or iterative approach might be faster.
        # Let's try a simple recursive generator or iterative expansion.
        
        # Iterative expansion:
        # Start with [0]
        # For each weight, expand list: new_list = [x-w, x, x+w for x in current_list]
        # This is efficient and avoids product overhead.
        
        current_sums = [0]
        for w in weights:
            next_sums = []
            # Optimization: local variable access
            cs = current_sums
            for s in cs:
                next_sums.append(s - w)
                next_sums.append(s)
                next_sums.append(s + w)
            current_sums = next_sums
        return current_sums

    # Generate sums for first half
    sums1 = generate_sums(w1)
    
    # Generate sums for second half
    sums2 = generate_sums(w2)
    
    # Sort sums2 for binary search
    sums2.sort()
    
    # Create a set of sums2 for fast lookup in pass 2
    # This is memory heavy but necessary for efficient pass 2
    set2 = set(sums2)
    
    best_abs_N = float('inf')
    best_total = 0
    
    # Pass 1: Find the minimum non-zero absolute sum
    # We iterate over sums1 and find closest s2 in sums2 to -s1
    for s1 in sums1:
        # We want s2 such that s1 + s2 is close to 0 but not 0
        # So s2 is close to -s1
        target = -s1
        pos = bisect_left(sums2, target)
        
        # Check neighbors: pos-1 and pos
        # Also pos+1 might be needed if pos is out of bounds or duplicates?
        # bisect_left gives first index >= target.
        # So pos is the first element >= target.
        # pos-1 is the last element < target.
        
        candidates = []
        if pos < len(sums2):
            candidates.append(sums2[pos])
        if pos > 0:
            candidates.append(sums2[pos-1])
            
        for s2 in candidates:
            total = s1 + s2
            if total == 0:
                continue
            abs_total = abs(total)
            if abs_total < best_abs_N:
                best_abs_N = abs_total
                best_total = total
                
    if best_total == 0:
        # Should not happen if there is any non-zero sum
        # But if all sums are 0? Impossible.
        # If best_abs_N is still inf, something went wrong.
        # Fallback to simple random search
        best_total = W[0]
        best_abs_N = W[0]

    # Pass 2: Find coefficients that achieve best_total
    # We need to find c1, c2 such that sum(c1*w1) + sum(c2*w2) = best_total
    
    # We will iterate through all combinations of c1, calculate s1.
    # If best_total - s1 is in set2, we have a match.
    # Then we need to find c2 that produces that specific s2.
    
    target_s2_val = None
    coeffs1_found = None
    coeffs2_found = None
    
    # To speed up Pass 2, we can stop early if we find a match.
    # But we need to iterate through all c1 to find one that works.
    
    # Let's generate c1 sums and coeffs together?
    # That requires storing coeffs, which is memory heavy.
    # Instead, we can just re-generate sums1 and keep track of coeffs?
    # No, we can just iterate through product again.
    
    # Optimization:
    # We can store a mapping from s1 to coeffs1 for the *first* occurrence?
    # Or just iterate.
    
    # Let's iterate through all combinations of idx1
    # 3^15 = 14M iterations.
    
    found = False
    # We need to be careful with time.
    # If we iterate 14M times in Python, it takes ~5s.
    # For each match, we might iterate 14M times for idx2?
    # That would be 14M * 14M which is impossible.
    
    # We need to find coeffs2 for a specific target_s2.
    # We can pre-compute a map from sum to coeffs for idx2?
    # Memory issue again.
    
    # Alternative:
    # In Pass 1, we found best_total.
    # We can iterate idx1 combos.
    # For each s1, check if target_s2 = best_total - s1 is in set2.
    # If yes, we have coeffs1 (from current iteration).
    # Now we need coeffs2 for target_s2.
    # We can iterate idx2 combos to find target_s2.
    # This inner loop is only executed ONCE (for the first s1 that matches).
    # So total work is 14M (outer) + 14M (inner) = 28M iterations.
    # This is feasible (~10-15s).
    
    for c1 in product([-1, 0, 1], repeat=len(idx1)):
        s1 = 0
        for i, w in zip(idx1, w1):
            s1 += c1[i] * w
        target_s2 = best_total - s1
        if target_s2 in set2:
            # Found s1
            coeffs1_found = c1
            target_s2_val = target_s2
            found = True
            break
            
    if not found:
        # Fallback: This should not happen
        coeffs1_found = (0,) * len(idx1)
        coeffs2_found = (1,) + (0,) * (len(idx2) - 1) # Dummy
    else:
        # Find coeffs2 for target_s2_val
        for c2 in product([-1, 0, 1], repeat=len(idx2)):
            s2 = 0
            for i, w in zip(idx2, w2):
                s2 += c2[i] * w
            if s2 == target_s2_val:
                coeffs2_found = c2
                break
                
    if coeffs2_found is None:
        # Fallback
        coeffs2_found = (0,) * len(idx2)

    # Construct final coefficients
    coeffs = [0] * n
    for i, c in zip(idx1, coeffs1_found):
        coeffs[i] = c
    for i, c in zip(idx2, coeffs2_found):
        coeffs[i] = c
        
    return coeffs
