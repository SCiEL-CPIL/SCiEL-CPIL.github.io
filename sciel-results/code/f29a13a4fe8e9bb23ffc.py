import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N using a sieve
    # sigma(n) = sum of divisors of n
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Step 2: Group numbers by their sigma value
    # sigma_to_numbers[s] = list of n such that sigma(n) = s
    sigma_to_numbers = defaultdict(list)
    for n in range(1, N + 1):
        sigma_to_numbers[sigma[n]].append(n)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Step 3: For each sigma value, find pairs (a, b) that minimize |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this over all pairs with sigma(a) = sigma(b), a != b (to avoid 0)
    
    best_error_num = float('inf')
    best_error_den = 1
    best_a, best_b = 11, 6  # fallback: sigma(11)=12, sigma(6)=12
    
    # For each group of numbers with the same sigma, we need to find pairs (a,b)
    # that minimize |2a - 3b| / (2b)
    
    # Strategy: For each b in the group, the best a is close to 3b/2
    # So for each b, we look for a in the group closest to 3b/2
    
    # Process groups - but only groups with at least 2 elements
    for s, nums in sigma_to_numbers.items():
        if len(nums) < 2:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 3:
            break
        
        # Sort the numbers for binary search
        nums_sorted = sorted(nums)
        
        # For each b in nums_sorted, find a in nums_sorted closest to 3b/2
        # But we need a != b (or at least 2a != 3b to avoid error 0)
        
        # To speed up, we can iterate b in a strategic order
        # The error is |2a - 3b| / (2b), so larger b gives smaller denominator
        # But we also need 2a - 3b to be small
        
        # Let's iterate through all b values and use binary search for a
        # But this could be slow for large groups
        
        # For very large groups, we might want to limit the search
        # Let's try all pairs for small groups, and a smarter approach for large ones
        
        if len(nums_sorted) <= 1000:
            # Brute force for small groups
            for i in range(len(nums_sorted)):
                for j in range(len(nums_sorted)):
                    if i == j:
                        continue
                    a = nums_sorted[i]
                    b = nums_sorted[j]
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    den = 2 * b
                    # Compare num/den with best_error_num/best_error_den
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_a, best_b = a, b
        else:
            # For large groups, use binary search
            for b in nums_sorted:
                # We want a close to 3b/2
                target = 3 * b / 2.0
                
                # Binary search for position where target would be inserted
                lo, hi = 0, len(nums_sorted)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nums_sorted[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                
                # Check lo and lo-1 (if valid)
                candidates = []
                if lo < len(nums_sorted):
                    candidates.append(nums_sorted[lo])
                if lo > 0:
                    candidates.append(nums_sorted[lo - 1])
                # Also check a couple more neighbors for safety
                if lo + 1 < len(nums_sorted):
                    candidates.append(nums_sorted[lo + 1])
                if lo - 2 >= 0:
                    candidates.append(nums_sorted[lo - 2])
                
                for a in candidates:
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    den = 2 * b
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_a, best_b = a, b
    
    # Ensure Python int types
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]
