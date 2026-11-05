import time
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget, 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # To handle exact rationals efficiently, we can work with integers.
    # Let L = LCM(2, ..., 48).
    # Sum = sum(1/d_i) = sum(L/d_i) / L.
    # We want to minimize L - sum(L/d_i) subject to sum(L/d_i) < L.
    # This is equivalent to finding a subset of weights w_i = L/d_i 
    # that maximizes the sum subject to sum < L.
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers
    weights = [L // d for d in denominators]
    
    # We want to find a subset with sum S such that S < L and S is as large as possible.
    # This is a variant of the Subset Sum problem (knapsack) where capacity is L-1.
    # L is very large (LCM of 1..48 is ~ 5e17), so we cannot do DP on the sum directly.
    
    # However, n=47.
    # Meet-in-the-middle is the standard approach for n~40-50.
    # But we must be exact.
    
    # Let's try a different angle: 
    # The deficit is 1 - sum(1/d_i).
    # We want to minimize this positive value.
    # This is equivalent to maximizing sum(1/d_i) < 1.
    
    # Since floating point meet-in-the-middle might have precision issues,
    # let's use exact integer arithmetic for the meet-in-the-middle.
    
    # Split into two sets A and B.
    mid = n // 2
    A = weights[:mid]
    B = weights[mid:]
    
    # Generate all subset sums for A
    # Store as (sum, mask)
    sums_A = [(0, 0)]
    for i, w in enumerate(A):
        # For each existing sum, create a new sum with w added
        # This doubles the list size
        new_sums = []
        for s, m in sums_A:
            new_s = s + w
            new_m = m | (1 << i)
            new_sums.append((new_s, new_m))
        sums_A.extend(new_sums)
        
    # Filter sums_A to those < L (strictly, since total must be < L)
    # Actually, we just need to find the best combination.
    # Sort sums_A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # To speed up search, we can prune sums_A.
    # If we have two entries (s1, m1) and (s2, m2) with s1 <= s2 and m1 == m2? No, masks are unique.
    # But if s1 >= s2 and m1 has more bits? No, we just need the max sum < L - s_B.
    # We can keep only the "Pareto optimal" sums? 
    # Actually, for a given sum, we don't care about the mask unless we want to reconstruct.
    # But we need to check if sum_A + sum_B < L.
    
    # Let's just sort all sums_A.
    # Size of sums_A is 2^23 = 8.4M. This might be too large for Python lists of tuples.
    # Let's try to be more memory efficient.
    
    # Alternative: Use numpy for the sums, but we need exact integers.
    # Python integers are arbitrary precision.
    # 8.4M integers might take ~ 100-200MB. Acceptable?
    # Let's try.
    
    # Generate sums for B as well
    sums_B = [(0, 0)]
    for i, w in enumerate(B):
        new_sums = []
        for s, m in sums_B:
            new_s = s + w
            new_m = m | (1 << i)
            new_sums.append((new_s, new_m))
        sums_B.extend(new_sums)
    
    # Sort sums_A
    # sums_A is already sorted by construction? No, the doubling method doesn't guarantee global sort.
    # We need to sort.
    sums_A.sort(key=lambda x: x[0])
    
    # For each sum in B, we want the largest sum in A such that sum_A + sum_B < L.
    # i.e., sum_A < L - sum_B.
    
    # We can use binary search on sums_A.
    # But we need to iterate over all sums_B (2^24 = 16.7M).
    # 16.7M * log(8.4M) ~ 16.7M * 23 ~ 380M operations.
    # This is likely too slow in Python.
    
    # We need a faster method.
    
    # Idea: 
    # Since we want to maximize sum_A + sum_B < L,
    # and sums_A is sorted, for a fixed sum_B, the best sum_A is the largest one < L - sum_B.
    # As sum_B increases, the target L - sum_B decreases.
    # So we can iterate sum_B in descending order?
    # If we sort sums_B in descending order, the target for sum_A increases? No.
    # If sum_B is large, target is small.
    # If sum_B is small, target is large.
    
    # Let's sort sums_B in ascending order.
    sums_B.sort(key=lambda x: x[0])
    
    # We can use a two-pointer technique?
    # Let i be index in sums_A (from end, largest), j be index in sums_B (from start, smallest).
    # We want to find pairs (a, b) such that a.sum + b.sum < L.
    # We want to maximize a.sum + b.sum.
    
    # Since sums_A and sums_B are sorted ascending:
    # Start with i = len(sums_A) - 1 (largest A)
    # Iterate j from 0 to len(sums_B) - 1 (smallest to largest B)
    # For each j, we need the largest i such that sums_A[i].sum + sums_B[j].sum < L.
    # As j increases, sums_B[j].sum increases, so the max allowed sums_A[i].sum decreases.
    # So i can only decrease (or stay same).
    # This is a classic two-pointer approach!
    
    best_sum = -1
    best_a_mask = 0
    best_b_mask = 0
    
    i = len(sums_A) - 1
    for j in range(len(sums_B)):
        b_sum, b_mask = sums_B[j]
        # We need sums_A[i].sum < L - b_sum
        # Since i is the largest valid index for previous j, and b_sum increased,
        # the valid i must be <= current i.
        while i >= 0 and sums_A[i][0] + b_sum >= L:
            i -= 1
            
        if i < 0:
            break # No more valid A sums for this or larger B sums
            
        a_sum, a_mask = sums_A[i]
        total_sum = a_sum + b_sum
        
        if total_sum > best_sum:
            best_sum = total_sum
            best_a_mask = a_mask
            best_b_mask = b_mask
            
        # Time check
        if time.time() > deadline - 5:
            break
            
    if best_sum < 0:
        # Fallback
        return [0] # Should not happen
        
    # Reconstruct indices
    selected = []
    for i in range(mid):
        if best_a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if best_b_mask & (1 << i):
            selected.append(mid + i)
            
    # Verify with exact fractions
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1:
        return selected
    else:
        # This should not happen due to integer check, but just in case
        # Try a simple greedy fallback
        selected_greedy = []
        total_greedy = Fraction(0)
        for k in range(n):
            if total_greedy + Fraction(1, denominators[k]) < 1:
                selected_greedy.append(k)
                total_greedy += Fraction(1, denominators[k])
        return selected_greedy
