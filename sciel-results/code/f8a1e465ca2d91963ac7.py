import time
import sys
from fractions import Fraction
import bisect

def construct():
    start = time.time()
    deadline = start + 82  # 82 second budget to be safe within 90s
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
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
    
    # Split into two sets A and B
    mid = n // 2
    A_weights = weights[:mid]
    B_weights = weights[mid:]
    
    # Generate subset sums for A
    # Using list of (sum, mask)
    sums_A = [(0, 0)]
    for i, w in enumerate(A_weights):
        # Optimization: iterate backwards to avoid modifying list while iterating?
        # No, we extend.
        current_len = len(sums_A)
        for k in range(current_len):
            s, m = sums_A[k]
            sums_A.append((s + w, m | (1 << i)))
            
    # Sort A by sum
    sums_A.sort(key=lambda x: x[0])
    
    # Generate subset sums for B
    sums_B = [(0, 0)]
    for i, w in enumerate(B_weights):
        current_len = len(sums_B)
        for k in range(current_len):
            s, m = sums_B[k]
            sums_B.append((s + w, m | (1 << i)))
            
    # Sort B by sum
    sums_B.sort(key=lambda x: x[0])
    
    # Two-pointer technique to find max A_sum + B_sum < L
    # i points to largest valid A
    # j points to largest valid B for current i?
    # Let's iterate i from largest to smallest.
    # As A_sum decreases, the max allowed B_sum increases.
    # So j can only move forward (towards larger B_sums) or stay.
    # Wait.
    # Condition: A[i] + B[j] < L
    # If we decrease i (smaller A), we can allow larger B.
    # So if we start with i = len(A)-1 (largest A), we need small B.
    # Let j = 0 (smallest B).
    # If A[i] + B[j] < L, we update best. Can we try larger B?
    # Yes, for this i, we want the *largest* B such that A[i] + B < L.
    # So for a fixed i, we can binary search B, or advance j.
    # If we advance i downwards, the max allowed B increases.
    # So j (index in B) should only increase.
    
    best_sum = -1
    best_a_idx = 0
    best_b_idx = 0
    
    # Create a separate list of B sums for faster bisect
    B_sums = [x[0] for x in sums_B]
    
    # Iterate A from largest to smallest
    for i in range(len(sums_A) - 1, -1, -1):
        a_sum, a_mask = sums_A[i]
        
        # Find the largest j such that a_sum + sums_B[j][0] < L
        # Since sums_B is sorted ascending, we can binary search.
        # But we can also use the fact that as a_sum decreases, the limit L - a_sum increases.
        # So the optimal j for current i is >= optimal j for previous i.
        
        # Binary search for the largest index j' such that sums_B[j'][0] < L - a_sum
        target = L - a_sum
        # We want largest j where sums_B[j][0] < target
        
        pos = bisect.bisect_left(B_sums, target)
        
        # The largest valid index is pos - 1
        if pos > 0:
            candidate_j = pos - 1
            b_sum, b_mask = sums_B[candidate_j]
            total = a_sum + b_sum
            if total > best_sum:
                best_sum = total
                best_a_idx = i
                best_b_idx = candidate_j
                
        # Time check
        if time.time() > deadline - 10:
            break

    if best_sum < 0:
        # Fallback to a simple valid set
        # 1/2 + 1/3 + 1/7 = 41/42 < 1
        return [0, 1, 5]

    # Reconstruct indices
    a_mask = sums_A[best_a_idx][1]
    b_mask = sums_B[best_b_idx][1]
    
    selected = []
    for i in range(mid):
        if a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if b_mask & (1 << i):
            selected.append(mid + i)
            
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1:
        return selected
    else:
        return [0, 1, 5]
