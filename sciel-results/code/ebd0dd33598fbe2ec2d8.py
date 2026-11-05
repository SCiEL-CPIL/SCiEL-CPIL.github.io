import time
import sys
from fractions import Fraction
import bisect

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget to be safe within 90s
    
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
    # 23 and 24 items
    mid = 23
    A_weights = weights[:mid]
    B_weights = weights[mid:]
    
    # Generate subset sums for A
    # List of (sum, mask)
    sums_A = [(0, 0)]
    for i, w in enumerate(A_weights):
        current_len = len(sums_A)
        # Iterate backwards to avoid index issues if we were modifying in place, 
        # but we are appending, so forward is fine.
        for k in range(current_len):
            s, m = sums_A[k]
            new_s = s + w
            if new_s < L:
                sums_A.append((new_s, m | (1 << i)))
                
    # Generate subset sums for B
    sums_B = [(0, 0)]
    for i, w in enumerate(B_weights):
        current_len = len(sums_B)
        for k in range(current_len):
            s, m = sums_B[k]
            new_s = s + w
            if new_s < L:
                sums_B.append((new_s, m | (1 << i)))
                
    # Sort A by sum descending
    sums_A.sort(key=lambda x: x[0], reverse=True)
    # Sort B by sum ascending
    sums_B.sort(key=lambda x: x[0])
    
    # Extract B sums for quick access
    B_sums = [x[0] for x in sums_B]
    
    best_sum = -1
    best_a_mask = 0
    best_b_mask = 0
    
    # Two-pointer technique
    # A is descending, B is ascending
    # For each A[i], we want largest B[j] such that A[i] + B[j] < L
    # i.e., B[j] < L - A[i]
    # Since B is sorted ascending, we can maintain a pointer j
    # As A[i] decreases, L - A[i] increases, so the valid j range expands to the right.
    
    j = 0
    len_B = len(sums_B)
    
    for i in range(len(sums_A)):
        a_sum, a_mask = sums_A[i]
        
        # We want the largest index j such that B_sums[j] < L - a_sum
        # Since B_sums is sorted ascending, we can advance j
        # Condition: B_sums[j] < L - a_sum  => B_sums[j] <= L - a_sum - 1
        
        limit = L - a_sum - 1
        
        # Advance j as far as possible
        # j starts from previous best position
        while j < len_B - 1 and B_sums[j + 1] <= limit:
            j += 1
            
        # Now B_sums[j] is the largest valid sum for this a_sum
        # Check if it's a valid candidate
        if B_sums[j] <= limit:
            total = a_sum + B_sums[j]
            if total > best_sum:
                best_sum = total
                best_a_mask = a_mask
                best_b_mask = sums_B[j][1]
        
        # Time check
        if time.time() > deadline - 5:
            break

    if best_sum < 0:
        # Fallback
        return [0, 1, 5]

    # Reconstruct indices
    selected = set()
    for i in range(mid):
        if best_a_mask & (1 << i):
            selected.add(i)
    for i in range(len(B_weights)):
        if best_b_mask & (1 << i):
            selected.add(mid + i)
            
    result = sorted(selected)
    
    # Verify
    total_frac = sum(Fraction(1, denominators[k]) for k in result)
    if total_frac < 1:
        return result
    else:
        return [0, 1, 5]

if __name__ == "__main__":
    print(construct())
