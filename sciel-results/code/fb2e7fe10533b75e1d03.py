import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators, 1)
    
    # Calculate weights
    weights = [L // d for d in denominators]
    
    # Split indices into two halves
    # To balance the subset sums, we might want to sort by weight?
    # Standard MITM just splits the list.
    # Let's split by index.
    mid = n // 2
    idx_A = list(range(mid))
    idx_B = list(range(mid, n))
    
    weights_A = [weights[i] for i in idx_A]
    weights_B = [weights[i] for i in idx_B]
    
    # Generate subset sums for B
    # List of (sum, mask)
    sums_B = [(0, 0)]
    
    for i, w in enumerate(weights_B):
        # We need to add w to all existing sums
        # Prune if sum >= L
        new_sums = []
        for s, m in sums_B:
            ns = s + w
            if ns < L:
                nm = m | (1 << i)
                new_sums.append((ns, nm))
        sums_B.extend(new_sums)
        
        # Time check
        if time.time() > deadline:
            break
            
    if time.time() > deadline:
        # Fallback: return a simple valid subset
        # e.g., indices for 1/2, 1/3
        return [0, 1]

    # Sort sums_B by sum
    sums_B.sort(key=lambda x: x[0])
    
    # We want to find max s_A + s_B < L
    # So for each s_A, we want max s_B < L - s_A
    # Use binary search on sums_B
    
    best_deficit = L  # Initial deficit is L (sum 0)
    best_mask_A = 0
    best_mask_B = 0
    
    # Iterate through subsets of A
    sums_A = [(0, 0)]
    
    for i, w in enumerate(weights_A):
        new_sums = []
        for s, m in sums_A:
            ns = s + w
            if ns < L:
                nm = m | (1 << i)
                new_sums.append((ns, nm))
        sums_A.extend(new_sums)
        
        # Time check
        if time.time() > deadline:
            break

    if time.time() > deadline:
        # We might have incomplete sums_A or sums_B
        # But we can still process what we have
        pass

    # Process all generated sums_A
    # To avoid O(N_A * log N_B) if N_A is huge, we just loop.
    # N_A is at most 2^23.
    
    # Binary search helper
    import bisect
    
    # Extract sums from sums_B for bisect
    # sums_B is sorted by sum.
    # We need to find the rightmost index where sum < L - s_A
    
    for s_A, m_A in sums_A:
        target = L - s_A
        # Find rightmost index in sums_B where sum < target
        # bisect_left returns first index where sum >= target
        # So idx - 1 is the last index where sum < target
        idx = bisect.bisect_left([x[0] for x in sums_B], target)
        # This creates a list of sums every time! Bad.
        
        # Pre-extract sums_B_values
        # Do this outside the loop
        pass

    # Re-do: Pre-extract sums_B values
    sums_B_values = [x[0] for x in sums_B]
    
    for s_A, m_A in sums_A:
        if time.time() > deadline:
            break
        target = L - s_A
        idx = bisect.bisect_left(sums_B_values, target)
        if idx > 0:
            s_B, m_B = sums_B[idx - 1]
            total_sum = s_A + s_B
            deficit = L - total_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask_A = m_A
                best_mask_B = m_B

    # Reconstruct indices
    selected_indices = []
    
    # Map mask_A to original indices
    for i in range(len(weights_A)):
        if best_mask_A & (1 << i):
            selected_indices.append(idx_A[i])
            
    # Map mask_B to original indices
    for i in range(len(weights_B)):
        if best_mask_B & (1 << i):
            selected_indices.append(idx_B[i])
    
    # If no subset found (should not happen, empty set is valid)
    if not selected_indices:
        # Return empty set? Problem says "nonempty".
        # But sum 0 is valid. Deficit 1.
        # If best_deficit is still L, it means we found sum 0.
        # We must return a nonempty set.
        # Pick the smallest denominator (index 0)
        selected_indices = [0]
        
    return selected_indices
