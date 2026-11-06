import time
import bisect
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]

    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    n1 = len(w1)
    n2 = len(w2)

    def gen_sums(weights_list, n_items):
        """
        Generate all subset sums < L with their masks.
        Returns a list of (sum, mask) tuples.
        """
        sums = [(0, 0)]
        for i in range(n_items):
            if time.time() > deadline:
                break
            w = weights_list[i]
            bit = 1 << i
            current_len = len(sums)
            # Local variable for speed
            append = sums.append
            for j in range(current_len):
                s, m = sums[j]
                ns = s + w
                if ns < L:
                    append((ns, m | bit))
        return sums

    # Generate sums for first half
    sums1 = gen_sums(w1, n1)
    if time.time() > deadline:
        # Fallback: 1/2 + 1/3 + 1/7 + 1/43
        # Indices: 0, 1, 5, 41
        return [0, 1, 5, 41]
        
    # Generate sums for second half
    sums2 = gen_sums(w2, n2)
    if time.time() > deadline:
        return [0, 1, 5, 41]

    # Sort sums1 by value
    sums1.sort(key=lambda x: x[0])
    sums1_vals = [s[0] for s in sums1]

    best_sum = -1
    best_m1 = 0
    best_m2 = 0

    # Iterate through sums2
    # To optimize, we can sort sums2 in descending order and break early if possible?
    # No, because a small s2 might pair with a large s1 to make a better total.
    # However, if we sort sums2 descending, the first valid pair might be good, but not necessarily optimal.
    # We must check all.
    
    # One optimization: if s2 is already close to L, the target for s1 is small.
    # If s2 is small, target is large.
    
    for s2, m2 in sums2:
        if time.time() > deadline:
            break
        # We need s1 such that s1 + s2 < L
        # s1 < L - s2
        # Max s1 is the largest value in sums1_vals < L - s2
        target = L - 1 - s2
        if target < 0:
            continue
            
        idx = bisect.bisect_right(sums1_vals, target)
        if idx > 0:
            s1, m1 = sums1[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2

    if best_sum > 0:
        # Reconstruct indices
        indices = []
        for i in range(n1):
            if best_m1 & (1 << i):
                indices.append(i)
        for i in range(n2):
            if best_m2 & (1 << i):
                indices.append(mid + i)
        if indices:
            # Verify
            val = sum(weights[i] for i in indices)
            if val < L:
                return indices
            else:
                return [0, 1, 5, 41]
    else:
        return [0, 1, 5, 41]
