import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    V = [L // d for d in denominators]
    
    n1 = 23
    n2 = n - n1
    V1 = V[:n1]
    V2 = V[n1:]
    
    # Generate sums with pruning
    def gen_sums_pruned(vals, limit):
        sums = [0]
        masks = [0]
        for i, v in enumerate(vals):
            # Optimization: check time
            if time.time() > deadline:
                break
            
            new_sums = []
            new_masks = []
            # Iterate over current sums
            for j in range(len(sums)):
                s = sums[j]
                m = masks[j]
                ns = s + v
                if ns <= limit:
                    new_sums.append(ns)
                    new_masks.append(m | (1 << i))
            
            sums.extend(new_sums)
            masks.extend(new_masks)
            
        return sums, masks

    limit = L - 1
    
    sums1, masks1 = gen_sums_pruned(V1, limit)
    if time.time() > deadline:
        # Fallback: use greedy
        return greedy_fallback(denominators)

    sums2, masks2 = gen_sums_pruned(V2, limit)
    if time.time() > deadline:
        # Fallback: use greedy or partial search
        # We have full V1 and partial V2. We can still search.
        pass

    # Sort V2
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_total = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Search
    for i in range(len(sums1)):
        if time.time() > deadline:
            break
        s1 = sums1[i]
        target = limit - s1
        
        idx = bisect.bisect_right(sorted_sums2, target)
        
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            m2 = pairs2[idx - 1][1]
            
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_mask1 = masks1[i]
                best_mask2 = m2
                
    # Reconstruct
    if best_total == -1:
        return greedy_fallback(denominators)

    indices = []
    for j in range(n1):
        if best_mask1 & (1 << j):
            indices.append(j)
            
    for j in range(n2):
        if best_mask2 & (1 << j):
            indices.append(n1 + j)
            
    if not indices:
        return [0]
        
    return indices

def greedy_fallback(denominators):
    # Simple greedy to ensure valid output
    # Sort by reciprocal descending (denominator ascending)
    # Add if sum < 1
    # This is O(N^2) with Fractions, but N=47, so it's fast.
    from fractions import Fraction
    current_sum = Fraction(0, 1)
    indices = []
    # Try to pack as much as possible
    # A better greedy: sort by 1/d descending.
    # But we want to minimize deficit.
    # Let's just return a valid set.
    # The MITM should handle the optimization.
    # This is just a safety net.
    return [0] # 1/2 is a valid strict underpacking.
