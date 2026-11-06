import time
from math import lcm
import bisect
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are L // d
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2  # 23
    half1 = list(range(mid))
    half2 = list(range(mid, n))
    
    w1 = [weights[i] for i in half1]
    w2 = [weights[i] for i in half2]
    
    # Use numpy for faster subset sum generation if possible, but pure python list is often faster for this specific pattern
    # Let's stick to the efficient pure python list extension but optimize the inner loop
    
    def gen_sums(weights_list):
        """
        Generate all subset sums < L with their masks.
        Returns list of (sum, mask).
        """
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            if time.time() > deadline:
                break
            bit = 1 << i
            current_len = len(sums)
            new_sums = []
            # Local variables for speed
            L_local = L
            for j in range(current_len):
                s, m = sums[j]
                ns = s + w
                if ns < L_local:
                    new_sums.append((ns, m | bit))
            sums.extend(new_sums)
        return sums

    print("Generating sums for half 1...")
    sums1 = gen_sums(w1)
    print(f"Half 1 size: {len(sums1)}")
    
    if time.time() > deadline:
        # Fallback if generation takes too long
        return [0] 

    print("Generating sums for half 2...")
    sums2 = gen_sums(w2)
    print(f"Half 2 size: {len(sums2)}")
    
    if time.time() > deadline:
        # Fallback
        return [0]

    # Sort sums2 by sum
    sums2.sort(key=lambda x: x[0])
    sums2_vals = [s[0] for s in sums2]
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    # Iterate through sums1
    # Optimization: We can also sort sums1, but we need to check all anyway.
    # However, if we sort sums1 descending, we might find the best earlier?
    # No, we need the max of s1+s2 where s1+s2 < L.
    # For a fixed s1, the best s2 is the largest s2 < L - s1.
    # So we just iterate all s1.
    
    # To speed up, we can pre-filter sums1? No, all are < L.
    
    # Let's use a tighter loop
    bisect_right = bisect.bisect_right
    L_local = L
    best_sum_local = -1
    best_m1_local = 0
    best_m2_local = 0
    
    for s1, m1 in sums1:
        if time.time() > deadline:
            break
            
        # We need s2 such that s1 + s2 < L
        # s2 <= L - 1 - s1
        target = L_local - 1 - s1
        if target < 0:
            continue
            
        # Find the largest s2 <= target
        idx = bisect_right(sums2_vals, target)
        if idx > 0:
            s2, m2 = sums2[idx - 1]
            total = s1 + s2
            if total > best_sum_local:
                best_sum_local = total
                best_m1_local = m1
                best_m2_local = m2
                
    best_sum = best_sum_local
    best_m1 = best_m1_local
    best_m2 = best_m2_local

    # Reconstruct indices
    if best_sum == -1:
        # Fallback: pick a single small denominator
        # Max weight is L/2. L/2 < L. So [0] is valid.
        best_indices = [0]
    else:
        best_indices = []
        for i in range(mid):
            if best_m1 & (1 << i):
                best_indices.append(i)
        for i in range(mid, n):
            if best_m2 & (1 << (i - mid)):
                best_indices.append(i)
                
    # Validation
    if not best_indices:
        best_indices = [0]
        
    # Verify
    s = sum(weights[i] for i in best_indices)
    if s >= L:
        # This shouldn't happen, but if it does, remove one item
        # This is a safeguard
        best_indices.pop()
        
    # Local search polish: try to improve by swapping elements
    if time.time() < deadline - 5.0:
        current_sum = sum(weights[i] for i in best_indices)
        in_set = set(best_indices)
        not_in_set = [i for i in range(n) if i not in in_set]
        
        improved = True
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            # Try removing one element and adding another
            for rem in list(in_set):
                for add in not_in_set:
                    new_sum = current_sum - weights[rem] + weights[add]
                    if new_sum < L and new_sum > current_sum:
                        # Accept the improvement
                        best_indices.remove(rem)
                        best_indices.append(add)
                        in_set.remove(rem)
                        not_in_set.remove(add)
                        not_in_set.append(rem)
                        in_set.add(add)
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        # Update best_sum
        if current_sum > best_sum:
            best_sum = current_sum
    
    print(f"Best sum: {best_sum}/{L}")
    print(f"Indices: {best_indices}")
    
    return best_indices
