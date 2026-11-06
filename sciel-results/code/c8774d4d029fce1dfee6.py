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
        
        # Strategy 1: Single swap (remove one, add one) - try all pairs, take best improvement
        improved = True
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            best_delta = 0
            best_rem = None
            best_add = None
            
            # Try removing one element and adding another - find best single swap
            for rem in list(in_set):
                for add in not_in_set:
                    delta = weights[add] - weights[rem]
                    if delta > 0 and current_sum + delta < L and delta > best_delta:
                        best_delta = delta
                        best_rem = rem
                        best_add = add
            
            if best_rem is not None:
                # Accept the best improvement
                best_indices.remove(best_rem)
                best_indices.append(best_add)
                in_set.remove(best_rem)
                not_in_set.remove(best_add)
                not_in_set.append(best_rem)
                in_set.add(best_add)
                current_sum += best_delta
                improved = True
            
            if not improved:
                break
        
        # Strategy 2: Double swap (remove two, add two)
        if time.time() < deadline - 5.0:
            improved2 = True
            max_iter2 = 50
            for _ in range(max_iter2):
                if time.time() > deadline - 5.0:
                    break
                improved2 = False
                
                best_delta2 = 0
                best_rem1 = None
                best_rem2 = None
                best_add1 = None
                best_add2 = None
                
                in_list = list(in_set)
                not_list = list(not_in_set)
                
                for i in range(len(in_list)):
                    for j in range(i+1, len(in_list)):
                        rem1 = in_list[i]
                        rem2 = in_list[j]
                        rem_weight = weights[rem1] + weights[rem2]
                        for k in range(len(not_list)):
                            for l in range(k+1, len(not_list)):
                                add1 = not_list[k]
                                add2 = not_list[l]
                                add_weight = weights[add1] + weights[add2]
                                delta = add_weight - rem_weight
                                if delta > 0 and current_sum + delta < L and delta > best_delta2:
                                    best_delta2 = delta
                                    best_rem1 = rem1
                                    best_rem2 = rem2
                                    best_add1 = add1
                                    best_add2 = add2
                
                if best_rem1 is not None:
                    best_indices.remove(best_rem1)
                    best_indices.remove(best_rem2)
                    best_indices.append(best_add1)
                    best_indices.append(best_add2)
                    in_set.discard(best_rem1)
                    in_set.discard(best_rem2)
                    not_in_set.remove(best_add1)
                    not_in_set.remove(best_add2)
                    not_in_set.append(best_rem1)
                    not_in_set.append(best_rem2)
                    in_set.add(best_add1)
                    in_set.add(best_add2)
                    current_sum += best_delta2
                    improved2 = True
                
                if not improved2:
                    break
        
        # Strategy 3: Add-only (try adding elements that fit)
        if time.time() < deadline - 5.0:
            improved3 = True
            while improved3 and time.time() < deadline - 5.0:
                improved3 = False
                for add in list(not_in_set):
                    if current_sum + weights[add] < L:
                        best_indices.append(add)
                        in_set.add(add)
                        not_in_set.remove(add)
                        current_sum += weights[add]
                        improved3 = True
                        break
        
        # Strategy 4: Remove-only (try removing elements to make room, then re-add)
        # This is handled by the swap strategies above
        
        # Update best_sum
        if current_sum > best_sum:
            best_sum = current_sum
    
    # Final validation and cleanup
    if not best_indices:
        best_indices = [0]
    
    # Ensure all indices are valid and distinct
    best_indices = sorted(set(best_indices))
    
    # Verify sum is strictly below L
    final_sum = sum(weights[i] for i in best_indices)
    if final_sum >= L:
        # Remove elements until valid
        while best_indices and final_sum >= L:
            idx_to_remove = best_indices.pop()
            final_sum -= weights[idx_to_remove]
    
    if not best_indices:
        best_indices = [0]
    
    print(f"Best sum: {best_sum}/{L}")
    print(f"Indices: {best_indices}")
    
    return best_indices
