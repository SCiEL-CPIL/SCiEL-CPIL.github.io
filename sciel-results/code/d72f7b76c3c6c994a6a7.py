import time
from math import gcd
from functools import reduce
from bisect import bisect_left

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]
    
    # Split items into two halves
    mid = n // 2
    half1 = weights[:mid]
    half2 = weights[mid:]
    
    # Generate all subset sums and masks for half1
    def gen_sums(weights):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights):
            # Use list comprehension for speed
            # But appending to existing list is faster than creating new list and extending?
            # Actually, creating new lists and extending is O(N) copy.
            # Let's just do:
            current_len = len(sums)
            for j in range(current_len):
                sums.append(sums[j] + w)
                masks.append(masks[j] | (1 << i))
        return sums, masks

    deadline = time.time() + 55
    
    # Generate for half1
    sums1, masks1 = gen_sums(half1)
    
    # Generate for half2
    sums2, masks2 = gen_sums(half2)
    
    # Sort half2 sums to enable binary search
    # We need to keep track of the mask associated with each sum.
    # If there are multiple masks for the same sum, we only need one (any).
    # So we can create a list of (sum, mask) pairs, sort by sum, and remove duplicates?
    # Or just sort the list of indices by sum.
    
    # To save memory and time, let's create a list of (sum, mask) for half2
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Remove duplicate sums? Not strictly necessary for correctness, but speeds up search if many duplicates.
    # However, sums are likely distinct enough.
    
    # We want to find max s1 + s2 < L
    best_deficit = L
    best_mask1 = 0
    best_mask2 = 0
    
    # Extract just the sums for binary search
    sorted_sums2 = [p[0] for p in pairs2]
    
    for i in range(len(sums1)):
        s1 = sums1[i]
        if s1 >= L:
            continue
            
        target = L - s1
        # We want the largest s2 such that s2 < target
        # bisect_left returns the index where target would be inserted to keep sorted order.
        # So index = bisect_left(sorted_sums2, target)
        # The candidate is at index - 1
        idx = bisect_left(sorted_sums2, target)
        
        if idx > 0:
            # The largest sum strictly less than target is at idx - 1
            # But we must ensure sorted_sums2[idx-1] < target
            # bisect_left ensures sorted_sums2[idx-1] < target if idx > 0
            # Wait, if target is present, bisect_left returns the first occurrence.
            # So sorted_sums2[idx] == target.
            # We want < target. So idx - 1 is correct.
            
            s2 = sorted_sums2[idx - 1]
            total_sum = s1 + s2
            deficit = L - total_sum
            
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask1 = masks1[i]
                best_mask2 = pairs2[idx - 1][1]
                
            # Early exit if deficit is 1 (which is 1/L in fraction, but we are working in integers)
            # Actually deficit is in units of 1/L. Smallest positive deficit is 1.
            if best_deficit == 1:
                break

    # Reconstruct indices
    # mask1 corresponds to indices 0..mid-1
    # mask2 corresponds to indices mid..n-1
    
    final_indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            final_indices.append(i)
            
    for i in range(len(half2)):
        if best_mask2 & (1 << i):
            final_indices.append(mid + i)
            
    # If no valid subset found (should not happen as empty set is valid but deficit L), 
    # or if we need to handle the case where best_deficit is still L (empty set)
    if not final_indices:
        # Fallback: return a single small element? 
        # The problem asks for nonempty.
        # If the best found was the empty set (deficit L), we should pick the smallest weight?
        # No, we want to minimize deficit, so maximize sum < L.
        # If the algorithm failed to find a better sum than 0, we return the best single element.
        # But gen_sums includes 0. So best_deficit will be at most L - max_single_weight.
        # So final_indices should not be empty unless L is very small or something.
        # But L is huge.
        pass
        
    return final_indices
