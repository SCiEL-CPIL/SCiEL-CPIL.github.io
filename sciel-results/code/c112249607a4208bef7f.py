import time
from math import lcm
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as floats for fast search
    recip_floats = [1.0 / d for d in denominators]
    
    # Split into two halves for Meet-in-the-Middle
    # Indices 0-22 correspond to denominators 2-24
    # Indices 23-47 correspond to denominators 25-48
    mid = 23
    n1 = mid
    n2 = n - mid
    
    # Generate subset sums for first half
    # We want sums < 1.0
    sums1 = [0.0]
    masks1 = [0]
    
    for i in range(n1):
        w = recip_floats[i]
        bit = 1 << i
        # Use list comprehension for speed, but we need to filter by sum < 1
        # Current size might be large. 
        # Optimization: Only add if sum + w < 1.0
        new_sums = []
        new_masks = []
        for idx in range(len(sums1)):
            ns = sums1[idx] + w
            if ns < 1.0:
                new_sums.append(ns)
                new_masks.append(masks1[idx] | bit)
        sums1.extend(new_sums)
        masks1.extend(new_masks)
        
        if time.time() > deadline - 30:
            break

    # Generate subset sums for second half
    sums2 = [0.0]
    masks2 = [0]
    
    for i in range(n2):
        w = recip_floats[mid + i]
        bit = 1 << i
        new_sums = []
        new_masks = []
        for idx in range(len(sums2)):
            ns = sums2[idx] + w
            if ns < 1.0:
                new_sums.append(ns)
                new_masks.append(masks2[idx] | bit)
        sums2.extend(new_sums)
        masks2.extend(new_masks)
        
        if time.time() > deadline - 30:
            break

    # Sort second half by sum for binary search
    order2 = sorted(range(len(sums2)), key=lambda i: sums2[i])
    sorted_sums2 = [sums2[i] for i in order2]
    sorted_masks2 = [masks2[i] for i in order2]
    
    best_sum = -1.0
    best_m1 = 0
    best_m2 = 0
    
    # For each sum in first half, find best complement in second half
    for idx1 in range(len(sums1)):
        if time.time() > deadline - 10:
            break
        s1 = sums1[idx1]
        m1 = masks1[idx1]
        # We want s1 + s2 < 1.0, so s2 < 1.0 - s1
        target = 1.0 - s1
        # Find rightmost index in sorted_sums2 that is < target
        # bisect_left gives first index >= target. We want the one before that.
        pos = bisect.bisect_left(sorted_sums2, target)
        if pos > 0:
            s2 = sorted_sums2[pos - 1]
            m2 = sorted_masks2[pos - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2
    
    # Reconstruct best subset from MITM
    result_indices = []
    for i in range(n1):
        if best_m1 & (1 << i):
            result_indices.append(i)
    for i in range(n2):
        if best_m2 & (1 << i):
            result_indices.append(mid + i)
    
    if not result_indices:
        return [0]
    
    # Local Search to polish the result
    # Use floats for speed
    current_subset = set(result_indices)
    current_sum = sum(recip_floats[i] for i in current_subset)
    
    def try_local_search(subset, time_limit):
        current_subset = set(subset)
        current_sum = sum(recip_floats[i] for i in current_subset)
        
        # 1-Add move: Add one element if sum < 1
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_gain = -1.0
            best_add = -1
            for i in range(n):
                if i not in current_subset:
                    new_sum = current_sum + recip_floats[i]
                    if new_sum < 1.0:
                        gain = recip_floats[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_add = i
            if best_add != -1:
                current_subset.add(best_add)
                current_sum += best_gain
                improved = True
        
        # 1-Swap move: Remove one, add one
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_gain = -1.0
            best_swap = None
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]
            
            for i in in_sub:
                for j in not_in_sub:
                    new_sum = current_sum - recip_floats[i] + recip_floats[j]
                    if new_sum < 1.0:
                        gain = recip_floats[j] - recip_floats[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (i, j)
            
            if best_swap:
                i, j = best_swap
                current_subset.remove(i)
                current_subset.add(j)
                current_sum += best_gain
                improved = True
        
        # 2-Add move: Add two elements (if possible)
        # This is O(N^2), might be slow if N is large, but N=47 is small.
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_gain = -1.0
            best_adds = None
            not_in_sub = [i for i in range(n) if i not in current_subset]
            
            for i1 in range(len(not_in_sub)):
                for i2 in range(i1 + 1, len(not_in_sub)):
                    a, b = not_in_sub[i1], not_in_sub[i2]
                    new_sum = current_sum + recip_floats[a] + recip_floats[b]
                    if new_sum < 1.0:
                        gain = recip_floats[a] + recip_floats[b]
                        if gain > best_gain:
                            best_gain = gain
                            best_adds = (a, b)
            
            if best_adds:
                a, b = best_adds
                current_subset.add(a)
                current_subset.add(b)
                current_sum += best_gain
                improved = True
                
        return current_subset, current_sum

    # Run local search from MITM result
    if time.time() < deadline - 5:
        best_subset, best_sum_float = try_local_search(result_indices, deadline - 2)
    
    # Verify with Fraction
    # We need to ensure the sum is strictly < 1
    # Find the subset with the smallest deficit (1 - sum)
    
    # Re-evaluate best_subset with Fraction
    best_deficit = Fraction(1, 1)
    best_final_subset = best_subset
    
    # It's possible local search found a better float sum, but we need to check exactness.
    # Since we used floats, there might be precision issues.
    # However, the denominators are small, so float precision should be sufficient for comparison.
    # But let's be safe and check the top candidates if we had a list. 
    # Here we just have one best_subset.
    
    # Let's also check if the MITM result was better (in case local search failed or timed out)
    mitm_subset = set(result_indices)
    
    candidates = [best_subset, mitm_subset]
    # Add a few random perturbations of best_subset if time permits
    if time.time() < deadline - 2:
        import random
        rng = random.Random(42)
        for _ in range(10):
            perturbed = set(best_subset)
            if perturbed:
                # Remove 1 random
                to_remove = rng.choice(list(perturbed))
                perturbed.remove(to_remove)
                # Add 1 random not in set
                not_in = [i for i in range(n) if i not in perturbed]
                if not_in:
                    to_add = rng.choice(not_in)
                    perturbed.add(to_add)
            candidates.append(perturbed)
    
    min_deficit = Fraction(1, 1)
    final_subset = best_subset
    
    for sub in candidates:
        if not sub:
            continue
        s = sum(Fraction(1, denominators[i]) for i in sub)
        if s < 1:
            deficit = 1 - s
            if deficit < min_deficit:
                min_deficit = deficit
                final_subset = sub
    
    if not final_subset:
        return [0]
        
    return list(final_subset)
