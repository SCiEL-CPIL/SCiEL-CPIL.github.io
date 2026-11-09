import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    N = 10000000

    # Sieve for sigma
    sigma = np.zeros(N + 1, dtype=np.int64)

    # Efficient sieve using numpy slicing
    for i in range(1, N + 1):
        if time.time() > deadline - 20.0:
            break
        sigma[i::i] += i

    if time.time() > deadline - 15.0:
        return [11, 6]

    # Group numbers by their sigma value
    # We only need to consider numbers that are close to the ratio 3/2
    # i.e., a ≈ 1.5 * b.
    # This means b is in [2N/5, N] roughly, and a in [2N/5, N].
    # Actually, a/b ≈ 3/2 => a ≈ 1.5b.
    # If b is small, a is small. If b is large, a is large.
    # To minimize |a/b - 3/2| = |2a - 3b| / (2b), we want small |2a - 3b| and large b.
    
    # Let's build a dictionary: sigma_value -> list of numbers n with that sigma.
    # But storing lists for 10^7 numbers might be memory heavy if we use Python lists.
    # However, we can just store the numbers in a flat array and sort by (sigma, n).
    # Or, we can iterate through the sigma array and group.
    
    # Memory check: 
    # sigma is int64 (8 bytes) * 10^7 = 80 MB.
    # indices is int32 (4 bytes) * 10^7 = 40 MB.
    # Total ~120 MB. Should be fine.
    
    # Create array of indices 1..N
    indices = np.arange(1, N + 1, dtype=np.int32)
    
    # Sort indices by sigma value
    # argsort is stable, but we don't need stability for grouping, just to group equal sigmas.
    # To find pairs, we need to look within groups of equal sigma.
    
    # Sorting 10^7 elements might take some time (maybe 2-5 seconds).
    # Let's check time.
    if time.time() > deadline - 10.0:
        return [11, 6]
        
    order = np.argsort(sigma[indices], kind='mergesort')
    
    if time.time() > deadline - 5.0:
        return [11, 6]
        
    sorted_indices = indices[order]
    sorted_sigmas = sigma[indices][order]
    
    best_err_num = 0
    best_err_den = 1
    best_a = 0
    best_b = 0
    
    # Iterate through groups of equal sigma
    i = 0
    while i < N:
        if time.time() > deadline - 2.0:
            break
            
        j = i
        while j < N and sorted_sigmas[j] == sorted_sigmas[i]:
            j += 1
            
        # Group is sorted_indices[i:j]
        # We want to find a, b in this group such that a != b and |2a - 3b| is minimized, 
        # and among those, b is maximized (to minimize denominator 2b).
        # Actually, we want to minimize |2a - 3b| / (2b).
        # Let k = |2a - 3b|. We want small k and large b.
        
        # Since the group is sorted by index (because we sorted indices by sigma, and within equal sigma, 
        # the original indices were increasing, and mergesort is stable), 
        # sorted_indices[i:j] is sorted in increasing order of the number value.
        
        # Let's extract the group
        # If the group is small, we can do a double loop or two-pointer.
        # If the group is large, we need to be efficient.
        
        # Two-pointer technique:
        # We want to find a, b in the group such that 2a ≈ 3b.
        # Let A = sorted_indices[i:j].
        # We can iterate through b in A, and binary search for a ≈ 1.5b in A.
        # Or, since A is sorted, we can use two pointers.
        
        # Let's use a two-pointer approach:
        # For each b in A (from largest to smallest), we want to find a in A closest to 1.5b.
        # Since A is sorted, we can maintain a pointer for a.
        
        # But doing this for every group might be slow if groups are large.
        # Most groups are size 1. Some are size 2, etc.
        # Only groups with size >= 2 can contribute to a pair.
        
        if j - i < 2:
            i = j
            continue
            
        # Extract group as list for faster access? Or use numpy slicing?
        # Numpy slicing is fast.
        group = sorted_indices[i:j]
        
        # We want to minimize |2a - 3b| / (2b).
        # Let's iterate b from largest to smallest in the group.
        # For each b, find a in the group that minimizes |2a - 3b|.
        # Since the group is sorted, we can binary search for 1.5b.
        
        # To be efficient, we can pre-compute 3*b for all b in the group, 
        # and 2*a for all a in the group.
        # Then we want to find pairs (2a, 3b) that are close.
        
        # Let's create arrays:
        # targets = 3 * group
        # sources = 2 * group
        # We want to find i, j such that |sources[i] - targets[j]| is minimized, 
        # and among those, group[j] (which is b) is maximized.
        
        # Actually, we can just iterate through b in the group (from largest to smallest)
        # and for each b, find the a in the group closest to 1.5b.
        
        # Since the group is sorted, we can use np.searchsorted.
        
        # Optimization: if the group is small (e.g., < 100), a double loop is fine.
        # If large, use searchsorted.
        
        # Let's use searchsorted for all groups with size >= 2.
        
        # We iterate b from largest to smallest to prioritize larger b.
        # For each b, we calculate target = 1.5 * b = 3b/2.
        # We search for this target in the group (which is sorted).
        # The closest a will be at pos or pos-1.
        
        # Note: a must be in the group.
        
        for b_idx in range(len(group) - 1, -1, -1):
            b_val = group[b_idx]
            target = 1.5 * b_val
            
            # Search for target in group
            pos = np.searchsorted(group, target)
            
            # Check pos and pos-1
            candidates = []
            if pos < len(group):
                candidates.append(group[pos])
            if pos > 0:
                candidates.append(group[pos - 1])
                
            # Also check if b_val itself is a candidate (a=b, but we need a!=b? 
            # The task says "positive integers a, b". It doesn't explicitly say a != b.
            # But if a=b, then a/b = 1, error = |1 - 1.5| = 0.5.
            # We want to minimize error. 0 is invalid.
            # If a=b, sigma(a)=sigma(b) is trivially true.
            # But we want to minimize |a/b - 3/2|.
            # If a=b, error is 0.5.
            # We can likely find pairs with much smaller error.
            # However, we must ensure a != b if we want to avoid trivial solutions?
            # The task says "Approximate a ratio...". Usually implies distinct or at least non-trivial.
            # But "Minimize |a/b - 3/2|". If a=b is allowed, and there is no pair with error < 0.5, 
            # then a=b is the best. But 0.5 is large.
            # Let's assume a and b should be distinct to get a good score.
            # The current code allows a=b? 
            # In the current code: a = (3b+k)//2. If k=0, a=1.5b. If b is even, a=1.5b is int.
            # Then sigma(a)==sigma(b). If a=b, then 1.5b=b => b=0, impossible.
            # So the current code naturally avoids a=b because k=0 is skipped (range starts at 1).
            # So a != b is enforced by k != 0.
            # In our new approach, we must explicitly check a != b.
            
            for a_val in candidates:
                if a_val == b_val:
                    continue
                    
                # Calculate error
                # error = |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a_val - 3 * b_val)
                den = 2 * b_val
                
                if num == 0:
                    continue # Invalid
                    
                # Compare with best
                if best_a == 0 or num * best_err_den < best_err_num * den:
                    best_err_num = num
                    best_err_den = den
                    best_a = a_val
                    best_b = b_val
                    
            # Early stopping?
            # If we found a solution with num=1, that's the best possible numerator.
            # We can only improve by increasing den (b).
            # So we should continue to check larger b?
            # Wait, we are iterating b from largest to smallest.
            # So we are checking larger b first.
            # If we find num=1 with the largest b, that's the best possible.
            # But maybe a smaller b with num=1 has smaller den? No, den=2b, so larger b is better.
            # So if we find num=1, we can't do better on numerator.
            # We can only improve if we find another num=1 with larger b.
            # But we are iterating from largest b down.
            # So the first num=1 we find is with the largest b in the group.
            # Can we find a num=1 with larger b in a different group?
            # Yes. So we must continue checking other groups.
            # But within this group, once we find num=1, we can stop checking smaller b in this group?
            # Yes, because for smaller b, den is smaller, so error is larger (if num is same).
            # And if num is larger, error is even larger.
            # So for this group, if we find num=1, we can break the inner loop.
            # But we still need to check other groups.
            
            if best_err_num == 1:
                # We found the best possible numerator.
                # Since we iterate b from largest to smallest, this is the best b in this group.
                # We can break the inner loop.
                break
        
        i = j

    if best_a != 0:
        return [int(best_a), int(best_b)]
        
    # Fallback
    return [11, 6]
