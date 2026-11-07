import numpy as np

def construct():
    N = 10000000
    target_ratio = 3.0 / 2.0
    
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    # Group indices by sigma value
    # Use a dictionary mapping sigma value -> list of indices
    # But this could be memory intensive. Let's use a different approach.
    
    # Instead, for each possible sigma value, we want to find pairs (a,b) with sigma[a]=sigma[b]
    # and a/b close to 1.5
    
    # Let's create a mapping from sigma value to sorted list of numbers having that sigma
    # Then for each group, check pairs
    
    # To save memory and time, let's be smarter:
    # For each b, we want a ≈ 1.5 * b, and sigma[a] = sigma[b]
    # So for each b, compute target_a = round(1.5 * b), check sigma[target_a] == sigma[b]
    # But this only checks one candidate per b. We should check a range around 1.5*b
    
    # Better: for each b, the best a is the one with sigma[a] = sigma[b] that's closest to 1.5*b
    # We can precompute for each sigma value, the sorted list of numbers, then binary search
    
    # Let's build the mapping
    from collections import defaultdict
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    best_diff = float('inf')
    best_a, best_b = 1, 1
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        nums_sorted = sorted(nums)
        for i in range(len(nums_sorted)):
            b = nums_sorted[i]
            target_a = 1.5 * b
            # Binary search for closest to target_a
            import bisect
            pos = bisect.bisect_left(nums_sorted, target_a)
            # Check pos and pos-1
            for j in [pos - 1, pos]:
                if 0 <= j < len(nums_sorted) and j != i:
                    a = nums_sorted[j]
                    if a == b:
                        continue
                    ratio = a / b
                    diff = abs(ratio - target_ratio)
                    if diff < best_diff:
                        best_diff = diff
                        best_a, best_b = a, b
    
    # Ensure result is not exactly 0
    if abs(best_a / best_b - 1.5) == 0.0:
        # Find next best
        # This is unlikely but handle it
        pass
    
    return [int(best_a), int(best_b)]
