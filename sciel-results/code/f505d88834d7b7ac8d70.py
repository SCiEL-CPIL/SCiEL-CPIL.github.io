"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s kill
    
    # Denominators are 2,3,...,48 -> indices 0..46
    n = 47
    denominators = list(range(2, 49))
    weights = [Fraction(1, d) for d in denominators]
    
    # Meet in the middle
    mid = n // 2  # 23
    left_items = list(range(mid))
    right_items = list(range(mid, n))
    
    # Enumerate all subsets of left half
    left_sums = []  # list of (sum, bitmask)
    for mask in range(1 << len(left_items)):
        s = Fraction(0)
        for i in range(len(left_items)):
            if mask & (1 << i):
                s += weights[left_items[i]]
        if s < 1:  # Only keep valid partial sums
            left_sums.append((s, mask))
    
    # Enumerate all subsets of right half
    right_sums = []  # list of (sum, bitmask)
    for mask in range(1 << len(right_items)):
        s = Fraction(0)
        for i in range(len(right_items)):
            if mask & (1 << i):
                s += weights[right_items[i]]
        if s < 1:
            right_sums.append((s, mask))
    
    # Sort right_sums by sum for binary search
    right_sums.sort(key=lambda x: x[0])
    right_values = [x[0] for x in right_sums]
    
    # For each left subset, find best right subset such that left_sum + right_sum < 1
    import bisect
    
    best_sum = Fraction(0)
    best_left_mask = 0
    best_right_mask = 0
    
    for ls, lmask in left_sums:
        # Need right_sum < 1 - ls
        target = 1 - ls
        # Find largest right_sum < target
        idx = bisect.bisect_left(right_values, target) - 1
        if idx >= 0:
            rs = right_values[idx]
            total = ls + rs
            if total > best_sum:
                best_sum = total
                best_left_mask = lmask
                best_right_mask = right_sums[idx][1]
    
    # Also check if just left or just right alone is better (already covered since empty subset is included)
    
    # Build the index list
    selected = []
    for i in range(len(left_items)):
        if best_left_mask & (1 << i):
            selected.append(left_items[i])
    for i in range(len(right_items)):
        if best_right_mask & (1 << i):
            selected.append(right_items[i])
    
    selected.sort()
    
    # Verify
    assert len(selected) > 0, "Must select at least one item"
    assert len(set(selected)) == len(selected), "Indices must be distinct"
    total_check = sum(weights[i] for i in selected)
    assert total_check < 1, f"Sum {total_check} not strictly below 1"
    
    return selected
# EVOLVE-BLOCK-END
