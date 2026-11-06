"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from fractions import Fraction
    from itertools import combinations
    
    deadline = time.time() + 50  # safety margin before 60s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves for meet-in-the-middle
    mid = n // 2
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    # Enumerate all subset sums for left half
    left_sums = []  # (sum_fraction, indices_tuple)
    for r in range(len(left_denoms) + 1):
        for combo in combinations(range(len(left_denoms)), r):
            s = Fraction()
            for idx in combo:
                s += Fraction(1, left_denoms[idx])
            if s < 1:
                left_sums.append((s, combo))
    
    # Enumerate all subset sums for right half
    right_sums = []  # (sum_fraction, indices_tuple)
    for r in range(len(right_denoms) + 1):
        for combo in combinations(range(len(right_denoms)), r):
            s = Fraction()
            for idx in combo:
                s += Fraction(1, right_denoms[idx])
            if s < 1:
                right_sums.append((s, combo))
    
    # Sort right sums by value for binary search
    right_sums.sort(key=lambda x: x[0])
    right_values = [rs[0] for rs in right_sums]
    
    best_deficit = None
    best_left_combo = None
    best_right_combo = None
    
    # For each left sum, find the right sum that maximizes total while keeping < 1
    # We want to maximize left_sum + right_sum subject to left_sum + right_sum < 1
    # So for each left_sum, find the largest right_sum such that right_sum < 1 - left_sum
    
    import bisect
    
    for left_sum, left_combo in left_sums:
        target = 1 - left_sum
        # Find the largest right_sum < target
        # bisect_left gives first position where value >= target
        pos = bisect.bisect_left(right_values, target)
        if pos > 0:
            # pos-1 is the largest right_sum < target
            right_sum = right_values[pos - 1]
            right_combo = right_sums[pos - 1][1]
            total = left_sum + right_sum
            deficit = 1 - total
            if best_deficit is None or deficit < best_deficit:
                best_deficit = deficit
                best_left_combo = left_combo
                best_right_combo = right_combo
    
    # Also check: maybe empty left is best with some right
    # Already covered since left_sums includes empty combo
    
    if best_deficit is None:
        # Fallback: greedy
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected
    
    # Convert to zero-based indices into denominators [2,3,...,48]
    # left_combo contains indices into left_denoms (which are denominators[0:mid])
    # right_combo contains indices into right_denoms (which are denominators[mid:])
    selected = []
    for idx in best_left_combo:
        selected.append(idx)  # already 0-based into denominators
    for idx in best_right_combo:
        selected.append(mid + idx)  # offset by mid
    
    selected.sort()
    return selected

# EVOLVE-BLOCK-END
