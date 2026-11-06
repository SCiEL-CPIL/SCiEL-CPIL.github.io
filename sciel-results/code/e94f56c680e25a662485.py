"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 85
    
    n = 5040
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    target = 4979
    
    usable = [d for d in divisors if d <= target]
    usable_desc = sorted(usable, reverse=True)
    
    # Try k = 1, 2, 3, 4, 5 with brute force
    for k in range(1, 6):
        if time.time() > deadline - 5:
            break
        for combo in combinations(usable_desc, k):
            s = sum(combo)
            if s == target:
                return list(combo)
            # Pruning: if even the largest k divisors can't reach target, skip
            # But we're iterating all combos, so just let it run
        if time.time() > deadline - 5:
            break
    
    # For k >= 6, use meet-in-the-middle
    # Split usable into two halves
    m = len(usable_desc)
    half = m // 2
    left = usable_desc[:half]
    right = usable_desc[half:]
    
    # Precompute all subset sums for left half, tracking cardinality
    # For left: all subsets, store (sum -> min_cardinality, subset)
    left_sums = {}
    for mask in range(1 << len(left)):
        s = 0
        cnt = 0
        subset = []
        for i in range(len(left)):
            if mask & (1 << i):
                s += left[i]
                cnt += 1
                subset.append(left[i])
        if s <= target:
            if s not in left_sums or cnt < left_sums[s][0]:
                left_sums[s] = (cnt, subset)
    
    # For each subset of right, check if target - right_sum is in left_sums
    best = None
    for mask in range(1 << len(right)):
        if time.time() > deadline - 2:
            break
        s = 0
        cnt = 0
        subset = []
        for i in range(len(right)):
            if mask & (1 << i):
                s += right[i]
                cnt += 1
                subset.append(right[i])
        if s > target:
            continue
        need = target - s
        if need in left_sums:
            l_cnt, l_subset = left_sums[need]
            total_cnt = cnt + l_cnt
            if total_cnt >= 6:  # We only care about k >= 6 here
                if best is None or total_cnt < best[0]:
                    best = (total_cnt, l_subset + subset)
    
    if best is not None:
        return best[1]
    
    # Fallback greedy
    remainder = target
    chosen = []
    for d in usable_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
    assert remainder == 0
    return chosen
# EVOLVE-BLOCK-END
