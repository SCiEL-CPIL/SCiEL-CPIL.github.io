"""Maximize pairwise-noncoprime subset of [1,255] containing 255.

255 = 3 * 5 * 17. Multiples of 3 give a clique of size 85.
Use bitmask-based compatibility for fast greedy + local search to find larger cliques.
"""

import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Candidates: numbers in [1, 255] with gcd(x, 255) > 1
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}

    # Precompute compatibility bitmask for each candidate
    # compat_mask[i] has bit j set iff gcd(candidates[i], candidates[j]) > 1
    compat_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and math.gcd(candidates[i], candidates[j]) > 1:
                compat_mask[i] |= (1 << j)

    # Initial solution: all multiples of 3
    initial = set(range(3, 256, 3))

    # Build bitmask representation of the set
    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << idx[v])
        return m

    def mask_to_set(m):
        s = set()
        for i in range(n):
            if m & (1 << i):
                s.add(candidates[i])
        return s

    # Check if candidate i is compatible with all elements in mask m
    def can_add(i, m):
        # i must be compatible with every element in m
        # i is compatible with element j iff bit j is set in compat_mask[i]
        # We need: (m & ~compat_mask[i]) == 0, i.e., all bits in m are in compat_mask[i]
        # But also i shouldn't be in m already (handled by caller)
        return (m & ~compat_mask[i]) == 0

    best_set = set(initial)
    best_mask = set_to_mask(best_set)

    # Greedy: try adding compatible elements
    for attempt in range(100):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            if best_mask & (1 << i):
                continue
            if can_add(i, best_mask):
                best_mask |= (1 << i)
                improved = True
        if not improved:
            break

    best_set = mask_to_set(best_mask)
    best_size = len(best_set)

    # Local search: 1-for-2 swaps (remove one, add two)
    for it in range(500):
        if time.time() > deadline:
            break

        found_swap = False
        # Try each element in the set as removal candidate
        for remove_i in range(n):
            if time.time() > deadline:
                break
            if not (best_mask & (1 << remove_i)):
                continue

            # Temp mask without remove_i
            temp_mask = best_mask & ~(1 << remove_i)

            # Find all candidates compatible with temp_mask
            addable = []
            for i in range(n):
                if best_mask & (1 << i):
                    continue
                if can_add(i, temp_mask):
                    addable.append(i)

            # Try to find two addable elements compatible with each other
            for ai in range(len(addable)):
                if time.time() > deadline:
                    break
                for bi in range(ai + 1, len(addable)):
                    a, b = addable[ai], addable[bi]
                    # Check if a and b are compatible
                    if (compat_mask[a] >> b) & 1:
                        # Perform swap
                        best_mask = temp_mask | (1 << a) | (1 << b)
                        found_swap = True
                        best_size += 1
                        break
                if found_swap:
                    break
            if found_swap:
                break

    best_set = mask_to_set(best_mask)

    # Final greedy pass
    for i in range(n):
        if time.time() > deadline:
            break
        if best_mask & (1 << i):
            continue
        if can_add(i, best_mask):
            best_mask |= (1 << i)

    best_set = mask_to_set(best_mask)

    # Ensure 255 is in the set
    if 255 not in best_set:
        best_set.add(255)

    # Final validation
    lst = sorted(best_set)
    valid = True
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if math.gcd(lst[i], lst[j]) == 1:
                valid = False
                break
        if not valid:
            break

    if not valid:
        return list(initial)

    return list(best_set)
