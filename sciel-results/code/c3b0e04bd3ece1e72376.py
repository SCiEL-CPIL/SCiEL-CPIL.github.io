import time
from fractions import Fraction
from math import gcd
import bisect

def construct():
    start = time.time()
    deadline = start + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)  # 47

    # Compute LCM(2, 3, ..., 48)
    L = 1
    for d in denominators:
        L = L // gcd(L, d) * d

    # Integer weights: w_i = L / d_i
    w = [L // d for d in denominators]

    # Split into two halves
    mid = n // 2  # 23
    n1 = mid
    n2 = n - mid  # 24

    # Sort each half in descending weight order for aggressive pruning
    left_w = w[:n1]
    right_w = w[n1:]

    # Create index mappings: sorted_pos -> original_pos within each half
    left_order = sorted(range(n1), key=lambda i: -left_w[i])
    right_order = sorted(range(n2), key=lambda i: -right_w[i])

    left_sorted_w = [left_w[i] for i in left_order]
    right_sorted_w = [right_w[i] for i in right_order]

    def gen_valid_subsets(weights, limit):
        """Generate all subsets with sum < limit. Returns list of (sum, mask)."""
        results = [(0, 0)]  # empty set
        for i, wt in enumerate(weights):
            bit = 1 << i
            new_entries = []
            for s, m in results:
                ns = s + wt
                if ns < limit:  # strictly below 1
                    new_entries.append((ns, m | bit))
            results.extend(new_entries)
        return results

    # Generate for left half (sorted descending for pruning)
    left_subsets = gen_valid_subsets(left_sorted_w, L)

    if time.time() > deadline - 10:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Generate for right half (sorted descending for pruning)
    right_subsets = gen_valid_subsets(right_sorted_w, L)

    if time.time() > deadline - 5:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Sort right subsets by sum for binary search
    right_sorted = sorted(right_subsets, key=lambda x: x[0])
    right_sums = [x[0] for x in right_sorted]
    right_masks = [x[1] for x in right_sorted]

    # For each left subset, find the best right subset such that total < L
    best_total = -1
    best_left_mask = 0
    best_right_mask = 0

    for ls, lm in left_subsets:
        if time.time() > deadline - 3:
            break

        bound = L - ls
        idx = bisect.bisect_left(right_sums, bound) - 1
        if idx < 0:
            continue

        rs = right_sums[idx]
        rm = right_masks[idx]
        total = ls + rs

        if total > best_total and (lm | rm) != 0:
            best_total = total
            best_left_mask = lm
            best_right_mask = rm

    # Also consider single-half solutions
    for ls, lm in left_subsets:
        if lm != 0 and ls > best_total:
            best_total = ls
            best_left_mask = lm
            best_right_mask = 0

    for rs, rm in right_subsets:
        if rm != 0 and rs > best_total:
            best_total = rs
            best_left_mask = 0
            best_right_mask = rm

    # Reconstruct selected indices using the permutation mappings
    selected = []
    for i in range(n1):
        if best_left_mask & (1 << i):
            # i is position in sorted order, map back to original
            orig_pos = left_order[i]
            selected.append(orig_pos)
    for i in range(n2):
        if best_right_mask & (1 << i):
            orig_pos = right_order[i]
            selected.append(n1 + orig_pos)

    if not selected:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    # Exact verification with Fraction
    total_frac = Fraction(0)
    for idx in selected:
        total_frac += Fraction(1, denominators[idx])

    if total_frac >= 1:
        sel = []
        s = Fraction(0)
        for k in range(n):
            if s + Fraction(1, denominators[k]) < 1:
                sel.append(k)
                s += Fraction(1, denominators[k])
        return sel

    return selected
