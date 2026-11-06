"""Rewritten program for large reciprocal-subsum-free sets using conflict-graph + bitset branch-and-bound."""
from fractions import Fraction
import time
import math

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 82  # safety margin before 90s wall-clock limit

    N = 24
    # Precompute LCM and integer representations of reciprocals
    def _lcm(a, b):
        return a * b // math.gcd(a, b)

    L = 1
    for i in range(1, N + 1):
        L = _lcm(L, i)
    val = {i: L // i for i in range(1, N + 1)}

    # For each element i, compute the set of elements j > i such that
    # 1/i can be expressed as a sum of reciprocals of a subset of {j, j+1, ..., 24}.
    # We store this as a bitmask over indices 1..24.
    # conflict[i] is a bitmask where bit j is set if j is "dangerous" for i.
    # An element i is subsumed by a set S iff (conflict[i] & mask(S \ {i})) != 0
    # is NOT quite right -- we need subset-sum, not just presence.
    #
    # Actually the condition is: 1/i = sum of 1/j for j in T, where T is a non-empty
    # subset of S \ {i}, T contains only elements > i (since 1/i > 1/j for j>i, and
    # we need the sum to equal 1/i, so all elements in T must be > i... wait, no.
    # If j < i, then 1/j > 1/i, so a single j < i already exceeds 1/i. So any subset
    # T with an element j < i would have sum >= 1/j > 1/i. Thus T must contain only
    # elements j > i.
    #
    # So: i is subsumed by S iff there exists a non-empty subset T of {j in S : j > i}
    # such that sum(1/j for j in T) = 1/i.
    #
    # In integer terms: val[i] = sum(val[j] for j in T), where T ⊆ {j in S : j > i}.
    #
    # For branch-and-bound, I need to quickly check if adding x to a current set C
    # creates a violation. The violation would be: some y in C ∪ {x} is subsumed by
    # the rest. Since we build incrementally, when we add x, we need to check:
    #   1. x is not subsumed by C: i.e., no subset of {j in C : j > x} sums to val[x]
    #   2. No y in C becomes newly subsumed: adding x could only help form a sum for
    #      y if x > y. So for each y in C with y < x, check if val[y] can be formed
    #      from {j in C : j > y} ∪ {x} (but not using y itself).
    #
    # For speed, I'll precompute for each target t and each "upper set" (elements > t),
    # whether t is a subset-sum. But that's 24 * 2^24 which is too much.
    #
    # Instead, I'll use a fast subset-sum check with bitsets.

    # Precompute: for each i in 1..24, the list of values val[j] for j > i
    # and use a DP-based subset sum check.

    def subset_sum_exists(target, candidates):
        """Check if target is a subset sum of the multiset candidates (all positive ints)."""
        if not candidates:
            return False
        if target in candidates:
            return True
        # Use a set of achievable sums, pruned
        achievable = {0}
        for c in sorted(candidates):
            if c > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + c
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return True
        return False

    # For the branch-and-bound, I'll represent the current set as a bitmask (24 bits).
    # Element i corresponds to bit (i-1).

    def mask_to_set(m):
        return {i+1 for i in range(N) if m & (1 << i)}

    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x-1))
        return m

    # Precompute val as a list indexed by i-1
    vals = [val[i] for i in range(1, N+1)]

    # Precompute for quick checks: for each i, the bitmask of elements j > i
    gt_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        m = 0
        for j in range(i + 1, N + 1):
            m |= (1 << (j - 1))
        gt_mask[i] = m

    # Check if a set (as bitmask) is subsum-free
    def is_subsum_free(m):
        s = mask_to_set(m)
        for x in s:
            others = [y for y in s if y > x]
            if not others:
                continue
            if subset_sum_exists(vals[x-1], [vals[y-1] for y in others]):
                return False
        return True

    # Check if adding element x to current set (mask m) keeps it subsum-free
    def can_add(m, x):
        # Check x is not subsumed by current set
        others = [y for y in mask_to_set(m) if y > x]
        if others and subset_sum_exists(vals[x-1], [vals[y-1] for y in others]):
            return False
        # Check no existing y < x becomes subsumed by adding x
        s = mask_to_set(m)
        for y in s:
            if y >= x:
                continue
            # y could now be subsumed using x
            others_y = [z for z in s if z > y and z != x]
            if subset_sum_exists(vals[y-1], [vals[z-1] for z in others_y] + [vals[x-1]]):
                return False
        return True

    # Start with a good initial solution
    # Known good: {7, 8, 9, ..., 24} has size 18. Let's verify and try to improve.
    initial = set(range(7, 25))
    if not is_subsum_free(set_to_mask(initial)):
        # Fallback: try removing problematic elements
        initial = set(range(1, 25))
        for x in sorted(initial):
            m = set_to_mask(initial)
            if not is_subsum_free(m):
                # Try removing x
                initial.remove(x)
        # Re-verify
        while not is_subsum_free(set_to_mask(initial)):
            # Remove the smallest problematic element
            s = sorted(initial)
            removed = False
            for x in s:
                test = initial - {x}
                if is_subsum_free(set_to_mask(test)):
                    initial = test
                    removed = True
                    break
            if not removed:
                initial = {24}
                break

    best_mask = set_to_mask(initial)
    best_size = len(initial)

    # Branch and bound
    def bnb(current_mask, available_mask, current_size):
        nonlocal best_mask, best_size, deadline

        if current_size >= best_size:
            return
        if time.time() >= deadline:
            return

        # Upper bound: current_size + popcount(available_mask)
        if current_size + bin(available_mask).count('1') <= best_size:
            return

        # Try adding elements in descending order (larger elements are "safer")
        avail_list = sorted([i+1 for i in range(N) if available_mask & (1 << i)], reverse=True)

        for x in avail_list:
            if time.time() >= deadline:
                return
            if current_size + 1 + bin(available_mask & ~(1 << (x-1))).count('1') <= best_size:
                continue

            if can_add(current_mask, x):
                new_mask = current_mask | (1 << (x-1))
                new_size = current_size + 1
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                bnb(new_mask, available_mask & ~(1 << (x-1)), new_size)

    # Run BnB from the initial solution
    available = set_to_mask(set(range(1, 25)) - initial)
    bnb(best_mask, available, best_size)

    # Also try a few different starting points
    alt_starts = [
        set(range(8, 25)),   # {8..24}, size 17
        set(range(6, 25)),   # {6..24}, size 19 (may not be valid)
        set(range(1, 25)),   # all, then prune
    ]

    for start in alt_starts:
        if time.time() >= deadline:
            break
        # Prune start to make it valid
        s = set(start)
        while s and not is_subsum_free(set_to_mask(s)):
            # Remove the smallest element that causes a problem
            removed = False
            for x in sorted(s):
                test = s - {x}
                if is_subsum_free(set_to_mask(test)):
                    s = test
                    removed = True
                    break
            if not removed:
                break

        if len(s) >= best_size:
            continue

        avail = set_to_mask(set(range(1, 25)) - s)
        bnb(set_to_mask(s), avail, len(s))

    # Local improvement: try adding any remaining element
    remaining = set(range(1, 25)) - mask_to_set(best_mask)
    for x in sorted(remaining, reverse=True):
        if time.time() >= deadline:
            break
        if can_add(best_mask, x):
            best_mask |= (1 << (x-1))
            best_size += 1
            remaining.discard(x)

    # Final verification
    result = mask_to_set(best_mask)
    if not result:
        result = {24}

    return list(result)
# EVOLVE-BLOCK-END
