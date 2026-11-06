import time
from math import lcm
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    weights = [L // d for d in denominators]

    # Split into two halves for meet-in-the-middle
    mid = n // 2
    indices_1 = list(range(mid))
    indices_2 = list(range(mid, n))

    weights_1 = [weights[i] for i in indices_1]
    weights_2 = [weights[i] for i in indices_2]

    def gen_sums(weights_list):
        # Returns list of (sum, mask) where sum < L
        sums = [(0, 0)]
        for i, w in enumerate(weights_list):
            bit = 1 << i
            # Filter and add
            new_sums = [(s + w, m | bit) for s, m in sums if s + w < L]
            sums.extend(new_sums)
            # Check time periodically
            if time.time() > deadline:
                break
        return sums

    # Generate sums for both halves
    sums_1 = gen_sums(weights_1)
    sums_2 = gen_sums(weights_2)

    # Sort sums_2 by sum value for binary search
    sums_2.sort(key=lambda x: x[0])
    sums_2_values = [s[0] for s in sums_2]

    best_sum = -1
    best_mask = 0

    for s1, m1 in sums_1:
        if time.time() > deadline:
            break
        # We need s2 such that s1 + s2 < L => s2 < L - s1
        target = L - s1 - 1
        # Find rightmost index in sums_2_values <= target
        idx = bisect.bisect_right(sums_2_values, target)
        if idx > 0:
            s2, m2 = sums_2[idx - 1]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask = m1 | (m2 << mid)

    # Convert best_mask to indices
    if best_mask == 0:
        return [0]

    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)

    # Now apply local search (1-swap and 2-swap) to improve the solution
    # Using Fraction for exact arithmetic
    recip = [Fraction(1, d) for d in denominators]
    current_subset = set(result_indices)
    current_sum = sum(recip[i] for i in current_subset)

    def update_best(indices, total):
        nonlocal best_sum, best_mask, result_indices
        if not indices:
            return
        deficit = 1 - total
        if deficit > 0 and deficit < best_deficit:
            best_deficit = deficit
            result_indices = list(indices)

    best_deficit = Fraction(1, 1) - current_sum
    if best_deficit <= 0:
        return result_indices

    # Local search with 1-swap moves
    improved = True
    while improved and time.time() < deadline:
        improved = False

        in_sub = list(current_subset)
        not_in_sub = [i for i in range(n) if i not in current_subset]

        best_swap = None
        best_gain = Fraction(0)

        for i in in_sub:
            for j in not_in_sub:
                new_sum = current_sum - recip[i] + recip[j]
                if new_sum < 1:
                    gain = recip[j] - recip[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j)

        if best_swap:
            i, j = best_swap
            current_subset.remove(i)
            current_subset.add(j)
            current_sum += best_gain
            improved = True
            new_deficit = 1 - current_sum
            if new_deficit < best_deficit:
                best_deficit = new_deficit
                result_indices = list(current_subset)

    return result_indices
