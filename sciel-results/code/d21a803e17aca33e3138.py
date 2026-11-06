import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin for 90s limit

    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)

    def lcm(a, b):
        return a * b // gcd(a, b)

    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))

    # Precompute integer weights
    # index i corresponds to denominator i+2
    weights = [L // d for d in denominators]

    # We want to find a subset S such that sum_{i in S} weights[i] < L
    # and sum_{i in S} weights[i] is as close to L as possible (maximize the sum)
    # The deficit is L - sum, and we want to minimize it.

    # Strategy: Meet-in-the-Middle
    # Split into two halves, enumerate all subset sums for each half,
    # then for each left sum, find the best right sum such that total < L.

    mid = n // 2  # 23
    left_weights = weights[:mid]
    right_weights = weights[mid:]

    def gen_subsets(w_list):
        """Generate all subset sums (sum, mask) pairs."""
        sums = [0]
        masks = [0]
        for i, w in enumerate(w_list):
            bit = 1 << i
            current_len = len(sums)
            new_sums = []
            new_masks = []
            for k in range(current_len):
                ns = sums[k] + w
                # We only care about sums < L (since we need total < L)
                if ns < L:
                    new_sums.append(ns)
                    new_masks.append(masks[k] | bit)
            sums.extend(new_sums)
            masks.extend(new_masks)
        return list(zip(sums, masks))

    # Generate left subsets
    left_subsets = gen_subsets(left_weights)
    left_subsets.sort(key=lambda x: x[0])
    left_sums = [x[0] for x in left_subsets]
    left_masks = [x[1] for x in left_subsets]

    if time.time() - start_time > timeout:
        # Fallback
        return [0]

    # Generate right subsets
    right_subsets = gen_subsets(right_weights)
    right_subsets.sort(key=lambda x: x[0])
    right_sums = [x[0] for x in right_subsets]
    right_masks = [x[1] for x in right_subsets]

    if time.time() - start_time > timeout:
        return [0]

    # For each left sum, find the best right sum such that left + right < L
    # Use binary search on right_sums
    import bisect

    best_sum = 0
    best_l_mask = 0
    best_r_mask = 0

    for i in range(len(left_subsets)):
        s_l = left_sums[i]
        # We need s_r < L - s_l
        # Find the largest s_r in right_sums such that s_r < L - s_l
        limit = L - s_l
        # bisect_right gives the insertion point; we want the element before it
        idx = bisect.bisect_left(right_sums, limit)
        if idx > 0:
            s_r = right_sums[idx - 1]
            total = s_l + s_r
            if total < L and total > best_sum:
                best_sum = total
                best_l_mask = left_masks[i]
                best_r_mask = right_masks[idx - 1]

    if best_sum > 0:
        indices = []
        for k in range(mid):
            if best_l_mask & (1 << k):
                indices.append(k)
        for k in range(len(right_weights)):
            if best_r_mask & (1 << k):
                indices.append(mid + k)

        # Local search refinement with multiple move types
        indices = local_search_refine(indices, weights, L, start_time, timeout)
        return indices

    # Fallback: greedy + local search
    order = sorted(range(n), key=lambda i: -weights[i])
    selected = []
    current_sum = 0
    for idx in order:
        if current_sum + weights[idx] < L:
            selected.append(idx)
            current_sum += weights[idx]

    if not selected:
        return [0]

    selected = local_search_refine(selected, weights, L, start_time, timeout)
    return selected


def local_search_refine(indices, weights, L, start_time, timeout):
    """Improve solution via multiple move types: add, remove, 1-1 swap, 1-2 swap, 2-1 swap."""
    if not indices:
        return indices

    selected = set(indices)
    current_sum = sum(weights[i] for i in selected)
    best_sum = current_sum
    best_indices = list(indices)

    all_items = set(range(len(weights)))
    unselected = all_items - selected

    def try_improve(selected, unselected, current_sum, best_sum, best_indices):
        improved = False

        # Move type 1: Add one item
        best_add = None
        best_add_sum = current_sum
        for add in unselected:
            candidate_sum = current_sum + weights[add]
            if candidate_sum < L and candidate_sum > best_add_sum:
                best_add_sum = candidate_sum
                best_add = add
        if best_add is not None:
            selected.add(best_add)
            unselected.remove(best_add)
            current_sum = best_add_sum
            best_sum = current_sum
            best_indices = list(selected)
            improved = True

        # Move type 2: Remove one item
        best_rem = None
        best_rem_sum = current_sum
        for rem in selected:
            candidate_sum = current_sum - weights[rem]
            if candidate_sum > best_rem_sum:
                best_rem_sum = candidate_sum
                best_rem = rem
        if best_rem is not None and best_rem_sum > best_sum:
            selected.remove(best_rem)
            unselected.add(best_rem)
            current_sum = best_rem_sum
            best_sum = current_sum
            best_indices = list(selected)
            improved = True

        # Move type 3: 1-1 swap (remove one, add one)
        for rem in list(selected):
            new_sum_without = current_sum - weights[rem]
            best_add = None
            best_add_sum = new_sum_without
            for add in unselected:
                candidate_sum = new_sum_without + weights[add]
                if candidate_sum < L and candidate_sum > best_add_sum:
                    best_add_sum = candidate_sum
                    best_add = add
            if best_add is not None and best_add_sum > best_sum:
                selected.remove(rem)
                selected.add(best_add)
                unselected.remove(best_add)
                unselected.add(rem)
                current_sum = best_add_sum
                best_sum = current_sum
                best_indices = list(selected)
                improved = True
                break

        # Move type 4: 1-2 swap (remove one, add two)
        for rem in list(selected):
            new_sum_without = current_sum - weights[rem]
            best_add1 = None
            best_add2 = None
            best_add_sum = new_sum_without
            for add1 in unselected:
                for add2 in unselected:
                    if add1 == add2:
                        continue
                    candidate_sum = new_sum_without + weights[add1] + weights[add2]
                    if candidate_sum < L and candidate_sum > best_add_sum:
                        best_add_sum = candidate_sum
                        best_add1 = add1
                        best_add2 = add2
            if best_add1 is not None and best_add_sum > best_sum:
                selected.remove(rem)
                selected.add(best_add1)
                selected.add(best_add2)
                unselected.remove(best_add1)
                unselected.remove(best_add2)
                unselected.add(rem)
                current_sum = best_add_sum
                best_sum = current_sum
                best_indices = list(selected)
                improved = True
                break

        # Move type 5: 2-1 swap (remove two, add one)
        sel_list = list(selected)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                rem1 = sel_list[i]
                rem2 = sel_list[j]
                new_sum_without = current_sum - weights[rem1] - weights[rem2]
                best_add = None
                best_add_sum = new_sum_without
                for add in unselected:
                    candidate_sum = new_sum_without + weights[add]
                    if candidate_sum < L and candidate_sum > best_add_sum:
                        best_add_sum = candidate_sum
                        best_add = add
                if best_add is not None and best_add_sum > best_sum:
                    selected.remove(rem1)
                    selected.remove(rem2)
                    selected.add(best_add)
                    unselected.remove(best_add)
                    unselected.add(rem1)
                    unselected.add(rem2)
                    current_sum = best_add_sum
                    best_sum = current_sum
                    best_indices = list(selected)
                    improved = True
                    break
            if improved:
                break

        return improved, best_sum, best_indices

    # Iterative improvement
    for iteration in range(200):
        if time.time() - start_time > timeout:
            break

        improved, best_sum, best_indices = try_improve(selected, unselected, current_sum, best_sum, best_indices)

        if not improved:
            # Try randomized kicks
            if len(unselected) > 0 and len(selected) > 2:
                # Randomly remove 2-3 items and re-optimize
                kick_size = min(3, len(selected) - 1)
                kick_items = random.sample(list(selected), kick_size)
                for item in kick_items:
                    selected.remove(item)
                    unselected.add(item)
                current_sum = sum(weights[i] for i in selected)

                # Re-optimize
                for _ in range(50):
                    if time.time() - start_time > timeout:
                        break
                    improved2, best_sum, best_indices = try_improve(selected, unselected, current_sum, best_sum, best_indices)
                    if not improved2:
                        break

    return best_indices
