import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    # Weights: each weight = L // d, so sum of weights / L = reciprocal sum
    weights = [L // d for d in denominators]

    # Total sum of all weights (should be > L since reciprocal total > 1)
    total_weight = sum(weights)

    # Precompute suffix sums: suffix_sum[i] = sum of weights from index i to end
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]

    # We process denominators in descending order of weight (i.e., ascending order of denominator)
    # Index i corresponds to denominator 2+i
    # weights[i] = L // (2+i), so larger i -> smaller weight
    # We want to try larger weights first, so process i from n-1 down to 0
    # Actually, let's reorder: sort by weight descending
    # But we need to track original indices
    order = list(range(n))
    order.sort(key=lambda i: weights[i], reverse=True)

    # Reorder weights
    rw = [weights[i] for i in order]

    # Precompute suffix sums on reordered weights
    rsuffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        rsuffix[i] = rsuffix[i + 1] + rw[i]

    best_sum = 0
    best_mask = 0

    # Branch and bound DFS
    # We use iterative DFS with a stack to avoid recursion depth issues
    # State: (index, current_sum, current_mask)
    # Pruning: if current_sum + rsuffix[index] <= best_sum, skip

    def dfs(idx, curr_sum, curr_mask):
        nonlocal best_sum, best_mask

        if time.time() > deadline - 3:
            return

        if idx == n:
            if curr_sum < L and curr_sum > best_sum:
                best_sum = curr_sum
                best_mask = curr_mask
            return

        # Pruning: even if we take all remaining, can we beat best?
        if curr_sum + rsuffix[idx] <= best_sum:
            return

        # Try NOT taking item idx
        dfs(idx + 1, curr_sum, curr_mask)

        # Try taking item idx
        new_sum = curr_sum + rw[idx]
        if new_sum < L:
            dfs(idx + 1, new_sum, curr_mask | (1 << idx))

    # Call DFS
    dfs(0, 0, 0)

    # If best_sum is 0, we need at least one element
    if best_sum == 0:
        # Just pick the largest weight that's < L
        best_sum = max(w for w in rw if w < L) if any(w < L for w in rw) else 0
        if best_sum > 0:
            for i in range(n):
                if rw[i] == best_sum:
                    best_mask = 1 << i
                    break

    # Convert best_mask (in reordered space) back to original indices
    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(order[i])

    if not result_indices:
        return [0]

    # Local search to polish
    recip = [Fraction(1, d) for d in denominators]
    current_subset = set(result_indices)
    current_sum = sum(recip[i] for i in current_subset)
    best_deficit = Fraction(1, 1) - current_sum

    if best_deficit <= 0:
        return result_indices

    # 1-swap local search
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        in_sub = list(current_subset)
        not_in_sub = [i for i in range(n) if i not in current_subset]

        best_gain = Fraction(0)
        best_swap = None

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
            new_deficit = Fraction(1, 1) - current_sum
            if new_deficit < best_deficit:
                best_deficit = new_deficit
                result_indices = list(current_subset)

    # Add single
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            not_in_sub = [i for i in range(n) if i not in current_subset]
            best_gain = Fraction(0)
            best_add = None
            for j in not_in_sub:
                new_sum = current_sum + recip[j]
                if new_sum < 1:
                    gain = recip[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_add = j
            if best_add is not None:
                current_subset.add(best_add)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)

    # Remove single to enable better additions
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]

            best_gain = Fraction(0)
            best_move = None

            for i in in_sub:
                for j in not_in_sub:
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum < 1:
                        gain = recip[j] - recip[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (i, j)

            if best_move:
                i, j = best_move
                current_subset.remove(i)
                current_subset.add(j)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)

    # 2-swap: remove 1, add 2
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]

            best_gain = Fraction(0)
            best_move = None

            for i in in_sub:
                for j1 in range(len(not_in_sub)):
                    for j2 in range(j1 + 1, len(not_in_sub)):
                        a, b = not_in_sub[j1], not_in_sub[j2]
                        new_sum = current_sum - recip[i] + recip[a] + recip[b]
                        if new_sum < 1:
                            gain = recip[a] + recip[b] - recip[i]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (i, a, b)

            if best_move:
                i, a, b = best_move
                current_subset.remove(i)
                current_subset.add(a)
                current_subset.add(b)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)

    # 2-swap: remove 2, add 1
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]

            best_gain = Fraction(0)
            best_move = None

            for i1 in range(len(in_sub)):
                for i2 in range(i1 + 1, len(in_sub)):
                    a, b = in_sub[i1], in_sub[i2]
                    for j in not_in_sub:
                        new_sum = current_sum - recip[a] - recip[b] + recip[j]
                        if new_sum < 1:
                            gain = recip[j] - recip[a] - recip[b]
                            if gain > best_gain:
                                best_gain = gain
                                best_move = (a, b, j)

            if best_move:
                a, b, j = best_move
                current_subset.remove(a)
                current_subset.remove(b)
                current_subset.add(j)
                current_sum += best_gain
                improved = True
                new_deficit = Fraction(1, 1) - current_sum
                if new_deficit < best_deficit:
                    best_deficit = new_deficit
                    result_indices = list(current_subset)

    return result_indices
