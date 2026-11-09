"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    denominators = list(range(2, 49))
    n = len(denominators)

    # Compute LCM of 2..48 for exact integer arithmetic
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)

    # Precompute numerator for each denominator: 1/d = (lcm_val/d) / lcm_val
    nums = [lcm_val // d for d in denominators]

    # Sort indices by weight descending (largest unit fractions first)
    sorted_indices = sorted(range(n), key=lambda i: -nums[i])

    def greedy_init(order):
        """Greedy: add elements in given order that keep sum < lcm_val."""
        selected = []
        total = 0
        for i in order:
            if total + nums[i] < lcm_val:
                selected.append(i)
                total += nums[i]
        return selected, total

    def local_search(selected, total):
        """Improve via 1-add, 1-swap, and 1-for-2-swap moves."""
        in_set = [False] * n
        for i in selected:
            in_set[i] = True

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False

            # 1-add: try adding an unselected element
            for j in range(n):
                if not in_set[j]:
                    if total + nums[j] < lcm_val:
                        selected.append(j)
                        in_set[j] = True
                        total += nums[j]
                        improved = True
                        break

            if improved:
                continue

            # 1-swap: remove i, add j where gain > 0 and new total < lcm_val
            best_gain = 0
            best_move = None
            for i in selected:
                vi = nums[i]
                for j in range(n):
                    if not in_set[j]:
                        vj = nums[j]
                        gain = vj - vi
                        if gain > 0:
                            new_total = total - vi + vj
                            if new_total < lcm_val:
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = (i, j)
            if best_move:
                i, j = best_move
                in_set[i] = False
                in_set[j] = True
                selected.remove(i)
                selected.append(j)
                total += nums[j] - nums[i]
                improved = True
                continue

            # 1-for-2-swap: remove i, add j and k
            best_gain = 0
            best_move = None
            for i in selected:
                vi = nums[i]
                base = total - vi
                for j in range(n):
                    if not in_set[j]:
                        vj = nums[j]
                        if base + vj >= lcm_val:
                            continue
                        for k in range(j + 1, n):
                            if not in_set[k]:
                                vk = nums[k]
                                new_total = base + vj + vk
                                if new_total < lcm_val:
                                    gain = vj + vk - vi
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = (i, j, k)
            if best_move:
                i, j, k = best_move
                in_set[i] = False
                in_set[j] = True
                in_set[k] = True
                selected.remove(i)
                selected.append(j)
                selected.append(k)
                total += nums[j] + nums[k] - nums[i]
                improved = True

        return selected, total

    # Best result tracking
    best_indices = None
    best_sum = 0

    # First try greedy with sorted order
    sel, tot = greedy_init(sorted_indices)
    sel, tot = local_search(sel, tot)
    best_indices = sel
    best_sum = tot

    # Random restarts with shuffled orderings
    random.seed(42)
    while time.time() < deadline - 2.0:
        perm = list(range(n))
        random.shuffle(perm)

        selected, total = greedy_init(perm)
        selected, total = local_search(selected, total)

        if total > best_sum:
            best_indices = selected
            best_sum = total

    # Fallback: if best_indices is empty, use a single element
    if not best_indices:
        best_indices = [0]  # index 0 corresponds to denominator 2

    return best_indices
# EVOLVE-BLOCK-END
