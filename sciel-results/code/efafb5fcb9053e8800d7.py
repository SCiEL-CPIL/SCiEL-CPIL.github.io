"""Strong local search over integer knapsack formulation of unit-fraction deficit."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction


def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin

    denominators = list(range(2, 49))
    n = len(denominators)

    # LCM of 2..48
    def lcm(a, b):
        return a // gcd(a, b) * b

    L = 1
    for d in denominators:
        L = lcm(L, d)

    # weights: w[i] = L // d_i
    weights = [L // d for d in denominators]
    W_total = sum(weights)

    # We want max sum of selected weights subject to sum < L
    # deficit = (L - sum) / L

    # --- Initial solution: greedy by smallest denominator (largest unit fraction) ---
    # Sort indices by weight descending (denominator ascending)
    order = sorted(range(n), key=lambda i: -weights[i])
    selected = set()
    total_w = 0
    for i in order:
        if total_w + weights[i] < L:
            selected.add(i)
            total_w += weights[i]

    best_set = set(selected)
    best_sum = total_w
    best_deficit = L - best_sum

    # --- Local search with 4 move types ---
    # Move types:
    # 1. Add one element (if sum + w < L)
    # 2. Remove one element
    # 3. Swap: remove one, add one (1-1)
    # 4. Remove two, add one (2-1) or remove one, add two (1-2)

    def try_add(selected, total_w):
        """Try adding any single element not in selected."""
        best_gain = 0
        best_i = -1
        for i in range(n):
            if i not in selected:
                if total_w + weights[i] < L:
                    gain = weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
        return best_i, best_gain

    def try_remove(selected, total_w):
        """Try removing one element (always valid, frees up space)."""
        # This is only useful in combination with additions
        return None  # handled in swap moves

    def try_swap_11(selected, total_w):
        """Remove one, add one: net change = w_new - w_old, need total_w + w_new - w_old < L."""
        best_net = 0
        best_pair = None
        for i in selected:
            for j in range(n):
                if j not in selected:
                    net = weights[j] - weights[i]
                    if total_w + net < L and net > best_net:
                        best_net = net
                        best_pair = (i, j)
        return best_pair, best_net

    def try_swap_21(selected, total_w):
        """Remove two, add one: net = w_new - w_i - w_k, need total_w + net < L."""
        best_net = 0
        best_triple = None
        sel_list = list(selected)
        for a in range(len(sel_list)):
            i = sel_list[a]
            for b in range(a + 1, len(sel_list)):
                k = sel_list[b]
                for j in range(n):
                    if j not in selected:
                        net = weights[j] - weights[i] - weights[k]
                        if total_w + net < L and net > best_net:
                            best_net = net
                            best_triple = (i, k, j)
        return best_triple, best_net

    def try_swap_12(selected, total_w):
        """Remove one, add two: net = w_j + w_k - w_i, need total_w + net < L."""
        best_net = 0
        best_quad = None
        for i in selected:
            for j in range(n):
                if j not in selected:
                    for k in range(n):
                        if k not in selected and k != j:
                            net = weights[j] + weights[k] - weights[i]
                            if total_w + net < L and net > best_net:
                                best_net = net
                                best_quad = (i, j, k)
        return best_quad, best_net

    # --- Best-improvement local search loop ---
    max_iter = 200000
    iter_count = 0

    while best_deficit > 0 and iter_count < max_iter:
        if time.time() > deadline:
            break
        iter_count += 1

        improved = False

        # Try add
        i, gain = try_add(best_set, best_sum)
        if i >= 0:
            best_set.add(i)
            best_sum += gain
            best_deficit = L - best_sum
            improved = True

        # Try 1-1 swap
        pair, net = try_swap_11(best_set, best_sum)
        if pair is not None and net > 0:
            i, j = pair
            best_set.discard(i)
            best_set.add(j)
            best_sum += net
            best_deficit = L - best_sum
            improved = True

        # Try 2-1 swap
        triple, net = try_swap_21(best_set, best_sum)
        if triple is not None and net > 0:
            i, k, j = triple
            best_set.discard(i)
            best_set.discard(k)
            best_set.add(j)
            best_sum += net
            best_deficit = L - best_sum
            improved = True

        # Try 1-2 swap
        quad, net = try_swap_12(best_set, best_sum)
        if quad is not None and net > 0:
            i, j, k = quad
            best_set.discard(i)
            best_set.add(j)
            best_set.add(k)
            best_sum += net
            best_deficit = L - best_sum
            improved = True

        if not improved:
            # Randomized kick: remove 1-3 random elements, then re-add greedily
            if time.time() > deadline:
                break
            kick_size = random.randint(1, 3)
            sel_list = list(best_set)
            random.shuffle(sel_list)
            for i in sel_list[:kick_size]:
                best_set.discard(i)
                best_sum -= weights[i]
            best_deficit = L - best_sum

            # Re-add greedily
            order2 = sorted(range(n), key=lambda i: -weights[i])
            for i in order2:
                if i not in best_set and best_sum + weights[i] < L:
                    best_set.add(i)
                    best_sum += weights[i]
                    best_deficit = L - best_sum

    # --- Complement restart: try starting from full set and removing ---
    if time.time() < deadline - 5.0:
        # Start with all elements that fit when removed one-by-one from full
        # Actually, try a different heuristic: start with largest denominators (smallest fractions)
        # and build up
        order3 = sorted(range(n), key=lambda i: weights[i])  # smallest weights first
        sel2 = set()
        sum2 = 0
        for i in order3:
            if sum2 + weights[i] < L:
                sel2.add(i)
                sum2 += weights[i]
        deficit2 = L - sum2
        if deficit2 < best_deficit:
            best_set = sel2
            best_sum = sum2
            best_deficit = deficit2

            # Run local search on this new starting point
            for _ in range(50000):
                if time.time() > deadline:
                    break
                improved2 = False
                i, gain = try_add(best_set, best_sum)
                if i >= 0:
                    best_set.add(i)
                    best_sum += gain
                    best_deficit = L - best_sum
                    improved2 = True
                pair, net = try_swap_11(best_set, best_sum)
                if pair is not None and net > 0:
                    a, b = pair
                    best_set.discard(a)
                    best_set.add(b)
                    best_sum += net
                    best_deficit = L - best_sum
                    improved2 = True
                triple, net = try_swap_21(best_set, best_sum)
                if triple is not None and net > 0:
                    a, b, c = triple
                    best_set.discard(a)
                    best_set.discard(b)
                    best_set.add(c)
                    best_sum += net
                    best_deficit = L - best_sum
                    improved2 = True
                if not improved2:
                    break

    # Final sanity check
    final_sum = sum(weights[i] for i in best_set)
    assert final_sum < L, "Sum must be strictly below L"
    assert len(best_set) > 0, "Must be nonempty"

    # Convert back to zero-based indices into denominators [2,3,...,48]
    # index i corresponds to denominator (i+2)
    result = sorted(list(best_set))
    return result
# EVOLVE-BLOCK-END
