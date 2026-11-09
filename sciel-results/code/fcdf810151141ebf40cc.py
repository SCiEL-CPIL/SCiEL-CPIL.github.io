"""Rewritten program: integer LCM-scaled subset search with local improvement."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    # Deadline guard: finish well before 90s wall-clock kill
    deadline = time.time() + 75.0  # 15s safety margin

    # Denominators are 2..48 (47 items)
    denoms = list(range(2, 49))
    n = len(denoms)  # 47

    # LCM of 1..48
    def lcm(a, b):
        return a // math.gcd(a, b) * b

    L = 1
    for d in denoms:
        L = lcm(L, d)

    # Precompute scaled reciprocals: val[i] = L // denoms[i]
    # Sum of all vals > L (since sum of 1/d for d=2..48 > 1)
    vals = [L // d for d in denoms]
    total_all = sum(vals)
    assert total_all > L, "sum of reciprocals must exceed 1"

    # We want subset S such that sum(vals[i] for i in S) < L, minimizing L - sum
    # Equivalently, maximize sum(vals[i] for i in S) subject to sum < L

    # Greedy initial solution: sort by value descending, add if still < L
    def greedy_init():
        # Try multiple orderings
        best_sum = 0
        best_set = set()
        # Ordering 1: descending value
        order = sorted(range(n), key=lambda i: -vals[i])
        s = 0
        sel = set()
        for i in order:
            if s + vals[i] < L:
                s += vals[i]
                sel.add(i)
        if s > best_sum:
            best_sum = s
            best_set = sel.copy()

        # Ordering 2: ascending value (small denominators first)
        order2 = sorted(range(n), key=lambda i: vals[i])
        s = 0
        sel = set()
        for i in order2:
            if s + vals[i] < L:
                s += vals[i]
                sel.add(i)
        if s > best_sum:
            best_sum = s
            best_set = sel.copy()

        # Ordering 3: random
        for _ in range(50):
            order3 = list(range(n))
            random.shuffle(order3)
            s = 0
            sel = set()
            for i in order3:
                if s + vals[i] < L:
                    s += vals[i]
                    sel.add(i)
            if s > best_sum:
                best_sum = s
                best_set = sel.copy()

        return best_set, best_sum

    best_set, best_sum = greedy_init()

    # Local search: 1-add, 1-remove, 1-swap, 1-for-2-swap
    def local_search(sel, s):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return sel, s

            # 1-add: try adding an unselected item
            for i in range(n):
                if i not in sel and s + vals[i] < L:
                    new_s = s + vals[i]
                    if new_s > s:
                        sel.add(i)
                        s = new_s
                        improved = True
                        break

            if improved:
                continue

            # 1-swap: remove one, add another
            for i in list(sel):
                for j in range(n):
                    if j not in sel:
                        new_s = s - vals[i] + vals[j]
                        if new_s < L and new_s > s:
                            sel.discard(i)
                            sel.add(j)
                            s = new_s
                            improved = True
                            break
                if improved:
                    break

            if improved:
                continue

            # 1-for-2-swap: remove one, add two
            for i in list(sel):
                base = s - vals[i]
                for j in range(n):
                    if j not in sel and j != i:
                        s1 = base + vals[j]
                        if s1 >= L:
                            continue
                        for k in range(n):
                            if k not in sel and k != i and k != j:
                                s2 = s1 + vals[k]
                                if s2 < L and s2 > s:
                                    sel.discard(i)
                                    sel.add(j)
                                    sel.add(k)
                                    s = s2
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break

            if improved:
                continue

            # 2-for-1-swap: remove two, add one
            sel_list = list(sel)
            for a_idx in range(len(sel_list)):
                for b_idx in range(a_idx + 1, len(sel_list)):
                    i, j = sel_list[a_idx], sel_list[b_idx]
                    base = s - vals[i] - vals[j]
                    for k in range(n):
                        if k not in sel:
                            new_s = base + vals[k]
                            if new_s < L and new_s > s:
                                sel.discard(i)
                                sel.discard(j)
                                sel.add(k)
                                s = new_s
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

        return sel, s

    best_set, best_sum = local_search(best_set, best_sum)

    # Random restarts with local search
    while time.time() < deadline:
        # Random subset initialization
        # Start from a random subset and try to improve
        sel = set()
        s = 0
        # Randomly add items
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            if s + vals[i] < L:
                sel.add(i)
                s += vals[i]

        # Also try: start with all, remove items to get below L
        sel2 = set(range(n))
        s2 = total_all
        order2 = list(range(n))
        random.shuffle(order2)
        for i in order2:
            if s2 - vals[i] < L:
                # Can't remove this one without going below... actually we want to remove to get below L
                # We want to remove items so that sum < L
                # Let's try removing items
                pass

        # Better: start with all, remove items until sum < L, then try to add back
        sel3 = set(range(n))
        s3 = total_all
        order3 = list(range(n))
        random.shuffle(order3)
        for i in order3:
            if s3 - vals[i] < L:
                sel3.discard(i)
                s3 -= vals[i]
                break  # Now s3 < L

        # Try to improve sel3
        if len(sel3) > 0:
            sel3, s3 = local_search(sel3, s3)
            if s3 > best_sum:
                best_sum = s3
                best_set = sel3.copy()

        # Also try the first random subset
        if len(sel) > 0:
            sel, s = local_search(sel, s)
            if s > best_sum:
                best_sum = s
                best_set = sel.copy()

    # Return indices (0-based) into denominators [2,3,...,48]
    # Index i corresponds to denominator denoms[i] = i+2
    result = sorted(best_set)
    if not result:
        # Fallback: should not happen, but ensure nonempty
        # Pick the largest single denominator
        result = [n - 1]  # index for denominator 48

    return result
# EVOLVE-BLOCK-END
