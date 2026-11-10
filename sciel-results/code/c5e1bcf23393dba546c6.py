import time
from math import gcd
import bisect

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # Safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)

    # 2. Weights: index i corresponds to denominator i+2
    weights = [L // (i + 2) for i in range(47)]
    n = 47

    # 3. Greedy initial solution: sort by weight descending, add if fits
    order = sorted(range(n), key=lambda i: -weights[i])
    current_set = set()
    current_sum = 0
    for i in order:
        if current_sum + weights[i] < L:
            current_set.add(i)
            current_sum += weights[i]

    # 4. Local search: 1-swap, 1-add, 1-remove moves
    # Precompute for speed
    w = weights  # alias

    def local_search_1swap(current_set, current_sum):
        """Try all 1-swap, 1-add, 1-remove moves. Return improved set/sum."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            # 1-add: try adding each unselected element
            for j in range(n):
                if j not in current_set:
                    ns = current_sum + w[j]
                    if ns < L and ns > current_sum:
                        current_set.add(j)
                        current_sum = ns
                        improved = True
                        break
            if elapsed() >= DEADLINE:
                break
            if improved:
                continue
            # 1-swap: remove one, add one
            for i in list(current_set):
                base = current_sum - w[i]
                for j in range(n):
                    if j in current_set:
                        continue
                    ns = base + w[j]
                    if ns < L and ns > current_sum:
                        current_set.discard(i)
                        current_set.add(j)
                        current_sum = ns
                        improved = True
                        break
                if elapsed() >= DEADLINE or improved:
                    break
            if elapsed() >= DEADLINE:
                break
            if improved:
                continue
            # 1-remove: remove one element (only if it helps? No, removing reduces sum, so skip)
            # Actually removing never helps maximize sum, so skip
        return current_set, current_sum

    current_set, current_sum = local_search_1swap(current_set, current_sum)

    # 5. 2-swap local search (remove 2, add 2)
    def local_search_2swap(current_set, current_sum):
        """Try all 2-swap moves."""
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            elems = sorted(current_set)
            ne = len(elems)
            for ii in range(ne):
                if elapsed() >= DEADLINE:
                    break
                for jj in range(ii + 1, ne):
                    if elapsed() >= DEADLINE:
                        break
                    i, j = elems[ii], elems[jj]
                    base = current_sum - w[i] - w[j]
                    # Try all pairs (k, l) not in current_set, k < l
                    for k in range(n):
                        if elapsed() >= DEADLINE:
                            break
                        if k in current_set:
                            continue
                        s1 = base + w[k]
                        if s1 >= L:
                            continue
                        for l in range(k + 1, n):
                            if l in current_set:
                                continue
                            ns = s1 + w[l]
                            if ns < L and ns > current_sum:
                                current_set.discard(i)
                                current_set.discard(j)
                                current_set.add(k)
                                current_set.add(l)
                                current_sum = ns
                                improved = True
                                break
                    if improved or elapsed() >= DEADLINE:
                        break
            if elapsed() >= DEADLINE:
                break
        return current_set, current_sum

    current_set, current_sum = local_search_2swap(current_set, current_sum)

    # 6. Repeat 1-swap and 2-swap until no improvement
    while elapsed() < DEADLINE:
        prev_sum = current_sum
        current_set, current_sum = local_search_1swap(current_set, current_sum)
        if current_sum == prev_sum:
            current_set, current_sum = local_search_2swap(current_set, current_sum)
        if current_sum == prev_sum:
            break

    # 7. Final validation
    if not current_set:
        current_set = {0}
        current_sum = w[0]
    if current_sum >= L:
        # Remove largest until valid
        for i in sorted(current_set, key=lambda x: -w[x]):
            if current_sum - w[i] < L:
                current_set.discard(i)
                current_sum -= w[i]
                break
        if current_sum >= L:
            current_set = {0}
            current_sum = w[0]

    return sorted(current_set)
