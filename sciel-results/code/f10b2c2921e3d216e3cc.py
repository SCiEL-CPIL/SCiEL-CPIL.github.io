import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    N = 4096
    MAX_SIZE = 80

    def compute_triple_sums(arr):
        """Compute the set of all unordered triple sums for arr."""
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(arr[i] + arr[j] + arr[k])
        return sums

    def can_add(arr, x, triple_sums):
        """Check if x can be added to arr. Returns (valid, new_sums)."""
        n = len(arr)
        new_sums = set()

        s = 3 * x
        if s in triple_sums:
            return False, set()
        new_sums.add(s)

        for y in arr:
            s = 2 * x + y
            if s in triple_sums or s in new_sums:
                return False, set()
            new_sums.add(s)

        for i in range(n):
            for j in range(i, n):
                s = x + arr[i] + arr[j]
                if s in triple_sums or s in new_sums:
                    return False, set()
                new_sums.add(s)

        return True, new_sums

    def build_greedy(order, max_size=MAX_SIZE, deadline=None):
        """Greedy: iterate through order, add element if triple sums remain unique."""
        selected = []
        triple_sums = set()

        for x in order:
            if len(selected) >= max_size:
                break
            if deadline is not None and time.time() > deadline:
                break

            valid, new_sums = can_add(selected, x, triple_sums)
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)

        return selected

    def local_search(arr, max_size=MAX_SIZE, deadline=None):
        """Local search: try to improve by adding elements and doing swaps."""
        if not arr:
            return arr

        arr = list(arr)
        arr_set = set(arr)
        triple_sums = compute_triple_sums(arr)

        # Phase 1: Add elements greedily
        for _round in range(10):
            if len(arr) >= max_size:
                break
            if deadline is not None and time.time() > deadline:
                break

            best_x = None
            for x in range(1, N + 1):
                if x in arr_set:
                    continue
                valid, _ = can_add(arr, x, triple_sums)
                if valid:
                    best_x = x
                    break

            if best_x is None:
                break

            valid, new_sums = can_add(arr, best_x, triple_sums)
            arr.append(best_x)
            arr_set.add(best_x)
            triple_sums.update(new_sums)

        # Phase 2: Try swaps to increase size
        if len(arr) < max_size and deadline is not None and time.time() < deadline - 3:
            improved = True
            while improved and len(arr) < max_size:
                improved = False
                if deadline is not None and time.time() > deadline:
                    break

                for idx in range(len(arr)):
                    if deadline is not None and time.time() > deadline:
                        break
                    if len(arr) >= max_size:
                        break

                    removed = arr[idx]
                    arr.pop(idx)
                    arr_set.discard(removed)
                    triple_sums = compute_triple_sums(arr)

                    # Try to add one or two elements
                    added = []
                    for x in range(1, N + 1):
                        if len(arr) + len(added) >= max_size:
                            break
                        if deadline is not None and time.time() > deadline:
                            break
                        if x in arr_set or x in added:
                            continue
                        valid, new_sums = can_add(arr, x, triple_sums)
                        if valid:
                            arr.append(x)
                            arr_set.add(x)
                            added.append(x)
                            triple_sums.update(new_sums)

                    if len(added) > 1:
                        improved = True
                    else:
                        # Restore
                        for x in added:
                            arr.remove(x)
                            arr_set.discard(x)
                        arr.insert(idx, removed)
                        arr_set.add(removed)
                        triple_sums = compute_triple_sums(arr)

        return arr

    # Strategy 1: Greedy with base-3 sparsity ordering
    def base3_sparsity_score(x):
        digits = []
        while x > 0:
            digits.append(x % 3)
            x //= 3
        if not digits:
            return 0
        non_zero_positions = [i for i, d in enumerate(digits) if d != 0]
        if len(non_zero_positions) <= 1:
            return 100 - len(digits)
        max_gap = 0
        for i in range(1, len(non_zero_positions)):
            gap = non_zero_positions[i] - non_zero_positions[i-1] - 1
            max_gap = max(max_gap, gap)
        return max_gap * 10 - len(digits)

    order1 = sorted(range(1, N + 1), key=lambda x: -base3_sparsity_score(x))
    best = build_greedy(order1, deadline=DEADLINE - 30)
    best_size = len(best)

    # Strategy 2: Simple greedy from 1
    if time.time() < DEADLINE - 30:
        candidate = build_greedy(range(1, N + 1), deadline=DEADLINE - 30)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)

    # Strategy 3: Greedy with powers-of-3 preference
    if time.time() < DEADLINE - 30:
        powers3 = [3**i for i in range(1, 8) if 3**i <= N]
        def p3_score(x):
            min_dist = float('inf')
            for p in powers3:
                min_dist = min(min_dist, abs(x - p))
            return -min_dist
        order3 = sorted(range(1, N + 1), key=lambda x: p3_score(x))
        candidate = build_greedy(order3, deadline=DEADLINE - 30)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)

    # Strategy 4: Local search on best
    if time.time() < DEADLINE - 20:
        improved = local_search(best, deadline=DEADLINE - 20)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    # Strategy 5: Multiple random restarts with local search
    if time.time() < DEADLINE - 10:
        random.seed(42)
        for trial in range(10):
            if time.time() > DEADLINE - 10:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            candidate = build_greedy(order, deadline=DEADLINE - 15)
            if time.time() < DEADLINE - 10:
                candidate = local_search(candidate, deadline=DEADLINE - 10)
            if len(candidate) > best_size:
                best = candidate
                best_size = len(candidate)

    # Final local search
    if time.time() < DEADLINE - 5:
        improved = local_search(best, deadline=DEADLINE - 5)
        if len(improved) > best_size:
            best = improved

    return best
