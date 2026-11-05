"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def construct():
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before the 150s kill

    def remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Ss = set(S)
        for a in S:
            for b in S:
                if a + b in Ss:
                    return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset. Returns list of chosen elements."""
        n = len(arr)
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            ok = True
            # Forward: x + y in chosen for some y
            for y in chosen:
                if x + y in chosen_set:
                    ok = False
                    break
            if not ok:
                continue
            # Backward: x = a + b for some a, b in chosen (a==b allowed)
            if len(chosen) >= 2:
                ch = sorted(chosen)
                m = len(ch)
                found = False
                for i in range(m):
                    a = ch[i]
                    # need b = x - a, with b >= a (to cover a==b and avoid double)
                    b = x - a
                    if b < a:
                        break
                    # binary search b in ch[i:]
                    lo, hi = i, m
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if ch[mid] < b:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo < m and ch[lo] == b:
                        found = True
                        break
                if found:
                    continue
            chosen.append(x)
            chosen_set.add(x)
        return chosen

    def solve_instance(arr):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        # Order 1: by absolute value ascending (small magnitudes first)
        cand = make_greedy(arr, lambda x: (abs(x), x))
        if len(cand) > len(best):
            best = cand
        # Order 2: by value ascending
        cand = make_greedy(arr, lambda x: x)
        if len(cand) > len(best):
            best = cand
        # Order 3: by value descending
        cand = make_greedy(arr, lambda x: -x)
        if len(cand) > len(best):
            best = cand
        # Order 4: by absolute value descending (large magnitudes first)
        cand = make_greedy(arr, lambda x: (-abs(x), x))
        if len(cand) > len(best):
            best = cand
        # Order 5: by absolute value descending, tie-break by value
        cand = make_greedy(arr, lambda x: (-abs(x), -x))
        if len(cand) > len(best):
            best = cand
        # Random restarts while time remains
        rng = random.Random(12345)
        guard = 0
        while remaining() > 4.0 and guard < 200:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = make_greedy(order, lambda x: x)  # identity on pre-shuffled list
            if len(cand) > len(best):
                best = cand
        # Local search: try to improve by swapping elements
        if remaining() > 1.0:
            best = _local_search(arr, best, remaining)
        # Final safety: ensure validity
        if not is_sum_free(best):
            # rebuild deterministically
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    def _local_search(arr, initial, time_budget):
        """Hill-climbing: try removing one element and adding two others."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = arr_set - current_set
        best = current
        best_len = len(current)
        
        def can_add(x, S, S_set):
            """Check if x can be added to sum-free set S."""
            # x + y in S for some y in S
            for y in S:
                if x + y in S_set:
                    return False
            # x = a + b for some a, b in S
            for i, a in enumerate(S):
                b = x - a
                if b in S_set and (b >= a or b == a):
                    # Need to check b is in S and b >= a to avoid double counting
                    # Actually we just need: exists a,b in S with a+b=x
                    # Since we're iterating all a, we check if (x-a) is in S
                    # and we don't need the ordering constraint for validity check
                    pass
            # Simpler: for each a in S, check if (x-a) in S
            for a in S:
                if (x - a) in S_set:
                    return False
            return True
        
        def can_add_fast(x, S_set, S_sorted):
            """Faster check using sorted list."""
            # Check x + y in S for all y in S
            for y in S_sorted:
                if x + y in S_set:
                    return False
            # Check x = a + b for all a in S
            for a in S_sorted:
                if (x - a) in S_set:
                    return False
            return True
        
        rng = random.Random(999)
        iterations = 0
        while time_budget() > 0.5 and iterations < 5000:
            iterations += 1
            # Try to add a random element from not_in
            if not not_in:
                break
            x = rng.choice(tuple(not_in))
            if can_add_fast(x, current_set, current):
                current.append(x)
                current_set.add(x)
                not_in.discard(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                # Try removing one element and adding x
                if current:
                    idx = rng.randrange(len(current))
                    removed = current[idx]
                    current.pop(idx)
                    current_set.discard(removed)
                    if can_add_fast(x, current_set, current):
                        current.append(x)
                        current_set.add(x)
                        not_in.discard(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                    else:
                        # Revert
                        current.append(removed)
                        current_set.add(removed)
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
