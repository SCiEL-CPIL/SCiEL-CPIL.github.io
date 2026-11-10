import time
import random
import sys
from collections import defaultdict
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0

    def compute_counts(chosen):
        """Compute sum counts for a chosen set."""
        counts = [0] * (max_sum + 1)
        n = len(chosen)
        for i in range(n):
            for j in range(i, n):
                counts[chosen[i] + chosen[j]] += 1
        return counts

    def can_add(chosen, counts, x, chosen_set):
        """Check if x can be added without violating B2[2]."""
        if x in chosen_set:
            return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            if counts[x + y] + 1 > 2:
                return False
        return True

    def do_add(chosen, counts, x, chosen_set):
        """Add x to the set."""
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        chosen_set.add(x)

    def do_remove(chosen, counts, x, chosen_set):
        """Remove x from the set."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
        chosen_set.discard(x)

    def numpy_validate(chosen):
        """Fast numpy-based validation of B2[2] property."""
        if not chosen:
            return True
        arr = np.array(chosen, dtype=np.int32)
        n = len(arr)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        if len(sums) == 0:
            return True
        _, cnt = np.unique(sums, return_counts=True)
        return bool(np.all(cnt <= 2))

    def greedy_build(candidates, counts_init=None):
        """Greedy build from candidate order."""
        if counts_init is not None:
            counts = list(counts_init)
        else:
            counts = [0] * (max_sum + 1)
        chosen = []
        chosen_set = set()
        for x in candidates:
            if x in chosen_set:
                continue
            if can_add(chosen, counts, x, chosen_set):
                do_add(chosen, counts, x, chosen_set)
        return chosen, counts, chosen_set

    def local_search(chosen, counts, chosen_set, time_limit=10.0, rng=None):
        """Local search: insertions, swaps, and 2-move moves."""
        if rng is None:
            rng = random.Random()
        start_time = time.time()
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts = list(counts)

        def try_insert(y):
            if can_add(chosen, counts, y, chosen_set):
                do_add(chosen, counts, y, chosen_set)
                return True
            return False

        def try_swap(x, y):
            """Try to replace x with y."""
            if y in chosen_set or y == x:
                return False
            # Remove x
            for z in chosen:
                if z != x:
                    counts[x + z] -= 1
            counts[2 * x] -= 1
            # Check if y can be added
            if can_add(chosen, counts, y, chosen_set):
                do_add(chosen, counts, y, chosen_set)
                return True
            else:
                # Restore x
                do_add(chosen, counts, x, chosen_set)
                return False

        def try_2move(x1, x2, y1, y2):
            """Try to replace {x1, x2} with {y1, y2}."""
            if y1 in chosen_set or y2 in chosen_set or y1 == y2:
                return False
            if x1 == x2:
                return False
            # Remove x1 and x2
            for z in chosen:
                if z != x1 and z != x2:
                    counts[x1 + z] -= 1
                    counts[x2 + z] -= 1
            if x1 != x2:
                counts[x1 + x2] -= 1
            counts[2 * x1] -= 1
            counts[2 * x2] -= 1

            # Check if y1 and y2 can be added
            if can_add(chosen, counts, y1, chosen_set):
                do_add(chosen, counts, y1, chosen_set)
                if can_add(chosen, counts, y2, chosen_set):
                    do_add(chosen, counts, y2, chosen_set)
                    return True
                else:
                    # Rollback y1
                    do_remove(chosen, counts, y1, chosen_set)
            # Restore x1 and x2
            do_add(chosen, counts, x1, chosen_set)
            do_add(chosen, counts, x2, chosen_set)
            return False

        improved = True
        max_iters = 500
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < time_limit:
            improved = False
            iters += 1
            remaining = time_limit - (time.time() - start_time)
            if remaining <= 0:
                break

            # Phase 1: Pure insertions
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            num_try = min(len(candidates), max(50, int(remaining * 30)))
            for y in candidates[:num_try]:
                if time.time() - start_time > time_limit:
                    break
                if try_insert(y):
                    improved = True

            if improved:
                continue

            # Phase 2: Swaps
            n = len(chosen)
            if n > 0:
                swap_candidates = list(range(1, N + 1))
                rng.shuffle(swap_candidates)
                num_swaps = min(n, max(3, int(remaining * 10)))
                for _ in range(num_swaps):
                    if time.time() - start_time > time_limit:
                        break
                    i = rng.randint(0, n - 1)
                    x = chosen[i]
                    num_y = max(30, int(remaining * 15))
                    for y in swap_candidates[:num_y]:
                        if time.time() - start_time > time_limit:
                            break
                        if try_swap(x, y):
                            improved = True
                            break
                    if improved:
                        break

            if improved:
                continue

            # Phase 3: 2-move swaps
            if n >= 2:
                num_2moves = max(2, int(remaining * 5))
                for _ in range(num_2moves):
                    if time.time() - start_time > time_limit:
                        break
                    i1 = rng.randint(0, n - 1)
                    i2 = rng.randint(0, n - 1)
                    if i1 == i2:
                        continue
                    x1 = chosen[i1]
                    x2 = chosen[i2]
                    num_y = max(10, int(remaining * 5))
                    y_candidates = [y for y in swap_candidates[:num_y * 2] if y != x1 and y != x2]
                    for y1 in y_candidates[:max(5, num_y)]:
                        if time.time() - start_time > time_limit:
                            break
                        for y2 in y_candidates[max(5, num_y):max(10, num_y * 2)]:
                            if time.time() - start_time > time_limit:
                                break
                            if y1 == y2:
                                continue
                            if try_2move(x1, x2, y1, y2):
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        return chosen, counts, chosen_set

    def final_polish(chosen, counts, chosen_set, time_limit=5.0):
        """Systematic final pass trying all insertions."""
        start_time = time.time()
        chosen = list(chosen)
        counts = list(counts)
        chosen_set = set(chosen)

        # Try all numbers in ascending order
        for x in range(1, N + 1):
            if time.time() - start_time > time_limit:
                break
            if can_add(chosen, counts, x, chosen_set):
                do_add(chosen, counts, x, chosen_set)

        # Try all numbers in descending order
        for x in range(N, 0, -1):
            if time.time() - start_time > time_limit:
                break
            if can_add(chosen, counts, x, chosen_set):
                do_add(chosen, counts, x, chosen_set)

        return chosen, counts, chosen_set

    def structured_build():
        """Build using modular arithmetic structure."""
        chosen = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)

        # Start with multiples of small primes
        for p in [3, 5, 7, 11]:
            for k in range(1, N // p + 1):
                x = p * k
                if x <= N and can_add(chosen, counts, x, chosen_set):
                    do_add(chosen, counts, x, chosen_set)

        # Fill in remaining
        for x in range(1, N + 1):
            if can_add(chosen, counts, x, chosen_set):
                do_add(chosen, counts, x, chosen_set)

        return chosen, counts, chosen_set

    start_time = time.time()
    timeout = 140  # Safety margin before 150s

    # Strategy 1: Structured build
    cur_set, cur_counts, cur_set_set = structured_build()
    cur_set, cur_counts, cur_set_set = local_search(cur_set, cur_counts, cur_set_set, time_limit=5)
    cur_set, cur_counts, cur_set_set = final_polish(cur_set, cur_counts, cur_set_set, time_limit=3)
    if len(cur_set) > best_len:
        best_len = len(cur_set)
        best_set = cur_set[:]

    # Strategy 2: Various greedy orderings + local search
    orders = ['random', 'random', 'random', 'ascending', 'descending', 'sparse']
    attempt = 0

    while time.time() - start_time < timeout:
        order = random.choice(orders)
        seed = attempt * 7919 + 12345
        rng_attempt = random.Random(seed)

        if order == 'random':
            candidates = list(range(1, N + 1))
            rng_attempt.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse':
            candidates = list(range(1, N + 1, 2))
            rng_attempt.shuffle(candidates)
            candidates += list(range(2, N + 1, 2))
            rng_attempt.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            rng_attempt.shuffle(candidates)

        cur_set, cur_counts, cur_set_set = greedy_build(candidates)

        time_left = timeout - (time.time() - start_time)
        if time_left > 20:
            ls_time = min(10, time_left - 8)
        elif time_left > 10:
            ls_time = max(3, time_left - 5)
        else:
            ls_time = max(1, time_left - 2)

        cur_set, cur_counts, cur_set_set = local_search(cur_set, cur_counts, cur_set_set, time_limit=ls_time)
        cur_set, cur_counts, cur_set_set = final_polish(cur_set, cur_counts, cur_set_set, time_limit=2)

        if len(cur_set) > best_len:
            best_len = len(cur_set)
            best_set = cur_set[:]

        attempt += 1

        # Periodically try structured build
        if attempt % 5 == 0 and time.time() - start_time < timeout - 15:
            cur_set, cur_counts, cur_set_set = structured_build()
            cur_set, cur_counts, cur_set_set = local_search(cur_set, cur_counts, cur_set_set, time_limit=3)
            cur_set, cur_counts, cur_set_set = final_polish(cur_set, cur_counts, cur_set_set, time_limit=2)
            if len(cur_set) > best_len:
                best_len = len(cur_set)
                best_set = cur_set[:]

    # Final validation
    if best_set and not numpy_validate(best_set):
        # Fallback: rebuild with ascending greedy
        best_set, _, _ = greedy_build(list(range(1, N + 1)))

    return best_set
