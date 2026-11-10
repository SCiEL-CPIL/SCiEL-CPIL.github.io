import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    timeout = 115  # Safety margin for 150s wall-clock limit

    def build_counts_numpy(chosen):
        """Build sum-count array using numpy convolution."""
        if len(chosen) == 0:
            return np.zeros(max_sum + 1, dtype=np.int32)
        arr = np.zeros(N + 1, dtype=np.int32)
        for x in chosen:
            arr[x] = 1
        # Convolution gives counts of a+b where a,b in set (order matters)
        conv = np.convolve(arr, arr, mode='full')
        # conv[s] = number of ordered pairs (a,b) with a+b=s
        # We need unordered pairs with a<=b
        # For s, ordered count = conv[s]
        # Unordered: if s is even, (s/2,s/2) appears once in ordered, so unordered = (conv[s]-1)//2 + 1 if s/2 in set else conv[s]//2
        # Actually simpler: counts[s] = (conv[s] + (1 if s%2==0 and arr[s//2]==1 else 0)) // 2
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for s in range(2, max_sum + 1):
            ordered = conv[s] if s < len(conv) else 0
            if s % 2 == 0 and s // 2 <= N and arr[s // 2] == 1:
                counts[s] = (ordered - 1) // 2 + 1
            else:
                counts[s] = ordered // 2
        return counts

    def can_add_fast(chosen_arr, counts, x):
        """Check if x can be added. chosen_arr is boolean array, counts is sum counts."""
        if chosen_arr[x]:
            return False
        if counts[2 * x] + 1 > 2:
            return False
        # Check all y in chosen where counts[x+y] + 1 > 2
        # Vectorized: find y where chosen_arr[y] and counts[x+y] == 2
        ys = np.where(chosen_arr)[0]
        if len(ys) == 0:
            return True
        sums = x + ys
        valid_sums = sums[(sums <= max_sum) & (sums >= 0)]
        if len(valid_sums) == 0:
            return True
        bad = counts[valid_sums] == 2
        return not np.any(bad)

    def greedy_construct(rng, strategy='gap'):
        """Greedy with different strategies."""
        chosen = []
        chosen_arr = np.zeros(N + 1, dtype=np.int32)
        counts = np.zeros(max_sum + 1, dtype=np.int32)

        if strategy == 'gap':
            # Prefer elements that are far from existing ones
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif strategy == 'ascending':
            candidates = list(range(1, N + 1))
        elif strategy == 'descending':
            candidates = list(range(N, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif strategy == 'sparse':
            # Start with sparse, then fill
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)

        for x in candidates:
            if can_add_fast(chosen_arr, counts, x):
                # Update counts
                for y in chosen:
                    counts[x + y] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_arr[x] = 1

        return chosen, chosen_arr, counts

    def local_search(chosen, chosen_arr, counts, rng, max_time=30):
        """Local search: try to add elements, then try swaps."""
        best_chosen = chosen[:]
        best_len = len(chosen)
        deadline = start_time + max_time

        while time.time() < deadline:
            improved = False

            # Phase 1: Try to add any missing element
            missing = [x for x in range(1, N + 1) if not chosen_arr[x]]
            if missing:
                rng.shuffle(missing)
                for x in missing[:50]:  # Limit per round
                    if can_add_fast(chosen_arr, counts, x):
                        for y in chosen:
                            counts[x + y] += 1
                        counts[2 * x] += 1
                        chosen.append(x)
                        chosen_arr[x] = 1
                        improved = True
                        if len(chosen) > best_len:
                            best_len = len(chosen)
                            best_chosen = chosen[:]
                        break

            if improved:
                continue

            # Phase 2: Try swaps - remove one, add another
            if len(chosen) > 0 and len(missing) > 0:
                # Sample a few removal candidates
                sample_size = min(len(chosen), 10)
                to_try_remove = random.sample(chosen, sample_size)

                for x in to_try_remove:
                    # Temporarily remove x
                    for y in chosen:
                        if y != x:
                            counts[x + y] -= 1
                    counts[2 * x] -= 1
                    chosen.remove(x)
                    chosen_arr[x] = 0

                    # Try to find something to add
                    missing_now = [y for y in range(1, N + 1) if not chosen_arr[y]]
                    if missing_now:
                        rng.shuffle(missing_now)
                        found = False
                        for y in missing_now[:30]:
                            if can_add_fast(chosen_arr, counts, y):
                                for z in chosen:
                                    counts[y + z] += 1
                                counts[2 * y] += 1
                                chosen.append(y)
                                chosen_arr[y] = 1
                                improved = True
                                found = True
                                if len(chosen) > best_len:
                                    best_len = len(chosen)
                                    best_chosen = chosen[:]
                                break

                    if not found:
                        # Put x back
                        for z in chosen:
                            counts[x + z] += 1
                        counts[2 * x] += 1
                        chosen.append(x)
                        chosen_arr[x] = 1

                    if improved:
                        break

            if not improved:
                # Stuck, break
                break

        return best_chosen

    def verify(chosen):
        """Verify B2[2] constraint."""
        counts = {}
        for i, a in enumerate(chosen):
            for b in chosen[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return all(v <= 2 for v in counts.values())

    best_set = []
    best_len = 0

    # Multiple greedy starts with different strategies
    strategies = ['gap', 'random', 'sparse', 'ascending', 'descending']
    rng = random.Random(42)

    for strat in strategies:
        if time.time() - start_time > timeout - 40:
            break
        chosen, chosen_arr, counts = greedy_construct(rng, strat)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]

    # Local search on best
    if best_set and time.time() - start_time < timeout - 20:
        chosen = best_set[:]
        chosen_arr = np.zeros(N + 1, dtype=np.int32)
        for x in chosen:
            chosen_arr[x] = 1
        counts = build_counts_numpy(chosen)
        remaining = min(30, timeout - 20 - (time.time() - start_time))
        if remaining > 5:
            improved_chosen = local_search(chosen, chosen_arr, counts, rng, max_time=remaining)
            if len(improved_chosen) > best_len:
                best_len = len(improved_chosen)
                best_set = improved_chosen[:]

    # More restarts if time allows
    while time.time() - start_time < timeout - 10:
        strat = random.choice(['gap', 'random', 'sparse'])
        chosen, chosen_arr, counts = greedy_construct(random.Random(), strat)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
            # Quick local search
            remaining = min(15, timeout - 10 - (time.time() - start_time))
            if remaining > 3:
                improved_chosen = local_search(chosen, chosen_arr, counts, random.Random(), max_time=remaining)
                if len(improved_chosen) > best_len:
                    best_len = len(improved_chosen)
                    best_set = improved_chosen[:]

    # Final verification
    if best_set and not verify(best_set):
        # Fallback: rebuild from scratch with simpler approach
        chosen = []
        for x in range(1, N + 1):
            ok = True
            for i, a in enumerate(chosen):
                for b in chosen[i:]:
                    if a + b == x + a or a + b == x + b:
                        # Check if this would create a 3rd representation
                        pass
            # Simple check
            counts = {}
            for i, a in enumerate(chosen):
                for b in chosen[i:]:
                    counts[a + b] = counts.get(a + b, 0) + 1
            # Try adding x
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    valid = False
            if valid:
                chosen.append(x)

        best_set = chosen
        best_len = len(chosen)

    return best_set
