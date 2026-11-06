import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set, S_list):
        """Check if x can be added to S without violating sum-free property."""
        for a in S_list:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def get_conflicts(S_set, S_list):
        """Count how many times each element is involved in a sum-free violation."""
        conflicts = {}
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                b = S_list[j]
                s = a + b
                if s in S_set:
                    conflicts[a] = conflicts.get(a, 0) + 1
                    if b != a:
                        conflicts[b] = conflicts.get(b, 0) + 1
                    conflicts[s] = conflicts.get(s, 0) + 1
        return conflicts

    def greedy_remove_then_add(V, time_limit, rng_seed=None):
        """
        Greedy removal to get sum-free set, then hill-climbing add-back.
        Returns the best sum-free subset found.
        """
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()

        S = set(V)
        removed = list(V)

        # Phase 1: Greedy removal
        while not is_sum_free(S):
            if time.time() > start_time + time_limit:
                break

            conflicts = get_conflicts(S, list(S))
            if not conflicts:
                break

            max_deg = max(conflicts.values())
            candidates = [x for x, c in conflicts.items() if c == max_deg]
            x_to_remove = rng.choice(candidates)

            S.discard(x_to_remove)
            if x_to_remove in removed:
                removed.remove(x_to_remove)

        # Phase 2: Hill climbing - add back elements
        improved = True
        while improved and time.time() < start_time + time_limit:
            improved = False
            new_removed = []
            rng.shuffle(removed)
            S_list = list(S)
            S_set = S

            for x in removed:
                if time.time() > start_time + time_limit:
                    new_removed.append(x)
                    continue

                if can_add(x, S_set, S_list):
                    S.add(x)
                    S_list.append(x)
                    improved = True
                else:
                    new_removed.append(x)

            if improved:
                removed = new_removed

        return list(S)

    def diverse_init_local_search(V, time_limit, num_starts=6):
        """
        Try multiple diverse initializations and local search from each.
        """
        best_S = []
        best_size = 0
        n = len(V)

        # Time per start
        time_per_start = max(1.0, time_limit / num_starts - 1)

        # Define diverse initializations
        inits = []

        # 1. Odds only
        odds = [x for x in V if x % 2 != 0]
        if odds:
            inits.append(odds)

        # 2. Positives only
        pos = [x for x in V if x > 0]
        if pos:
            inits.append(pos)

        # 3. Negatives only
        neg = [x for x in V if x < 0]
        if neg:
            inits.append(neg)

        # 4. High absolute value elements
        sorted_by_abs = sorted(V, key=lambda x: abs(x), reverse=True)
        inits.append(sorted_by_abs[:n//2])

        # 5. Random sample
        rng = random.Random()
        inits.append(rng.sample(V, n//2))

        # 6. Another random sample
        inits.append(rng.sample(V, n//3))

        for init in inits:
            if time.time() > start_time + time_limit - 2:
                break

            # First make the init sum-free via greedy removal
            S_init = set(init)
            while not is_sum_free(S_init):
                if time.time() > start_time + time_limit - 2:
                    break
                conflicts = get_conflicts(S_init, list(S_init))
                if not conflicts:
                    break
                max_deg = max(conflicts.values())
                candidates = [x for x, c in conflicts.items() if c == max_deg]
                x_to_remove = random.choice(candidates)
                S_init.discard(x_to_remove)

            # Now do local search add-back
            remaining = [x for x in V if x not in S_init]
            improved = True
            while improved and time.time() < start_time + time_limit - 1:
                improved = False
                new_remaining = []
                random.shuffle(remaining)
                S_list = list(S_init)
                S_set = S_init

                for x in remaining:
                    if time.time() > start_time + time_limit - 1:
                        new_remaining.append(x)
                        continue
                    if can_add(x, S_set, S_list):
                        S_init.add(x)
                        S_list.append(x)
                        improved = True
                    else:
                        new_remaining.append(x)

                if improved:
                    remaining = new_remaining

            if len(S_init) > best_size:
                best_size = len(S_init)
                best_S = list(S_init)

        # Also try the full greedy approach with different seeds
        for r in range(2):
            if time.time() > start_time + time_limit - 2:
                break
            res = greedy_remove_then_add(V, time_limit - (time.time() - start_time) - 1, rng_seed=r*1000)
            if len(res) > best_size:
                best_size = len(res)
                best_S = res

        if not best_S:
            # Fallback: odds
            best_S = [x for x in V if x % 2 != 0]

        return best_S

    results = []

    # Time allocation: give more time to larger instances
    # Instance 1 (60): fast
    # Instance 2 (120): medium
    # Instance 3 (240): slow
    time_alloc = {60: 8.0, 120: 20.0, 240: 75.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)

        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        if remaining_time < 3:
            # Emergency: just return odds
            odds = [x for x in V if x % 2 != 0]
            results.append(odds)
            continue

        t_limit = min(time_alloc.get(size, 15.0), remaining_time - 3)

        # Run diverse init local search
        best_S = diverse_init_local_search(V, t_limit, num_starts=6)

        # Final verification
        if not is_sum_free(best_S):
            # Fallback to odds
            best_S = [x for x in V if x % 2 != 0]

        results.append(sorted(best_S))

    return results
