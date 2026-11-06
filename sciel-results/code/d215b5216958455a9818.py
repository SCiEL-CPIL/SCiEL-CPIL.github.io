import time
import random

def construct():
    start = time.time()
    DEADLINE = 135.0  # safety margin (limit is 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def conflicts_count(S):
        """Return list of conflict counts per element (optimized)."""
        Sset = set(S)
        n = len(S)
        conflicts = [0] * n
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                s = a + b
                if s in Sset:
                    conflicts[i] += 1
                    if j != i:
                        conflicts[j] += 1
                    # Find index of s in S
                    # Use bisect for sorted S
                    import bisect
                    k = bisect.bisect_left(S, s)
                    if k < n and S[k] == s:
                        conflicts[k] += 1
        return conflicts

    def can_add(x, Sset, S):
        """Check if x can be added to S without creating a sum-free violation."""
        for a in S:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def try_swap(S, Sset, removed, time_limit, start_time):
        """Try to swap: remove one element from S, add one or more from removed."""
        S_list = sorted(S)
        n = len(S)
        if n == 0 or not removed:
            return S, removed, False
        
        improved = False
        # Try removing each element and see if we can add more
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            a = S_list[i]
            # Tentatively remove a
            S_tentative = set(S_list[:i] + S_list[i+1:])
            Sset_tentative = set(S_tentative)
            
            # Try to add elements from removed
            added = []
            for x in removed:
                if time.time() - start_time > time_limit:
                    break
                if can_add(x, Sset_tentative, S_tentative):
                    S_tentative.add(x)
                    Sset_tentative.add(x)
                    added.append(x)
            
            # If we added more than we removed, accept the swap
            if len(added) > 1:  # removed 1, added >1
                S = S_tentative
                Sset = Sset_tentative
                removed = [x for x in removed if x not in added]
                improved = True
                # Re-sort for next iteration
                S_list = sorted(S)
                n = len(S)
        return S, removed, improved

    def greedy_remove(V, time_limit, start_time):
        """Remove elements with most conflicts until sum-free."""
        S = set(V)
        while not is_sum_free(S):
            if time.time() - start_time > time_limit:
                break
            S_list = sorted(S)
            conflicts = conflicts_count(S_list)
            if not any(conflicts):
                break
            max_c = max(conflicts)
            S.remove(S_list[conflicts.index(max_c)])
        return S

    def hill_climb(S, removed, time_limit, start_time):
        """Try to add back removed elements, with swap-based improvements."""
        S = set(S)
        Sset = S
        removed = list(removed)
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            still_removed = []
            random.shuffle(removed)
            for x in removed:
                if (time.time() - start_time) > time_limit:
                    still_removed.append(x)
                    continue
                if can_add(x, Sset, S):
                    S.add(x)
                    Sset.add(x)
                    improved = True
                else:
                    still_removed.append(x)
            removed = still_removed
            
            # Try swap-based improvements
            if removed and (time.time() - start_time) < time_limit:
                S, removed, swap_improved = try_swap(S, Sset, removed, time_limit, start_time)
                Sset = S
                improved = improved or swap_improved
        
        return S, removed

    def modular_seeded(V, time_limit, start_time):
        """
        DIRECT construction: use modular sum-free intervals.
        For a set of integers, numbers in (n/3, 2n/3] mod n are sum-free.
        We apply this to the range of V and use it as a seed, then optimize.
        """
        n = len(V)
        min_v = min(V)
        max_v = max(V)
        
        # Try multiple modular constructions with different moduli
        best_S = set()
        best_removed = list(V)
        
        # Construction 1: modular interval (n/3, 2n/3]
        # For integers in [min_v, max_v], pick those where
        # ((x - min_v) mod M) is in (M/3, 2M/3] for some M
        range_size = max_v - min_v + 1
        candidates_mods = list(set([n, 2*n, 3*n, 4*n, 5*n, range_size, range_size // 2, range_size // 3, range_size // 4]))
        
        for M in candidates_mods:
            if M <= 0:
                continue
            lower = M // 3 + 1
            upper = (2 * M) // 3
            S = set()
            for x in V:
                r = (x - min_v) % M
                if lower <= r <= upper:
                    S.add(x)
            if len(S) > len(best_S):
                best_S = set(S)
                best_removed = [x for x in V if x not in S]
        
        # Construction 2: sign-based (all positive or all negative)
        pos = [x for x in V if x > 0]
        neg = [x for x in V if x < 0]
        if len(pos) > len(best_S):
            best_S = set(pos)
            best_removed = neg
        if len(neg) > len(best_S):
            best_S = set(neg)
            best_removed = pos
        
        # Construction 3: parity (odd numbers are sum-free)
        odds = [x for x in V if x % 2 != 0]
        if len(odds) > len(best_S):
            best_S = set(odds)
            best_removed = [x for x in V if x % 2 == 0]
        
        # Now optimize from the best seed
        # First ensure sum-free
        if not is_sum_free(best_S):
            best_S = greedy_remove(list(best_S), time_limit * 0.3, start_time)
            best_removed = [x for x in V if x not in best_S]
        
        # Hill climb to add more
        if (time.time() - start_time) < time_limit:
            best_S, best_removed = hill_climb(best_S, best_removed, time_limit * 0.7, start_time)
        
        return best_S

    def greedy_all(V, time_limit, start_time):
        """Start from all, greedily remove, then hill climb."""
        S = greedy_remove(V, time_limit * 0.4, start_time)
        removed = [x for x in V if x not in S]
        if (time.time() - start_time) < time_limit:
            S, removed = hill_climb(S, removed, time_limit * 0.6, start_time)
        return S

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            results.append(sorted(V))  # fallback
            continue
        
        # Allocate time per instance based on remaining
        instances_left = 3 - len(results)
        time_per_instance = remaining_total / instances_left - 2
        
        best_size = 0
        best_set = set()
        
        # Strategy 1: modular seeded
        t1 = time_per_instance * 0.5
        S1 = modular_seeded(V, t1, start)
        if len(S1) > best_size:
            best_size = len(S1)
            best_set = set(S1)
        
        # Strategy 2: greedy remove
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t2 = min(time_per_instance * 0.4, DEADLINE - elapsed - 3)
            S2 = greedy_all(V, t2, start)
            if len(S2) > best_size:
                best_size = len(S2)
                best_set = set(S2)
        
        # Strategy 3: more modular variations with different seeds
        elapsed = time.time() - start
        if DEADLINE - elapsed > 5:
            t3 = min(time_per_instance * 0.3, DEADLINE - elapsed - 3)
            # Try with different random shuffles for the hill climb
            for trial in range(3):
                if DEADLINE - (time.time() - start) < 3:
                    break
                rng = random.Random(seed + trial * 1000 + 42)
                # Use a slightly different approach: random subset seed
                sample_size = max(5, size // 4)
                sample = set(rng.sample(V, sample_size))
                if not is_sum_free(sample):
                    sample = greedy_remove(list(sample), min(2.0, t3 / 4.0), start)
                removed = [x for x in V if x not in sample]
                sample, removed = hill_climb(sample, removed, t3 / (3.0 + trial), start)
                if len(sample) > best_size:
                    best_size = len(sample)
                    best_set = set(sample)
        
        # Strategy 4: Final polish - try individual swaps on the best set
        elapsed = time.time() - start
        if DEADLINE - elapsed > 3 and best_set:
            t4 = min(2.0, DEADLINE - elapsed - 1)
            removed_final = [x for x in V if x not in best_set]
            if removed_final:
                best_set, _, _ = try_swap(set(best_set), set(best_set), removed_final, t4, start)
                if len(best_set) > best_size:
                    best_size = len(best_set)
        
        results.append(sorted(best_set))
    
    return results
