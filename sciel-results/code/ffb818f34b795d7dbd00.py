import time
import random

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin (limit is 150s)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def can_add(x, Sset, S):
        """O(|S|) check: can x join S without violating sum-free?"""
        for a in S:
            if (x + a) in Sset or (x - a) in Sset:
                return False
        return True

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def greedy_build(order, S, Sset, Vset):
        """Add elements from `order` if they fit; returns (S, Sset)."""
        for x in order:
            if x in Sset:
                continue
            if can_add(x, Sset, S):
                S.add(x)
                Sset.add(x)
        return S, Sset

    def local_search(S, Sset, V, time_budget, start_time, rng):
        """Perturbation-based local search: repeatedly try to add
        any excluded element, and when stuck, randomly remove a few
        elements to escape the local optimum, then re-add greedily.
        Returns the best (S, Sset) found."""
        best_S = set(S)
        best_size = len(S)
        Vset = set(V)

        def try_add_all(S, Sset, budget):
            """Repeatedly add any element that fits until none fit."""
            progress = True
            while progress and (time.time() - start_time) < budget:
                progress = False
                for x in V:
                    if (time.time() - start_time) > budget:
                        break
                    if x in Sset:
                        continue
                    if can_add(x, Sset, S):
                        S.add(x)
                        Sset.add(x)
                        progress = True
            return S, Sset

        S, Sset = try_add_all(S, Sset, time_budget)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

        # Perturbation loop
        while (time.time() - start_time) < time_budget:
            # Try adding once more (cheap check of all excluded)
            improved = False
            for x in V:
                if x in Sset:
                    continue
                if can_add(x, Sset, S):
                    S.add(x)
                    Sset.add(x)
                    improved = True
            if not improved:
                # Stuck: perturb by removing a random subset of S
                n_rem = max(1, min(len(S), rng.randint(2, max(2, len(S) // 4))))
                to_remove = rng.sample(sorted(S), n_rem)
                for x in to_remove:
                    S.discard(x)
                    Sset.discard(x)
                # Re-add greedily
                S, Sset = try_add_all(S, Sset, time_budget)
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
            # Early stop if we hit a strong bound
            if best_size * 3 >= len(V) + 2:
                break
        return best_S

    def solve_instance(V, time_budget, start_time, seed):
        """Solve one instance using randomized greedy + local search."""
        Vset = set(V)
        n = len(V)
        rng = random.Random(seed)
        best_set = set()
        best_size = 0

        # Multiple random-order greedy constructions, then local search on best
        # Number of full attempts scales with available time
        # Each attempt: random greedy build (O(n^2) worst) + local search
        # We budget so that greedy_build is cheap; local_search does the heavy lifting.

        # First, a few quick random-order greedy builds to get a good seed
        quick_budget = time_budget * 0.15
        quick_deadline = start_time + quick_budget
        for _ in range(6):
            if time.time() > quick_deadline:
                break
            order = V[:]
            rng.shuffle(order)
            S = set()
            Sset = set()
            S, Sset = greedy_build(order, S, Sset, Vset)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)

        # Also try a couple of structured seeds (all positive, all negative, odds)
        for subset in ([x for x in V if x > 0], [x for x in V if x < 0],
                       [x for x in V if x % 2 != 0]):
            S = set(subset)
            Sset = set(S)
            # Ensure sum-free by removing violators
            changed = True
            while changed and time.time() < quick_deadline:
                changed = False
                Slist = sorted(S)
                for i in range(len(Slist)):
                    a = Slist[i]
                    for j in range(i, len(Slist)):
                        b = Slist[j]
                        if (a + b) in Sset:
                            # remove the one with the larger absolute value
                            if abs(a) >= abs(b):
                                S.discard(a)
                                Sset.discard(a)
                            else:
                                S.discard(b)
                                Sset.discard(b)
                            changed = True
                            break
                    if changed:
                        break
            # Then try to add more
            S, Sset = greedy_build(V, S, Sset, Vset)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)

        # Now run local search with the remaining budget
        ls_budget = time_budget
        ls_set = local_search(set(best_set), set(best_set), V, ls_budget, start_time, rng)
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_set = ls_set

        # If we still have time, run a second local search from a fresh random seed
        if time.time() - start_time < time_budget * 0.5 and len(best_set) < n:
            order = V[:]
            rng.shuffle(order)
            S = set()
            Sset = set()
            S, Sset = greedy_build(order, S, Sset, Vset)
            ls2 = local_search(S, Sset, V, time_budget * 0.4, start_time, rng)
            if len(ls2) > best_size:
                best_size = len(ls2)
                best_set = ls2

        return best_set

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed

        if remaining_total < 5:
            # Fallback: return a known sum-free subset (all positive, else odds)
            pos = [x for x in V if x > 0]
            if not pos:
                pos = [x for x in V if x % 2 != 0]
            results.append(sorted(pos))
            continue

        instances_left = 3 - len(results)
        time_per_instance = remaining_total / instances_left - 2

        S = solve_instance(V, time_per_instance, start, seed)
        # Final safety: verify sum-free; if violated, trim
        if not is_sum_free(S):
            Sset = set(S)
            Slist = sorted(S)
            changed = True
            while changed:
                changed = False
                for i in range(len(Slist)):
                    a = Slist[i]
                    for j in range(i, len(Slist)):
                        b = Slist[j]
                        if (a + b) in Sset:
                            S.discard(a)
                            Sset.discard(a)
                            Slist = sorted(S)
                            changed = True
                            break
                    if changed:
                        break
        results.append(sorted(S))

    return results
