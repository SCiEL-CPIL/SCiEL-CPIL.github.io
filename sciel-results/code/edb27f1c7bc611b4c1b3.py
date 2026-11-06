"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from random import Random

    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Verify S is sum-free."""
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def is_safe_to_add(S, x):
        """Check if x can be added to sum-free set S."""
        # Check x + x = 2x
        if 2 * x in S:
            return False
        # Check a + x = c
        for a in S:
            if a + x in S:
                return False
        # Check a + b = x
        for a in S:
            if x - a in S:
                return False
        return True

    def greedy_build(vals, order_key):
        """Greedily build a sum-free set by adding elements in given order."""
        S = set()
        for x in sorted(vals, key=order_key):
            if is_safe_to_add(S, x):
                S.add(x)
        return S

    def local_search(S, vals, max_iter=100):
        """Try to improve S by swapping elements out for ones in vals\S."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            # Try removing each element
            for a in list(S):
                new_S = S - {a}
                # Try adding each candidate
                best_add = None
                for x in vals:
                    if x in new_S:
                        continue
                    if is_safe_to_add(new_S, x):
                        best_add = x
                        break
                if best_add is not None:
                    S = new_S | {best_add}
                    improved = True
                    break
        return S

    def multi_strategy(vals, rng, time_budget):
        """Try multiple strategies and return the best sum-free subset."""
        candidates = []

        # Strategy 1: Odds (original)
        odds = [x for x in vals if x % 2]
        candidates.append(set(odds))

        # Strategy 2: Greedy by absolute value descending
        S2 = greedy_build(vals, lambda x: -abs(x))
        candidates.append(S2)

        # Strategy 3: Greedy by value descending (large positives first)
        S3 = greedy_build(vals, lambda x: -x)
        candidates.append(S3)

        # Strategy 4: Greedy by value ascending (large negatives first)
        S4 = greedy_build(vals, lambda x: x)
        candidates.append(S4)

        # Strategy 5: Greedy by abs value ascending
        S5 = greedy_build(vals, lambda x: abs(x))
        candidates.append(S5)

        # Strategy 6: Randomized greedy with different seeds
        for trial in range(3):
            shuffled = list(vals)
            rng.shuffle(shuffled)
            S6 = set()
            for x in shuffled:
                ok = True
                for a in S6:
                    if a + x in S6 or x + x in S6:
                        ok = False
                        break
                if ok:
                    S6.add(x)
            candidates.append(S6)

        # Local search on the best candidates
        best = max(candidates, key=len)
        if time.time() < time_budget:
            best = local_search(best, vals, max_iter=100)

        # Try local search on a few more candidates if time permits
        for c in sorted(candidates, key=len, reverse=True)[:3]:
            if time.time() >= time_budget:
                break
            improved = local_search(c, vals, max_iter=50)
            if len(improved) > len(best):
                best = improved
        
        # Final validation
        if not is_sum_free(best):
            # Fallback to odds which is always valid
            best = set([x for x in vals if x % 2])

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        rng = Random(seed + 1000)
        time_budget = min(deadline - 2.0, time.time() + 30.0)  # cap per-instance work
        best_subset = multi_strategy(vals, rng, time_budget)
        results.append(sorted(best_subset))

    return results
# EVOLVE-BLOCK-END
