"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def build_conflict_sets(S_list):
        """Precompute pairwise sums and differences for fast conflict checking."""
        S_set = set(S_list)
        pairwise_sums = set()
        pairwise_diffs = set()
        for i, a in enumerate(S_list):
            for b in S_list[i:]:
                pairwise_sums.add(a + b)
                if a != b:
                    pairwise_diffs.add(a - b)
                    pairwise_diffs.add(b - a)
        return S_set, pairwise_sums, pairwise_diffs

    def fast_greedy_add(S_list, candidates, S_set=None, pairwise_sums=None, pairwise_diffs=None):
        """
        Greedily add elements from candidates to S while maintaining sum-free property.
        Uses conflict-tracking for O(n) per candidate check.
        Returns (new_list, new_set, new_sums, new_diffs).
        """
        if S_set is None:
            S_set, pairwise_sums, pairwise_diffs = build_conflict_sets(S_list)
        result = list(S_list)
        s_list = list(S_list)

        for c in candidates:
            if c in S_set:
                continue
            # Check conflicts: c is invalid if
            # 1. 2*c in S_set
            # 2. c in pairwise_sums (c = a+b for some a,b in S)
            # 3. c in pairwise_diffs (c = a-b for some a,b in S)
            if 2 * c in S_set:
                continue
            if c in pairwise_sums:
                continue
            if c in pairwise_diffs:
                continue
            # c is valid, add it
            result.append(c)
            S_set.add(c)
            for a in s_list:
                pairwise_sums.add(a + c)
                pairwise_diffs.add(a - c)
                pairwise_diffs.add(c - a)
            s_list.append(c)

        return result, S_set, pairwise_sums, pairwise_diffs

    def local_search_improved(S_list, values, time_limit=10):
        """
        Improve S by:
        1. Fast greedy additions
        2. Remove-one-and-readd: remove one element, try to add multiple others
        3. Swap: remove one, add one different
        """
        S_set, pairwise_sums, pairwise_diffs = build_conflict_sets(S_list)
        cand_set = set(values) - S_set
        start_time = time.time()

        # Phase 1: Fast greedy additions
        S_list, S_set, pairwise_sums, pairwise_diffs = fast_greedy_add(
            S_list, list(cand_set), S_set, pairwise_sums, pairwise_diffs)
        cand_set = set(values) - S_set

        # Phase 2: Remove-one-and-readd
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)

            for x in S_list_copy:
                if time.time() - start_time > time_limit:
                    break
                # Try removing x and re-adding others
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                S_test_set, S_test_sums, S_test_diffs = build_conflict_sets(S_list_test)

                # Try to add back elements that were previously excluded
                candidates = [y for y in values if y not in S_test_set and y != x]
                S_list_new, S_set_new, _, _ = fast_greedy_add(
                    S_list_test, candidates, S_test_set, S_test_sums, S_test_diffs)

                if len(S_set_new) > len(S_set):
                    S_list = S_list_new
                    S_set = S_set_new
                    pairwise_sums, pairwise_diffs = build_conflict_sets(S_list)
                    cand_set = set(values) - S_set
                    improved = True
                    break

        # Phase 3: Single swap improvement
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)

            for x in S_list_copy:
                if time.time() - start_time > time_limit:
                    break
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                S_test_set, S_test_sums, S_test_diffs = build_conflict_sets(S_list_test)

                for y in values:
                    if y in S_test_set or y == x:
                        continue
                    # Check if y can be added to S_test
                    if 2 * y in S_test_set:
                        continue
                    if y in S_test_sums:
                        continue
                    if y in S_test_diffs:
                        continue
                    # Swap x with y
                    S_list = S_list_test + [y]
                    S_set = S_test_set | {y}
                    pairwise_sums, pairwise_diffs = build_conflict_sets(S_list)
                    improved = True
                    break
                if improved:
                    break

        return S_list

    def solve_one(values):
        """Find a large sum-free subset of values."""
        if time.time() > deadline:
            return []

        n = len(values)
        if n == 0:
            return []

        # Start with odd numbers (always sum-free)
        odd_subset = [x for x in values if x % 2 != 0]
        best = odd_subset
        best_set = set(best)

        # Try greedy from several orderings, starting from odd_subset
        orderings = [
            sorted(values, key=abs),
            sorted(values, key=abs, reverse=True),
            sorted(values),
            sorted(values, reverse=True),
            # Heuristic: larger absolute values first
            sorted(values, key=lambda x: -abs(x)),
            # Positive first, then negative
            sorted([x for x in values if x > 0] + [x for x in values if x < 0], key=lambda x: -x),
            # Negative first, then positive
            sorted([x for x in values if x < 0] + [x for x in values if x > 0], key=lambda x: -x),
        ]

        for ordering in orderings:
            if time.time() > deadline:
                break
            candidates = [x for x in ordering if x not in best_set]
            if not candidates:
                continue
            trial_list, trial_set, _, _ = fast_greedy_add(list(best), candidates, set(best))
            if len(trial_list) > len(best):
                best = trial_list
                best_set = trial_set

        # Also try from scratch with different orderings
        for ordering in orderings[:4]:
            if time.time() > deadline:
                break
            trial_list, trial_set, _, _ = fast_greedy_add([], ordering)
            if len(trial_list) > len(best):
                best = trial_list
                best_set = trial_set

        # Local search on best
        if time.time() < deadline - 5:
            time_per_instance = min(15, (deadline - time.time()) / 2)
            improved = local_search_improved(best, values, time_limit=time_per_instance)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)

        # Try a few random restarts from scratch
        rng = Random(42)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            shuffled = list(values)
            rng.shuffle(shuffled)
            trial_list, trial_set, _, _ = fast_greedy_add([], shuffled)
            if len(trial_list) > len(best):
                # Local search on this new best
                improved = local_search_improved(trial_list, values, time_limit=5)
                if len(improved) > len(best):
                    best = improved
                    best_set = set(best)

        # Final polish: try to add any remaining elements
        if time.time() < deadline - 2:
            S_set = set(best)
            candidates = [x for x in values if x not in S_set]
            if candidates:
                final_list, final_set, _, _ = fast_greedy_add(best, candidates, S_set)
                if len(final_list) > len(best):
                    best = final_list

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            results.append([])
            continue
        values = instance(size, seed)
        subset = solve_one(values)

        # Final verification and fallback
        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]

        results.append(subset)

    return results
# EVOLVE-BLOCK-END
