"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before the 150s kill

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

    def can_add_fast(x, S_set):
        """Fast check if x can be added to sum-free set S.
        x conflicts if: exists z in S such that (x+z) in S, or (x-z) in S, or (z-x) in S.
        """
        for z in S_set:
            if (x + z) in S_set:
                return False
            if (x - z) in S_set:
                return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset with add-back refinement. Returns list of chosen elements."""
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            if can_add_fast(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        # Add-back pass: try adding excluded elements in multiple passes
        all_set = set(arr)
        changed = True
        passes = 0
        while changed and passes < 5:
            changed = False
            passes += 1
            excluded = [x for x in arr if x not in chosen_set]
            for x in excluded:
                if can_add_fast(x, chosen_set):
                    chosen.append(x)
                    chosen_set.add(x)
                    changed = True
        return chosen

    def local_search_add(arr, initial, time_budget):
        """Hill-climbing: try adding elements with conflict-aware priority."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = list(arr_set - current_set)
        best = current
        best_len = len(current)
        
        rng = random.Random(999)
        iterations = 0
        
        # Pre-compute conflict counts for not_in elements
        def compute_conflicts(not_in_list, cur_set):
            cc = {}
            for x in not_in_list:
                cnt = 0
                for z in cur_set:
                    if (x + z) in cur_set or (x - z) in cur_set:
                        cnt += 1
                cc[x] = cnt
            return cc
        
        conflict_count = compute_conflicts(not_in, current_set)
        
        while time_budget() > 0.3 and iterations < 30000:
            iterations += 1
            if not not_in:
                break
            
            # Find zero-conflict elements
            zero_conflict = [x for x in not_in if conflict_count.get(x, 0) == 0]
            
            if zero_conflict:
                # Pick the one with smallest abs value first (heuristic)
                x = min(zero_conflict, key=lambda v: abs(v))
            else:
                # Pick random from not_in
                x = rng.choice(not_in)
            
            if can_add_fast(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                # Update conflict counts for remaining not_in
                for y in not_in:
                    if (x + y) in current_set or (x - y) in current_set or (y - x) in current_set:
                        conflict_count[y] = conflict_count.get(y, 0) + 1
        return best

    def local_search_swap(arr, initial, time_budget):
        """Hill-climbing: try removing one element and adding one or two others."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = list(arr_set - current_set)
        best = current
        best_len = len(current)
        
        rng = random.Random(777)
        iterations = 0
        while time_budget() > 0.3 and iterations < 15000:
            iterations += 1
            if not not_in or not current:
                break
            x = rng.choice(not_in)
            # Try adding x directly
            if can_add_fast(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
                continue
            # Try removing one element and adding x
            idx = rng.randrange(len(current))
            removed = current[idx]
            current.pop(idx)
            current_set.discard(removed)
            if can_add_fast(x, current_set):
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                # Try removing two elements and adding x
                if current:
                    idx2 = rng.randrange(len(current))
                    removed2 = current[idx2]
                    current.pop(idx2)
                    current_set.discard(removed2)
                    if can_add_fast(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.remove(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                    else:
                        # Revert both
                        current.append(removed2)
                        current_set.add(removed2)
                # Revert first
                current.append(removed)
                current_set.add(removed)
        return best

    def solve_instance(arr):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []
        
        # 1. Structural Candidates
        # Positive numbers: keep the largest half to avoid sums
        pos = [x for x in arr if x > 0]
        if pos:
            pos_sorted = sorted(pos, reverse=True)
            cand = pos_sorted[:len(pos_sorted)//2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(pos, lambda x: -x)
                if len(cand) > len(best):
                    best = cand
        
        # Negative numbers: keep the smallest half
        neg = [x for x in arr if x < 0]
        if neg:
            neg_sorted = sorted(neg)
            cand = neg_sorted[:len(neg_sorted)//2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(neg, lambda x: x)
                if len(cand) > len(best):
                    best = cand
        
        # Odd numbers
        odds = [x for x in arr if x % 2 != 0]
        if odds:
            cand = odds
            if len(cand) > len(best):
                best = cand

        # 2. Greedy Candidates with diverse orderings
        orderings = [
            lambda x: (abs(x), x),
            lambda x: x,
            lambda x: -x,
            lambda x: (-abs(x), x),
            lambda x: (-abs(x), -x),
            lambda x: (x % 3, abs(x)),
            lambda x: (-x % 3, -abs(x)),
            lambda x: (abs(x) % 3, abs(x)),
            lambda x: (abs(x) % 5, abs(x)),
            lambda x: (abs(x) % 7, abs(x)),
            lambda x: (abs(x) % 11, abs(x)),
            lambda x: (abs(x) % 13, abs(x)),
            lambda x: (abs(x) % 17, abs(x)),
            lambda x: (x % 2, abs(x)),
            lambda x: (-x % 2, -abs(x)),
            lambda x: (abs(x) % 2, -abs(x)),
            lambda x: (abs(x) % 4, abs(x)),
            lambda x: (abs(x) % 6, abs(x)),
            lambda x: (abs(x) % 8, abs(x)),
            lambda x: (abs(x) % 12, abs(x)),
        ]
        for key in orderings:
            cand = make_greedy(arr, key)
            if len(cand) > len(best):
                best = cand

        # 3. Local Improvement with conflict-count tracking
        improved = local_search_add(arr, best, lambda: remaining())
        if len(improved) > len(best):
            best = improved

        # 4. Swap-based local search to escape local optima
        improved2 = local_search_swap(arr, best, lambda: remaining())
        if len(improved2) > len(best):
            best = improved2

        # 5. Random Restarts
        rng = random.Random(12345)
        guard = 0
        while remaining() > 2.0 and guard < 50:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = make_greedy(order, lambda x: x)
            # Quick local improve on random cand
            cand = local_search_add(arr, cand, lambda: max(0, remaining() - 1.0))
            if len(cand) > len(best):
                best = cand
                # Refine with swap search
                improved3 = local_search_swap(arr, best, lambda: max(0, remaining() - 1.0))
                if len(improved3) > len(best):
                    best = improved3

        # Final safety
        if not is_sum_free(best):
            best = make_greedy(arr, lambda x: (abs(x), x))
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        results.append(solve_instance(arr))
    return results
# EVOLVE-BLOCK-END
