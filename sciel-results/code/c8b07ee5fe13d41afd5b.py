"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


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

    def can_add(x, S_set):
        """x conflicts if exists z in S: (x+z) in S or (x-z) in S or (z-x) in S."""
        for z in S_set:
            if (x + z) in S_set or (x - z) in S_set:
                return False
        return True

    def make_greedy(arr, key):
        """Greedy sum-free subset with add-back refinement."""
        chosen = []
        chosen_set = set()
        for x in sorted(arr, key=key):
            if can_add(x, chosen_set):
                chosen.append(x)
                chosen_set.add(x)
        changed = True
        passes = 0
        while changed and passes < 5:
            changed = False
            passes += 1
            for x in arr:
                if x not in chosen_set and can_add(x, chosen_set):
                    chosen.append(x)
                    chosen_set.add(x)
                    changed = True
        return chosen

    def local_search_vectorized(arr, initial, time_budget, seed=999):
        """Vectorized local search: add-only then 2-opt swaps."""
        current = list(initial)
        current_set = set(current)
        arr_set = set(arr)
        not_in = [x for x in arr if x not in current_set]
        best = list(current)
        best_len = len(current)

        rng = random.Random(seed)
        n_arr = len(arr)
        arr_arr = np.array(arr)

        # Precompute for vectorized conflict checking
        # conflict[x_i] = number of z in current such that (x_i+z) in current or (x_i-z) in current
        def compute_conflict_vec(cur_set, cand_list):
            if not cand_list or not cur_set:
                return np.zeros(len(cand_list), dtype=np.int32)
            cur_arr = np.array(sorted(cur_set))
            cand_arr = np.array(cand_list)
            # For each candidate x, count z in cur such that (x+z) in cur or (x-z) in cur
            # Vectorize: for each x, check (x + cur_arr) in cur_set and (x - cur_arr) in cur_set
            # This is O(len(cand) * len(cur)) but vectorized
            # Use broadcasting for small sets
            if len(cand_list) * len(cur_set) < 200000:
                # Broadcasting approach
                x_col = cand_arr[:, None]  # (n_cand, 1)
                z_row = cur_arr[None, :]   # (1, n_cur)
                sum_mat = x_col + z_row     # (n_cand, n_cur)
                diff_mat = x_col - z_row    # (n_cand, n_cur)
                # Check membership in cur_set
                cur_set_arr = cur_arr
                # For each (i,j), check if sum_mat[i,j] is in cur_set
                # Use np.isin with a set for speed
                # Actually, let's use a different approach: build a set lookup
                # For moderate sizes, use set membership in a loop over candidates
                conflicts = np.zeros(len(cand_list), dtype=np.int32)
                for i, x in enumerate(cand_list):
                    # Count z in cur_set where (x+z) in cur_set or (x-z) in cur_set
                    cnt = 0
                    for z in cur_set:
                        if (x + z) in cur_set or (x - z) in cur_set:
                            cnt += 1
                    conflicts[i] = cnt
                return conflicts
            else:
                conflicts = np.zeros(len(cand_list), dtype=np.int32)
                for i, x in enumerate(cand_list):
                    cnt = 0
                    for z in cur_set:
                        if (x + z) in cur_set or (x - z) in cur_set:
                            cnt += 1
                    conflicts[i] = cnt
                return conflicts

        # Phase 1: Add-only with conflict counting
        if not_in:
            conflict_counts = compute_conflict_vec(current_set, not_in)
            not_in_set = set(not_in)

            while time_budget() > 0.5:
                # Find zero-conflict candidates
                zero_mask = conflict_counts == 0
                if not np.any(zero_mask):
                    break

                # Pick the zero-conflict candidate with smallest abs value
                zero_indices = np.where(zero_mask)[0]
                zero_cands = [not_in[i] for i in zero_indices]
                x = min(zero_cands, key=lambda v: abs(v))

                # Add x
                current.append(x)
                current_set.add(x)
                not_in.remove(x)
                not_in_set.discard(x)

                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)

                # Update conflict counts for remaining not_in
                if not_in:
                    # Recompute conflicts for remaining
                    conflict_counts = compute_conflict_vec(current_set, not_in)

        # Phase 2: 2-opt swaps (remove one or two, add one)
        not_in = [x for x in arr if x not in current_set]
        if not_in:
            iterations = 0
            while time_budget() > 0.5 and iterations < 50000:
                iterations += 1
                if not not_in or not current:
                    break
                x = rng.choice(not_in)

                # Try adding x directly
                if can_add(x, current_set):
                    current.append(x)
                    current_set.add(x)
                    not_in.remove(x)
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(current)
                    continue

                # Try 1-opt: remove one element, add x
                improved = False
                for _ in range(3):  # try a few random removals
                    idx = rng.randrange(len(current))
                    removed = current[idx]
                    current.pop(idx)
                    current_set.discard(removed)
                    if can_add(x, current_set):
                        current.append(x)
                        current_set.add(x)
                        not_in.remove(x)
                        if len(current) > best_len:
                            best = list(current)
                            best_len = len(current)
                        improved = True
                        break
                    else:
                        # Try 2-opt: remove another element
                        if current:
                            idx2 = rng.randrange(len(current))
                            removed2 = current[idx2]
                            current.pop(idx2)
                            current_set.discard(removed2)
                            if can_add(x, current_set):
                                current.append(x)
                                current_set.add(x)
                                not_in.remove(x)
                                if len(current) > best_len:
                                    best = list(current)
                                    best_len = len(current)
                                improved = True
                                break
                            else:
                                current.append(removed2)
                                current_set.add(removed2)
                        current.append(removed)
                        current_set.add(removed)
                if not improved:
                    # Revert: x wasn't added
                    pass

        return best

    def solve_instance(arr):
        """Try multiple orderings, return the largest valid sum-free subset."""
        n = len(arr)
        best = []

        # 1. Structural Candidates
        pos = [x for x in arr if x > 0]
        if pos:
            pos_sorted = sorted(pos, reverse=True)
            cand = pos_sorted[:len(pos_sorted) // 2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(pos, lambda x: -x)
                if len(cand) > len(best):
                    best = cand

        neg = [x for x in arr if x < 0]
        if neg:
            neg_sorted = sorted(neg)
            cand = neg_sorted[:len(neg_sorted) // 2 + 1]
            if is_sum_free(cand):
                if len(cand) > len(best):
                    best = cand
            else:
                cand = make_greedy(neg, lambda x: x)
                if len(cand) > len(best):
                    best = cand

        odds = [x for x in arr if x % 2 != 0]
        if odds:
            if len(odds) > len(best):
                best = odds

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

        # 3. Local improvement with vectorized conflict checking
        improved = local_search_vectorized(arr, best, lambda: remaining(), seed=999)
        if len(improved) > len(best):
            best = improved

        # 4. Second local search pass with different seed
        improved2 = local_search_vectorized(arr, best, lambda: remaining(), seed=777)
        if len(improved2) > len(best):
            best = improved2

        # 5. Random Restarts
        rng = random.Random(12345)
        guard = 0
        while remaining() > 2.0 and guard < 30:
            guard += 1
            order = list(arr)
            rng.shuffle(order)
            cand = make_greedy(order, lambda x: x)
            cand = local_search_vectorized(arr, cand, lambda: max(0, remaining() - 1.0), seed=rng.randint(0, 10000))
            if len(cand) > len(best):
                best = cand

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
