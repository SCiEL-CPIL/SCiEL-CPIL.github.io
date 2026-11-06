import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
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

    def solve_one(values, time_budget=15.0):
        if time.time() > deadline:
            return []

        n = len(values)
        if n == 0:
            return []

        V_set = set(values)
        val_to_idx = {v: i for i, v in enumerate(values)}

        # Precompute conflict index tables for fast bitmask can_add
        sum_idx = [[-1] * n for _ in range(n)]
        diff_idx = [[-1] * n for _ in range(n)]
        for i in range(n):
            vi = values[i]
            for j in range(n):
                s = vi + values[j]
                if s in val_to_idx:
                    sum_idx[i][j] = val_to_idx[s]
                d = vi - values[j]
                if d in val_to_idx:
                    diff_idx[i][j] = val_to_idx[d]

        double_idx = [-1] * n
        for i in range(n):
            d = 2 * values[i]
            if d in val_to_idx:
                double_idx[i] = val_to_idx[d]

        all_mask = (1 << n) - 1

        def can_add_bitmask(i, mask_S):
            """Check if element at index i can be added to set represented by mask_S."""
            di = double_idx[i]
            if di >= 0 and (mask_S >> di) & 1:
                return False
            m = mask_S
            while m:
                lb = m & (-m)
                j = lb.bit_length() - 1
                si = sum_idx[i][j]
                if si >= 0 and (mask_S >> si) & 1:
                    return False
                di2 = diff_idx[i][j]
                if di2 >= 0 and (mask_S >> di2) & 1:
                    return False
                m ^= lb
            return True

        def greedy_add_bitmask(candidates_idx, initial_mask=0):
            """Greedily add candidate indices to initial_mask."""
            mask = initial_mask
            for i in candidates_idx:
                if (mask >> i) & 1:
                    continue
                if can_add_bitmask(i, mask):
                    mask |= (1 << i)
            return mask

        def mask_to_list(mask):
            result = []
            m = mask
            while m:
                lb = m & (-m)
                i = lb.bit_length() - 1
                result.append(values[i])
                m ^= lb
            return result

        def list_to_mask(lst):
            mask = 0
            for v in lst:
                if v in val_to_idx:
                    mask |= (1 << val_to_idx[v])
            return mask

        def mask_popcount(mask):
            return bin(mask).count('1')

        # Generate diverse initial solutions
        candidates = []

        # Strategy 1: Odd numbers
        s1 = [x for x in values if x % 2 != 0]
        candidates.append(s1)

        # Strategy 2: Large positive numbers (above max/2)
        max_v = max(values)
        thresh_pos = max_v / 2.0
        s2 = [x for x in values if x > thresh_pos]
        candidates.append(s2)

        # Strategy 3: Large negative numbers (below min/2)
        min_v = min(values)
        thresh_neg = min_v / 2.0
        s3 = [x for x in values if x < thresh_neg]
        candidates.append(s3)

        # Strategy 4: Union of large pos and neg
        s4 = sorted(set(s2).union(s3))
        candidates.append(s4)

        # Strategy 5: Greedy by absolute value (largest first)
        s5_idx = sorted(range(n), key=lambda i: abs(values[i]), reverse=True)
        candidates.append(mask_to_list(greedy_add_bitmask(s5_idx)))

        # Strategy 6: Greedy by value (smallest first)
        s6_idx = sorted(range(n), key=lambda i: values[i])
        candidates.append(mask_to_list(greedy_add_bitmask(s6_idx)))

        # Strategy 7: Greedy by value (largest first)
        s7_idx = sorted(range(n), key=lambda i: values[i], reverse=True)
        candidates.append(mask_to_list(greedy_add_bitmask(s7_idx)))

        # Strategy 8: Random shuffles
        rng = Random(42)
        for trial in range(8):
            shuffled_idx = list(range(n))
            rng.shuffle(shuffled_idx)
            candidates.append(mask_to_list(greedy_add_bitmask(shuffled_idx)))

        # Strategy 9: Random perturbation (partial initial, then complete)
        for trial in range(5):
            shuffled_idx = list(range(n))
            rng.shuffle(shuffled_idx)
            initial_size = max(1, n // 4)
            initial_mask = 0
            for i in shuffled_idx[:initial_size]:
                if can_add_bitmask(i, initial_mask):
                    initial_mask |= (1 << i)
            candidates.append(mask_to_list(greedy_add_bitmask(shuffled_idx[initial_size:], initial_mask)))

        # Evaluate and keep best initial
        best = []
        best_mask = 0
        for c in candidates:
            if not is_sum_free(c):
                # Repair by removing conflicting elements
                c_valid = list(c)
                c_set = set(c_valid)
                changed = True
                while changed:
                    changed = False
                    i = 0
                    while i < len(c_valid):
                        x = c_valid[i]
                        conflict = False
                        for y in c_set:
                            if y == x:
                                continue
                            if (x + y) in c_set or (x - y) in c_set:
                                conflict = True
                                break
                        if conflict:
                            c_valid.pop(i)
                            c_set.remove(x)
                            changed = True
                        else:
                            i += 1
                c = c_valid

            if len(c) > len(best):
                best = c
                best_mask = list_to_mask(c)

        # Rich local search using bitmask operations
        def local_search_bitmask(mask, time_limit=5.0):
            start = time.time()
            mask = mask

            # Phase 1: Try adding any missing element
            missing_mask = all_mask ^ mask
            if missing_mask:
                m = missing_mask
                while m:
                    lb = m & (-m)
                    i = lb.bit_length() - 1
                    if can_add_bitmask(i, mask):
                        mask |= (1 << i)
                    m ^= lb

            # Phase 2: 1-flip moves - remove one, try to add others
            improved = True
            while improved and (time.time() - start) < time_limit:
                improved = False
                m = mask
                while m:
                    if (time.time() - start) >= time_limit:
                        break
                    lb = m & (-m)
                    i = lb.bit_length() - 1
                    m ^= lb

                    test_mask = mask & ~(1 << i)
                    missing = all_mask ^ test_mask
                    mm = missing
                    while mm:
                        lb2 = mm & (-mm)
                        j = lb2.bit_length() - 1
                        if can_add_bitmask(j, test_mask):
                            test_mask |= (1 << j)
                        mm ^= lb2

                    if mask_popcount(test_mask) > mask_popcount(mask):
                        mask = test_mask
                        improved = True
                        break

            # Phase 3: 2-flip moves - remove two, add up to three
            if (time.time() - start) < time_limit:
                improved = True
                while improved and (time.time() - start) < time_limit:
                    improved = False
                    mask_list = []
                    m = mask
                    while m:
                        lb = m & (-m)
                        i = lb.bit_length() - 1
                        mask_list.append(i)
                        m ^= lb

                    for i in range(len(mask_list)):
                        if (time.time() - start) >= time_limit:
                            break
                        for j in range(i + 1, len(mask_list)):
                            if (time.time() - start) >= time_limit:
                                break

                            idx1, idx2 = mask_list[i], mask_list[j]
                            test_mask = mask & ~(1 << idx1) & ~(1 << idx2)

                            missing = all_mask ^ test_mask
                            mm = missing
                            added = 0
                            while mm and added < 3:
                                lb2 = mm & (-mm)
                                k = lb2.bit_length() - 1
                                if can_add_bitmask(k, test_mask):
                                    test_mask |= (1 << k)
                                    added += 1
                                mm ^= lb2

                            if mask_popcount(test_mask) > mask_popcount(mask):
                                mask = test_mask
                                improved = True
                                break
                        if improved:
                            break

            # Phase 4: Swap - remove one, add two specific elements
            if (time.time() - start) < time_limit:
                improved = True
                while improved and (time.time() - start) < time_limit:
                    improved = False
                    mask_list = []
                    m = mask
                    while m:
                        lb = m & (-m)
                        i = lb.bit_length() - 1
                        mask_list.append(i)
                        m ^= lb

                    for i in mask_list:
                        if (time.time() - start) >= time_limit:
                            break

                        test_mask = mask & ~(1 << i)
                        missing = all_mask ^ test_mask
                        mm = missing
                        added = 0
                        while mm and added < 2:
                            lb2 = mm & (-mm)
                            j = lb2.bit_length() - 1
                            if can_add_bitmask(j, test_mask):
                                test_mask |= (1 << j)
                                added += 1
                            mm ^= lb2

                        if mask_popcount(test_mask) > mask_popcount(mask):
                            mask = test_mask
                            improved = True
                            break

            return mask

        # Local search on best
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ts = min(time_budget, remaining_time - 3)
            improved_mask = local_search_bitmask(best_mask, time_limit=ts)
            if mask_popcount(improved_mask) > mask_popcount(best_mask):
                best_mask = improved_mask
                best = mask_to_list(best_mask)

        # Additional random restarts with local search if time permits
        if time.time() < deadline - 10:
            for r in range(20):
                if time.time() > deadline - 5:
                    break
                shuffled_idx = list(range(n))
                rng.shuffle(shuffled_idx)
                sl_mask = greedy_add_bitmask(shuffled_idx)
                ts = min(2.5, deadline - time.time() - 2)
                if ts > 0:
                    improved_mask = local_search_bitmask(sl_mask, time_limit=ts)
                    if mask_popcount(improved_mask) > mask_popcount(best_mask):
                        best_mask = improved_mask
                        best = mask_to_list(best_mask)

        return list(best)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)

        # Fallback if time is tight
        if time.time() > deadline - 15:
            results.append([x for x in S if x % 2])
            continue

        res = solve_one(S)

        # Verify
        if not is_sum_free(res):
            res = [x for x in S if x % 2]

        results.append(res)

    return results
