import time
import random
import math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 116.0  # safety margin vs 120s limit

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1

    def get_diff_mask(row):
        mask = 0
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                d = row[j] - ai
                mask |= (1 << d)
        return mask

    def scope_of(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used_mask = 0
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            m = get_diff_mask(r)
            if m & used_mask:
                return False
            if bin(m).count("1") != needed:
                return False
            used_mask |= m
        return True

    # ---- Deterministic greedy (baseline, always valid) ----
    def greedy():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs_mask = 0
                conflict = False
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        conflict = True
                        break
                    new_diffs_mask |= bit
                if not conflict:
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
        return rows

    # ---- Randomized Backtracking Constructor (with internal-duplicate guard) ----
    def backtrack_construct(target_max, rng, time_budget=3.0):
        used_mask = 0
        rows = []
        t_start = time.time()

        def build_row(used_mask, target_max):
            def _build(cur_row, cur_mask, depth):
                if depth == k:
                    return cur_row[:]
                if time.time() - t_start > time_budget:
                    return None
                last = cur_row[-1]
                rem = k - depth - 1
                candidates = []
                for x in range(last + 1, target_max + 1):
                    if x > target_max - rem:
                        continue  # not enough room for remaining elements
                    nm = 0
                    ok = True
                    for a in cur_row:
                        d = x - a
                        bit = 1 << d
                        if (used_mask & bit) or (cur_mask & bit):
                            ok = False
                            break
                        nm |= bit
                    if ok:
                        # internal-duplicate guard: all diffs in this row must be distinct
                        if bin(nm).count("1") != len(cur_row):
                            continue
                        candidates.append((x, nm))
                if not candidates:
                    return None
                rng.shuffle(candidates)
                for x, nm in candidates:
                    res = _build(cur_row + [x], cur_mask | nm, depth + 1)
                    if res is not None:
                        return res
                return None

            return _build([0], 0, 0)

        for _ in range(n_rows):
            if time.time() - t_start > time_budget:
                return None
            row = build_row(used_mask, target_max)
            if row is None:
                return None
            rows.append(row)
            used_mask |= get_diff_mask(row)
        return rows

    # ---- Simulated Annealing (with 2-opt swap move) ----
    def local_search(start, rng, max_time=8.0, T0=5.0):
        cur = [list(r) for r in start]
        row_masks = [get_diff_mask(r) for r in cur]
        cur_used_mask = 0
        for m in row_masks:
            cur_used_mask |= m

        row_maxes = [max(r) for r in cur]
        cur_scope = max(row_maxes)
        cur_sum = sum(sum(r) for r in cur)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0
        t_start = time.time()

        def try_rebuild_row(other_mask, limit, rng):
            def _bt(cur_row, cur_mask, depth):
                if depth == k:
                    return cur_row[:]
                last = cur_row[-1]
                rem = k - depth - 1
                cands = []
                for x in range(last + 1, limit + 1):
                    if x > limit - rem:
                        continue
                    nm = 0
                    ok = True
                    for a in cur_row:
                        d = x - a
                        bit = 1 << d
                        if (other_mask & bit) or (cur_mask & bit):
                            ok = False
                            break
                        nm |= bit
                    if ok:
                        if bin(nm).count("1") != len(cur_row):
                            continue
                        cands.append((x, nm))
                if not cands:
                    return None
                rng.shuffle(cands)
                for x, nm in cands:
                    res = _bt(cur_row + [x], cur_mask | nm, depth + 1)
                    if res:
                        return res
                return None
            return _bt([0], 0, 0)

        def apply_row(r, new_row, new_mask):
            nonlocal cur, row_masks, cur_used_mask, row_maxes, cur_scope, cur_sum, cur_obj, best, best_obj
            other_used_mask = cur_used_mask & ~(row_masks[r])
            cur[r] = new_row
            row_masks[r] = new_mask
            cur_used_mask = other_used_mask | new_mask
            new_max_row = max(new_row)
            row_maxes[r] = new_max_row
            cur_sum = cur_sum - sum(new_row) + sum(new_row)  # recompute below properly
            cur_sum = sum(sum(row) for row in cur)
            cur_scope = max(row_maxes)
            cur_obj = cur_scope * 1000000 + cur_sum
            if cur_obj < best_obj:
                best_obj = cur_obj
                best = [list(x) for x in cur]

        while time.time() - t_start < max_time and remaining() > 5:
            if rng.random() < 0.6:
                r = row_maxes.index(cur_scope)
            else:
                r = rng.randrange(n_rows)

            row = cur[r]
            other_used_mask = cur_used_mask & ~(row_masks[r])
            move_type = rng.random()
            new_row = None
            new_mask = None

            if move_type < 0.4:
                # MOVE 1: targeted compression of the max element
                p = len(row) - 1
                current_max = row[p]
                prev_val = row[p - 1]
                for v in range(current_max - 1, prev_val, -1):
                    test_row = row[:]
                    test_row[p] = v
                    nd_test = get_diff_mask(test_row)
                    if bin(nd_test).count("1") != needed:
                        continue
                    if nd_test & other_used_mask:
                        continue
                    new_row = test_row
                    new_mask = nd_test
                    break
            elif move_type < 0.65:
                # MOVE 2: random interior mutation
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 15)
                if lo > hi:
                    continue
                span = hi - lo + 1
                v = lo + int(span * (rng.random() ** 1.3))
                if v == row[p]:
                    continue
                test_row = row[:]
                test_row[p] = v
                nd_test = get_diff_mask(test_row)
                if bin(nd_test).count("1") != needed:
                    continue
                if nd_test & other_used_mask:
                    continue
                new_row = test_row
                new_mask = nd_test
            elif move_type < 0.85:
                # MOVE 3: 2-opt swap of two interior elements
                p = rng.randrange(1, len(row) - 1)
                q = rng.randrange(p + 1, len(row))
                if row[p] == row[q]:
                    continue
                test_row = row[:]
                test_row[p], test_row[q] = row[q], row[p]
                if any(test_row[i] >= test_row[i + 1] for i in range(len(test_row) - 1)):
                    continue
                nd_test = get_diff_mask(test_row)
                if bin(nd_test).count("1") != needed:
                    continue
                if nd_test & other_used_mask:
                    continue
                new_row = test_row
                new_mask = nd_test
            else:
                # MOVE 4: full row reconstruction
                new_row = try_rebuild_row(other_used_mask, cur_scope, rng)
                if new_row is None:
                    new_row = try_rebuild_row(other_used_mask, cur_scope + 8, rng)
                if new_row is None:
                    continue
                new_mask = get_diff_mask(new_row)
                if bin(new_mask).count("1") != needed or (new_mask & other_used_mask):
                    continue

            if new_row is None:
                continue

            new_max_row = max(new_row)
            new_sum = cur_sum - sum(row) + sum(new_row)
            new_scope = max(cur_scope, new_max_row)
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj

            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                other_used_mask = cur_used_mask & ~(row_masks[r])
                cur[r] = new_row
                row_masks[r] = new_mask
                cur_used_mask = other_used_mask | new_mask
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]

            T = max(0.05, T * 0.999)

        return best

    # ---- Main Search Loop ----
    best_rows = greedy()
    best_scope = scope_of(best_rows)

    rng = random.Random(42)

    while remaining() > 8:
        target = best_scope - 1
        bt_time_budget = min(2.5, remaining() * 0.3)
        res = backtrack_construct(target, rng, time_budget=bt_time_budget)
        if res is not None:
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

        if remaining() > 5:
            sa_time = min(8.0, remaining() * 0.5)
            res = local_search(best_rows, rng, max_time=sa_time)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    if not is_valid(best_rows):
        best_rows = greedy()

    return best_rows
