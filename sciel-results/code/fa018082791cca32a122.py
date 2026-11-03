import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 140.0
    m = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    max_diff_bound = 256

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def row_diffs_bits(row):
        b = 0
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                b |= (1 << (row[j] - row[i]))
        return b

    def all_diffs_bits(rows):
        b = 0
        for r in rows:
            b |= row_diffs_bits(r)
        return b

    # ---- Row regeneration (core structural move) ----
    def gen_row(used, k_val, randomize, p_smallest=0.8, window=4, rng=None, cap=400):
        if rng is None:
            rng = random
        row = [0]
        cur = set()
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            forbidden = used | cur
            while x < cap:
                new_d = {x - a for a in row}
                if not (new_d & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            if not candidates:
                return None
            if randomize:
                if rng.random() < p_smallest:
                    nxt = candidates[0]
                else:
                    nxt = rng.choice(candidates)
            else:
                nxt = candidates[0]
            row.append(nxt)
            cur |= {nxt - a for a in row[:-1]}
        return row

    # ---- Bitset-based row generation ----
    def gen_row_bits(used_bits, k_val, randomize, p_smallest=0.8, window=4, rng=None, cap=400):
        if rng is None:
            rng = random
        row = [0]
        cur_bits = 0
        for _ in range(k_val):
            candidates = []
            x = row[-1] + 1
            while x < cap:
                new_bits = 0
                conflict = False
                for a in row:
                    d = x - a
                    if d > 256:
                        conflict = True
                        break
                    bit = 1 << d
                    if (used_bits | cur_bits) & bit:
                        conflict = True
                        break
                    if new_bits & bit:
                        conflict = True
                        break
                    new_bits |= bit
                if not conflict:
                    candidates.append((x, new_bits))
                    if len(candidates) >= window:
                        break
                x += 1
            if not candidates:
                return None
            if randomize:
                if rng.random() < p_smallest:
                    nxt, nb = candidates[0]
                else:
                    nxt, nb = rng.choice(candidates)
            else:
                nxt, nb = candidates[0]
            row.append(nxt)
            cur_bits |= nb
        return row

    # ---- Aggressive tightening with 2-element swaps (from inspiration) ----
    def tighten_aggressive(rows, max_iter=300):
        rows = [list(r) for r in rows]
        row_bits = [row_diffs_bits(r) for r in rows]
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            improved = False
            # Phase 1: Pull each element to smallest feasible
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    cand = row[:pos] + [target] + row[pos + 1:]
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    id_bits = 0
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d > 256:
                                ok = False
                                break
                            bit = 1 << d
                            if id_bits & bit:
                                ok = False
                                break
                            id_bits |= bit
                        if not ok:
                            break
                    if not ok:
                        continue
                    other_bits = 0
                    for idx in range(n_rows):
                        if idx != r:
                            other_bits |= row_bits[idx]
                    if id_bits & other_bits:
                        continue
                    row[pos] = target
                    row_bits[r] = id_bits
                    improved = True
            # Phase 2: Try 2-element swaps to escape local minima
            if not improved:
                for r in range(n_rows):
                    row = rows[r]
                    for pos1 in range(1, len(row) - 1):
                        for pos2 in range(pos1 + 1, len(row)):
                            # Try swapping or adjusting two elements
                            for d1 in [-1, 1]:
                                for d2 in [-1, 1]:
                                    nv1 = row[pos1] + d1
                                    nv2 = row[pos2] + d2
                                    if nv1 <= row[pos1 - 1]:
                                        continue
                                    if pos1 < len(row) - 1 and nv1 >= row[pos1 + 1]:
                                        continue
                                    if nv2 <= row[pos2 - 1]:
                                        continue
                                    if pos2 < len(row) - 1 and nv2 >= row[pos2 + 1]:
                                        continue
                                    cand = row[:]
                                    cand[pos1] = nv1
                                    cand[pos2] = nv2
                                    ok = True
                                    for i in range(1, len(cand)):
                                        if cand[i] <= cand[i - 1]:
                                            ok = False
                                            break
                                    if not ok:
                                        continue
                                    id_bits = 0
                                    for i in range(len(cand)):
                                        for j in range(i + 1, len(cand)):
                                            d = cand[j] - cand[i]
                                            if d > 256:
                                                ok = False
                                                break
                                            bit = 1 << d
                                            if id_bits & bit:
                                                ok = False
                                                break
                                            id_bits |= bit
                                        if not ok:
                                            break
                                    if not ok:
                                        continue
                                    other_bits = 0
                                    for idx in range(n_rows):
                                        if idx != r:
                                            other_bits |= row_bits[idx]
                                    if id_bits & other_bits:
                                        continue
                                    rows[r] = cand
                                    row_bits[r] = id_bits
                                    improved = True
                                    break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return rows

    # ---- Deterministic tightening (bitset-accelerated) ----
    def tighten(rows, max_iter=200):
        rows = [list(r) for r in rows]
        row_bits = [row_diffs_bits(r) for r in rows]
        for _ in range(max_iter):
            if time.time() > deadline - 20:
                break
            improved = False
            for r in range(n_rows):
                row = rows[r]
                for pos in range(len(row) - 1, 0, -1):
                    target = row[pos - 1] + 1
                    if target >= row[pos]:
                        continue
                    cand = row[:pos] + [target] + row[pos + 1:]
                    ok = True
                    for i in range(1, len(cand)):
                        if cand[i] <= cand[i - 1]:
                            ok = False
                            break
                    if not ok:
                        continue
                    id_bits = 0
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            d = cand[j] - cand[i]
                            if d > 256:
                                ok = False
                                break
                            bit = 1 << d
                            if id_bits & bit:
                                ok = False
                                break
                            id_bits |= bit
                        if not ok:
                            break
                    if not ok:
                        continue
                    other_bits = 0
                    for idx in range(n_rows):
                        if idx != r:
                            other_bits |= row_bits[idx]
                    if id_bits & other_bits:
                        continue
                    row[pos] = target
                    row_bits[r] = id_bits
                    improved = True
            if not improved:
                break
        return rows

    # ---- Greedy initial solution ----
    def greedy_initial(rng):
        used = set()
        rows = []
        prev_first = 0
        for r in range(n_rows):
            row = None
            attempts = 0
            while row is None and attempts < 200:
                attempts += 1
                row = gen_row(used, k, randomize=True, p_smallest=0.85,
                              window=3, rng=rng, cap=400)
                if row is not None:
                    if r > 0 and row[1] <= prev_first:
                        row = None
                        continue
            if row is None:
                row = gen_row(used, k, randomize=False, rng=rng, cap=500)
            rows.append(row)
            prev_first = row[1]
            used |= row_diffs(row)
        return rows

    # ---- Gap-based backtracking with stronger pruning (from inspiration) ----
    def backtrack_gaps(target_scope, rng, max_nodes=50000):
        used_bits = 0
        rows = []
        nodes_visited = 0
        max_diff = target_scope

        def build_row_gaps(row_idx, gap_pos, current_gaps, current_diffs_bits, current_span):
            nonlocal nodes_visited, used_bits
            if time.time() > deadline - 5:
                return False
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False

            if gap_pos == k:
                row = [0]
                s = 0
                for g in current_gaps:
                    s += g
                    row.append(s)
                rows.append(row)
                used_bits = current_diffs_bits
                return True

            remaining_gaps = k - gap_pos
            min_add_span = remaining_gaps
            max_add_span = target_scope - current_span
            if max_add_span < min_add_span:
                return False

            min_gap = 1
            if gap_pos == k - 1:
                max_gap = max_add_span
            else:
                max_gap = max_add_span - (remaining_gaps - 1)
                if max_gap < min_gap:
                    return False

            valid_candidates = []
            for g in range(min_gap, max_gap + 1):
                new_diffs = []
                suffix = 0
                for i in range(gap_pos - 1, -1, -1):
                    suffix += current_gaps[i]
                    d = g + suffix
                    if d > max_diff:
                        break
                    new_diffs.append(d)

                conflict = False
                for d in new_diffs:
                    if used_bits & (1 << d):
                        conflict = True
                        break
                    if current_diffs_bits & (1 << d):
                        conflict = True
                        break
                if conflict:
                    continue

                seen_new = set()
                for d in new_diffs:
                    if d in seen_new:
                        conflict = True
                        break
                    seen_new.add(d)
                if conflict:
                    continue

                valid_candidates.append((g, new_diffs))

            if not valid_candidates:
                return False

            # MRV-inspired: prefer candidates that constrain future gaps more
            if rng.random() < 0.7:
                valid_candidates.sort(key=lambda x: x[0])
            else:
                rng.shuffle(valid_candidates)

            for g, new_diffs in valid_candidates:
                new_span = current_span + g
                new_diffs_bits = current_diffs_bits
                for d in new_diffs:
                    new_diffs_bits |= (1 << d)

                if build_row_gaps(row_idx, gap_pos + 1, current_gaps + [g], new_diffs_bits, new_span):
                    return True

            return False

        for r in range(n_rows):
            if not build_row_gaps(r, 0, [], 0, 0):
                return None

        if len(rows) != n_rows:
            return None

        return rows

    # ---- Simulated annealing with row regeneration ----
    def anneal(rows, rng, max_iter=20000):
        rows = [list(r) for r in rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        T = 8.0
        cooling = 0.998
        min_T = 0.3

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = row_diffs(old_row)
            used -= old_diffs

            p_smallest = 0.6 + 0.3 * rng.random()
            window = 2 + rng.randint(0, 3)
            new_row = gen_row(used, k, randomize=True, p_smallest=p_smallest,
                              window=window, rng=rng, cap=300)

            if new_row is None:
                used |= old_diffs
                continue

            new_diffs = row_diffs(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_global = max(other_max, new_row[-1])
            delta = new_global - current_max

            if delta < 0:
                rows[r_idx] = new_row
                used |= new_diffs
                current_max = new_global
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            else:
                if delta == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / max(T, 0.01))
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used |= new_diffs
                    current_max = new_global
                else:
                    used |= old_diffs

            T = max(min_T, T * cooling)

        return best_rows

    # ---- Row-rewrite local search ----
    def row_rewrite_search(rows, rng, max_iter=8000):
        rows = [list(r) for r in rows]
        used_bits = all_diffs_bits(rows)
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_idx = rng.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_bits = row_diffs_bits(old_row)
            used_wo = used_bits & ~old_bits

            p_smallest = 0.7 + 0.25 * rng.random()
            window = 2 + rng.randint(0, 3)
            new_row = gen_row_bits(used_wo, k, randomize=True, p_smallest=p_smallest,
                                    window=window, rng=rng, cap=250)
            if new_row is None:
                continue

            new_bits = row_diffs_bits(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_scope = max(other_max, new_row[-1])
            delta = new_scope - current_max

            if delta < 0:
                rows[r_idx] = new_row
                used_bits = used_wo | new_bits
                current_max = new_scope
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
            elif delta == 0:
                if rng.random() < 0.5:
                    rows[r_idx] = new_row
                    used_bits = used_wo | new_bits
                    current_max = new_scope
            else:
                prob = math.exp(-delta / 4.0)
                if rng.random() < prob:
                    rows[r_idx] = new_row
                    used_bits = used_wo | new_bits
                    current_max = new_scope

        return best_rows

    # ---- Local element perturbation ----
    def local_refine(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        row_bits = [row_diffs_bits(r) for r in rows]
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            for delta in [-1, -2, 1, 2]:
                new_val = row[pos] + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                test_row = row[:]
                test_row[pos] = new_val
                id_bits = 0
                ok = True
                for i in range(len(test_row)):
                    for j in range(i + 1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        if d > 256:
                            ok = False
                            break
                        bit = 1 << d
                        if id_bits & bit:
                            ok = False
                            break
                        id_bits |= bit
                    if not ok:
                        break
                if not ok:
                    continue
                other_bits = 0
                for idx in range(n_rows):
                    if idx != r_idx:
                        other_bits |= row_bits[idx]
                if id_bits & other_bits:
                    continue
                other_max = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row[-1])
                if new_scope <= current_max:
                    rows[r_idx] = test_row
                    row_bits[r_idx] = id_bits
                    current_max = new_scope
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break
        return best_rows

    # ---- Main search ----
    rng = random.Random(42)

    # 1. Initial solution
    best_rows = greedy_initial(rng)
    if best_rows is None or not is_valid(best_rows):
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = gen_row(used, k, randomize=False, rng=rng, cap=600)
            best_rows.append(row)
            used |= row_diffs(row)

    # 2. Aggressive tighten (with 2-element swaps)
    best_rows = tighten_aggressive(best_rows, max_iter=300)
    if not is_valid(best_rows):
        best_rows = greedy_initial(rng)
        best_rows = tighten(best_rows)
    best_scope = scope_of(best_rows)

    # 3. Anneal
    rows = anneal(best_rows, rng, max_iter=15000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 4. Local refine
    rows = local_refine(best_rows, rng, max_iter=4000)
    if is_valid(rows):
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s

    # 4b. Row-rewrite search
    rows_rw = row_rewrite_search(best_rows, rng, max_iter=6000)
    if is_valid(rows_rw):
        s_rw = scope_of(rows_rw)
        if s_rw < best_scope:
            best_rows = rows_rw
            best_scope = s_rw

    # 5. Tighten again
    best_rows = tighten_aggressive(best_rows, max_iter=200)
    if is_valid(best_rows):
        best_scope = scope_of(best_rows)

    # 6. Gap-based backtracking search for lower scope
    target = best_scope - 1
    max_bt_attempts = 40
    bt_attempts = 0

    while target > 100 and bt_attempts < max_bt_attempts and time.time() < deadline - 30:
        bt_attempts += 1
        rng_bt = random.Random(10000 + bt_attempts * 7919)

        time_left = deadline - 30 - time.time()
        est_nodes = int(time_left * 600)
        if est_nodes < 5000:
            est_nodes = 5000
        if est_nodes > 60000:
            est_nodes = 60000

        sol = backtrack_gaps(target, rng_bt, max_nodes=est_nodes)

        if sol and is_valid(sol):
            new_scope = scope_of(sol)
            if new_scope < best_scope:
                best_rows = sol
                best_scope = new_scope
                best_rows = tighten_aggressive(best_rows, max_iter=200)
                if is_valid(best_rows):
                    best_scope = scope_of(best_rows)
                rows_ref = local_refine(best_rows, rng_bt, max_iter=2000)
                if is_valid(rows_ref):
                    s_ref = scope_of(rows_ref)
                    if s_ref < best_scope:
                        best_rows = rows_ref
                        best_scope = s_ref
                target = best_scope - 1
            else:
                target -= 1
        else:
            target -= 1

    # 7. Diversified restarts
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        rng2 = random.Random(1000 + trial * 137)
        rows = greedy_initial(rng2)
        if rows is None or not is_valid(rows):
            continue
        rows = tighten_aggressive(rows, max_iter=200)
        if not is_valid(rows):
            continue
        s = scope_of(rows)
        if s < best_scope:
            best_rows = rows
            best_scope = s
        rows2 = anneal(rows, rng2, max_iter=6000)
        if is_valid(rows2):
            s2 = scope_of(rows2)
            if s2 < best_scope:
                best_rows = rows2
                best_scope = s2
        base = rows2 if is_valid(rows2) else rows
        rows_rw = row_rewrite_search(base, rng2, max_iter=3000)
        if is_valid(rows_rw):
            s_rw = scope_of(rows_rw)
            if s_rw < best_scope:
                best_rows = rows_rw
                best_scope = s_rw
        rows3 = local_refine(rows_rw if is_valid(rows_rw) else base, rng2, max_iter=2000)
        if is_valid(rows3):
            s3 = scope_of(rows3)
            if s3 < best_scope:
                best_rows = rows3
                best_scope = s3

    # Final tightening
    best_rows = tighten_aggressive(best_rows, max_iter=300)
    if not is_valid(best_rows):
        best_rows = greedy_initial(random.Random(7))
        best_rows = tighten(best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
