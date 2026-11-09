import time, random, math

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = 114.0

    def remaining():
        return DEADLINE - (time.time() - t0)

    needed = k * (k + 1) // 2  # pairwise differences per row of length k+1

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def scope_of(rows):
        return max(max(r) for r in rows)

    def sum_of(rows):
        return sum(sum(r) for r in rows)

    def obj(rows):
        return scope_of(rows) * 1000000 + sum_of(rows)

    def is_valid(rows):
        used = set()
        for r in rows:
            if r[0] != 0 or any(r[i] >= r[i + 1] for i in range(len(r) - 1)):
                return False
            d = row_diffs(r)
            if len(d) != needed:
                return False
            if d & used:
                return False
            used |= d
        return True

    # ---- deterministic greedy (baseline, always valid) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # ---- randomized greedy: diversify the myopic "smallest" choice ----
    def rand_greedy(rng, bias, window):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                cands = []
                y = x + 1
                scanned = 0
                while len(cands) < window and scanned < 2000:
                    scanned += 1
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                while not cands:
                    new = {y - a for a in row}
                    if not (new & used):
                        cands.append(y)
                    y += 1
                if rng.random() < bias:
                    chosen = cands[0]
                else:
                    idx = int(len(cands) * (rng.random() ** 1.6))
                    idx = min(idx, len(cands) - 1)
                    chosen = cands[idx]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
                x = chosen
            rows.append(row)
        return rows

    # ---- Row reconstruction: given a fixed set of "blocked" diffs (from other rows),
    #      build a new row of k+1 increasing integers starting at 0, minimizing its max.
    #      This is a structural move: it replaces an entire row rather than tweaking one point.
    def reconstruct_row(blocked, rng, max_scope):
        """Build a row avoiding diffs in `blocked`, starting at 0, k+1 elements.
        Uses a greedy with backtracking-lite: try smallest first, but if stuck,
        allow a small lookahead."""
        row = [0]
        used_local = set()
        x = 0
        while len(row) < k + 1:
            found = False
            y = x + 1
            # Scan forward looking for a valid next element
            limit = max_scope + 20
            while y <= limit:
                new = {y - a for a in row}
                if not (new & blocked) and not (new & used_local):
                    # Check internal uniqueness: all diffs in this row must be distinct
                    # new contains diffs of y with all existing elements
                    # We also need to ensure no diff in `new` collides with used_local
                    if not (new & used_local):
                        row.append(y)
                        used_local |= new
                        x = y
                        found = True
                        break
                y += 1
            if not found:
                return None  # couldn't complete within scope
        return row

    # ---- Simulated-annealing local search with row-reconstruction moves ----
    def local_search(start, rng, steps, T0=3.0):
        cur = [list(r) for r in start]
        rowd = [row_diffs(r) for r in cur]
        cur_used = set()
        for d in rowd:
            cur_used |= d

        row_sums = [sum(r) for r in cur]
        row_maxes = [max(r) for r in cur]
        cur_sum = sum(row_sums)
        cur_scope = max(row_maxes)
        cur_obj = cur_scope * 1000000 + cur_sum

        best = [list(r) for r in cur]
        best_obj = cur_obj
        T = T0

        for it in range(steps):
            if it % 500 == 0 and remaining() < 15:
                break
            r = rng.randrange(n_rows)
            row = cur[r]

            # Decide move type:
            # 40%: single-point mutation (attack max)
            # 20%: single-point mutation (random interior)
            # 20%: step move on max (decrement by 1)
            # 20%: ROW RECONSTRUCTION (structural)
            move_type = rng.random()

            if move_type < 0.4:
                # Attack the maximum
                p = row.index(max(row))
                if p == 0:
                    continue
                lo = row[p - 1] + 1
                hi = row[p] - 1
                if lo > hi:
                    continue
                v = rng.randrange(lo, hi + 1)
                if v == row[p]:
                    continue
                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue
            elif move_type < 0.6:
                # Random interior
                p = rng.randrange(1, len(row))
                lo = row[p - 1] + 1
                hi = (row[p + 1] - 1) if p + 1 < len(row) else (cur_scope + 30)
                if lo > hi:
                    continue
                v = rng.randrange(lo, hi + 1)
                if v == row[p]:
                    continue
                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue
            elif move_type < 0.8:
                # Step move on max
                p = row.index(max(row))
                if p == 0:
                    continue
                v = row[p] - 1
                if v <= row[p - 1]:
                    continue
                new_row = row[:]
                new_row[p] = v
                if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                    continue
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                other = cur_used - rowd[r]
                if nd & other:
                    continue
            else:
                # ROW RECONSTRUCTION: rebuild row r from scratch, avoiding diffs from other rows
                other = cur_used - rowd[r]
                new_row = reconstruct_row(other, rng, cur_scope)
                if new_row is None:
                    # Try with a slightly larger scope to allow flexibility
                    new_row = reconstruct_row(other, rng, cur_scope + 15)
                if new_row is None:
                    continue
                # Verify
                nd = row_diffs(new_row)
                if len(nd) != needed:
                    continue
                if nd & other:
                    continue
                # Accept if it improves or is comparable (SA will handle)
                pass

            # Compute new objective incrementally
            new_max_row = max(new_row)
            new_sum_row = sum(new_row)

            if new_max_row < row_maxes[r]:
                if row_maxes[r] == cur_scope:
                    others_max = max(row_maxes[i] for i in range(n_rows) if i != r) if n_rows > 1 else 0
                    new_scope = max(new_max_row, others_max)
                else:
                    new_scope = cur_scope
            else:
                new_scope = max(cur_scope, new_max_row)

            new_sum = cur_sum - row_sums[r] + new_sum_row
            new_obj = new_scope * 1000000 + new_sum
            d = new_obj - cur_obj

            if d <= 0 or (T > 0.01 and rng.random() < math.exp(-d / T)):
                cur[r] = new_row
                rowd[r] = nd
                cur_used = other | nd
                row_sums[r] = new_sum_row
                row_maxes[r] = new_max_row
                cur_sum = new_sum
                cur_scope = new_scope
                cur_obj = new_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best = [list(x) for x in cur]
            T = max(0.05, T * 0.9997)

        return best

    # ---- Main search ----
    base = greedy()
    best_rows = base
    best_scope = scope_of(base)
    rng = random.Random(20240607)

    # Phase 1: many fast randomized-greedy restarts
    while remaining() > 35:
        bias = rng.uniform(0.55, 0.97)
        window = rng.choice([3, 4, 5, 6, 8, 10])
        rows = rand_greedy(rng, bias, window)
        s = scope_of(rows)
        if s < best_scope:
            best_scope = s
            best_rows = rows
        # Quick polish of promising candidates
        if s < best_scope + 8 and remaining() > 20:
            pol = local_search(rows, rng, 600)
            sp = scope_of(pol)
            if sp < best_scope:
                best_scope = sp
                best_rows = pol

    # Phase 2: deeper annealing from the best + perturbed restarts
    for _ in range(8):
        if remaining() < 10:
            break
        res = local_search(best_rows, rng, 4000, T0=rng.uniform(2.0, 6.0))
        s = scope_of(res)
        if s < best_scope:
            best_scope = s
            best_rows = res
        else:
            # Perturb: try reconstructing a random row, then re-anneal
            pert = [list(r) for r in best_rows]
            rr = rng.randrange(n_rows)
            other = set()
            for i, row in enumerate(pert):
                if i != rr:
                    other |= row_diffs(row)
            new_row = reconstruct_row(other, rng, best_scope + 10)
            if new_row is not None:
                nd = row_diffs(new_row)
                if len(nd) == needed and not (nd & other):
                    pert[rr] = new_row
            res = local_search(pert, rng, 3000, T0=4.0)
            s = scope_of(res)
            if s < best_scope:
                best_scope = s
                best_rows = res

    # Final safety: guarantee a valid result
    if not is_valid(best_rows):
        best_rows = base
    return best_rows
