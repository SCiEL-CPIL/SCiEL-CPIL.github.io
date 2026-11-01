import time
import random

_START = None
_DEADLINE = 112.0  # safety margin vs 150s kill


def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + _DEADLINE

    n = k + 1
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(r[-1] for r in rows)

    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            ai = row[i]
            for j in range(i + 1, len(row)):
                s.add(row[j] - ai)
        return s

    def full_validate(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                ai = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ai
                    if diff in seen:
                        return None
                    seen.add(diff)
        return get_scope(rows)

    def greedy_init(seed=0):
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init(0)
    best_scope = get_scope(best_rows)
    rng = random.Random(42)

    # ---------- Row reconstruction (ILS move) ----------
    def rebuild_row(used_other, max_val, jitter=0.0, seed=None):
        local_rng = random.Random(seed) if seed is not None else rng
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < n:
            found = False
            for trial_x in range(x + 1, max_val + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=6):
        rows = [list(r) for r in rows]
        improved = True
        pc = 0
        while improved and time.time() < deadline - 12 and pc < max_passes:
            improved = False
            pc += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs(rows[j])
                best_row = list(rows[i])
                best_max = max(rows[i])
                # Try multiple max_val reductions: first try best_max-1, then best_max-2, etc.
                for max_reduction in range(1, 4):
                    if time.time() > deadline - 12:
                        break
                    target_max = best_max - max_reduction
                    if target_max < 0:
                        break
                    for attempt in range(20):
                        jitter = 0.15 if attempt > 5 else 0.0
                        new_row = rebuild_row(used_other, target_max, jitter=jitter)
                        if new_row is not None and max(new_row) < best_max:
                            best_row = new_row
                            best_max = max(new_row)
                            break  # try next max_reduction
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        # Element-level descent: try to reduce each position individually
        if time.time() < deadline - 12:
            for i in range(n_rows):
                if time.time() > deadline - 12:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs(rows[j])
                row = list(rows[i])
                changed = True
                while changed and time.time() < deadline - 12:
                    changed = False
                    for pos in range(len(row) - 1, 0, -1):
                        if time.time() > deadline - 12:
                            break
                        cur_val = row[pos]
                        # Try to reduce row[pos] to any value between row[pos-1]+1 and cur_val-1
                        for trial in range(cur_val - 1, row[pos - 1], -1):
                            new_row = row[:pos] + [trial] + row[pos + 1:]
                            # Check validity: all diffs in this row must be distinct and not in used_other
                            valid = True
                            row_diff_set = set()
                            for a in range(len(new_row)):
                                for b in range(a + 1, len(new_row)):
                                    d = new_row[b] - new_row[a]
                                    if d in used_other or d in row_diff_set:
                                        valid = False
                                        break
                                    row_diff_set.add(d)
                                if not valid:
                                    break
                            if valid:
                                row[pos] = trial
                                changed = True
                                break  # restart from last position
                    if not changed:
                        break
                if row != rows[i]:
                    rows[i] = row
                    improved = True
        return rows

    # ---------- MRV backtracking: tight-scope partition solver ----------
    # Treat differences 1..M as items to assign to rows; a row is a Sidon set
    # of marks. We build rows mark-by-mark; MRV = pick the row whose next
    # mark has the fewest feasible candidates (tightest gap).
    def mrv_solve(M, budget=2.5, seed=None):
        local_rng = random.Random(seed) if seed is not None else rng
        start = time.time()
        used = set()
        rows = [[0] for _ in range(n_rows)]

        # feasible next-mark candidates for a row given its current marks and used diffs
        def candidates(row, M):
            last = row[-1]
            cands = []
            # we must keep strictly increasing; next mark in (last, M]
            for v in range(last + 1, M + 1):
                new = {v - a for a in row}
                if (new & used) or (len(new) != len(row)):  # len(row) new diffs
                    continue
                cands.append(v)
            return cands

        def backtrack(rows, used, depth):
            if time.time() - start > budget:
                return None
            if depth == n_rows:
                # all rows have n marks? check
                if all(len(r) == n for r in rows):
                    return [list(r) for r in rows]
                return None
            # MRV: among rows not yet complete, pick one with fewest candidates
            best_i = -1
            best_cands = None
            best_len = 10 ** 9
            for i in range(n_rows):
                if len(rows[i]) >= n:
                    continue
                c = candidates(rows[i], M)
                if len(c) < best_len:
                    best_len = len(c)
                    best_i = i
                    best_cands = c
                    if best_len == 0:
                        break
            if best_i == -1:
                # all rows complete
                if all(len(r) == n for r in rows):
                    return [list(r) for r in rows]
                return None
            if best_len == 0:
                return None
            # order candidates: smallest first (tighter), occasionally shuffle
            cands = list(best_cands)
            if local_rng.random() < 0.3:
                local_rng.shuffle(cands)
            else:
                cands.sort()
            row = rows[best_i]
            for v in cands:
                new = {v - a for a in row}
                # place
                row.append(v)
                used |= new
                res = backtrack(rows, used, depth + 1)
                if res is not None:
                    return res
                # undo
                used.difference_update(new)
                row.pop()
            return None

        return backtrack(rows, used, 0)

    # ---------- Kick ----------
    def kick(rows, num_kicks=1):
        rows = [list(r) for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs(rows[j])
            success = False
            for attempt in range(12):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, best_scope + 20, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, best_scope + 30, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    # ---------- Main loop ----------
    it = 0
    while time.time() < deadline - 15:
        it += 1
        if it == 1:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))

        cs = full_validate(cur)
        if cs is None:
            continue

        if time.time() < deadline - 15:
            polished = local_polish(cur)
            ps = full_validate(polished)
            if ps is not None:
                cur = polished
                cs = ps

        # Structural move: try MRV tight-solve below current best scope
        if time.time() < deadline - 15:
            # target a scope smaller than current best
            target = min(best_scope - 1, cs - 1)
            if target >= 1:
                sol = mrv_solve(target, budget=2.5, seed=it)
                if sol is not None:
                    ss = full_validate(sol)
                    if ss is not None and ss < best_scope:
                        best_scope = ss
                        best_rows = [list(r) for r in sol]
                        cs = ss
                        cur = [list(r) for r in sol]

        if cs < best_scope:
            best_scope = cs
            best_rows = [list(r) for r in cur]

    # final high-precision polish: exhaustive element-level descent
    if time.time() < deadline - 2:
        polished = [list(r) for r in best_rows]
        # Multiple passes of element-level descent
        for _pass in range(3):
            if time.time() > deadline - 2:
                break
            any_change = False
            for i in range(n_rows):
                if time.time() > deadline - 2:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs(polished[j])
                row = list(polished[i])
                changed = True
                while changed and time.time() < deadline - 2:
                    changed = False
                    for pos in range(len(row) - 1, 0, -1):
                        if time.time() > deadline - 2:
                            break
                        cur_val = row[pos]
                        for trial in range(cur_val - 1, row[pos - 1], -1):
                            new_row = row[:pos] + [trial] + row[pos + 1:]
                            valid = True
                            row_diff_set = set()
                            for a in range(len(new_row)):
                                for b in range(a + 1, len(new_row)):
                                    d = new_row[b] - new_row[a]
                                    if d in used_other or d in row_diff_set:
                                        valid = False
                                        break
                                    row_diff_set.add(d)
                                if not valid:
                                    break
                            if valid:
                                row[pos] = trial
                                changed = True
                                any_change = True
                                break
                        if changed:
                            break
                if row != polished[i]:
                    polished[i] = row
            if not any_change:
                break
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    return best_rows
