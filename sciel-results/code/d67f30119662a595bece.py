"""Difference triangle set (n_rows, k) via simulated annealing."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 138.0  # safety margin before 150s kill
    rng = random.Random(7)

    def row_diffs(row):
        out = []
        L = len(row)
        for i in range(L):
            ai = row[i]
            for j in range(i + 1, L):
                out.append(row[j] - ai)
        return out

    def scope_of(rows):
        m = 0
        for r in rows:
            v = r[-1]
            if v > m:
                m = v
        return m

    def greedy_row(used, scope):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > scope:
                return None
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                row.append(x)
        return row

    def build_diff_count(rows):
        cnt = {}
        for r in rows:
            for d in row_diffs(r):
                cnt[d] = cnt.get(d, 0) + 1
        return cnt

    def conflict_count(diff_counts):
        c = 0
        for v in diff_counts.values():
            if v > 1:
                c += 1
        return c

    # initial
    used, rows = set(), []
    for _ in range(n_rows):
        r = greedy_row(used, 2000)
        if r is None:
            r = greedy_row(set(), 2000)
        used |= set(row_diffs(r))
        rows.append(r)

    best_rows = [r[:] for r in rows]
    best_scope = scope_of(best_rows)

    # SA
    T = float(best_scope)
    T_min = 0.5
    cooling = 0.9997
    last_improve = time.time()

    while time.time() < deadline:
        if time.time() - last_improve > 25.0:
            # stagnation: reheat
            T = max(T, float(best_scope) * 0.5)
            last_improve = time.time()

        diff_counts = build_diff_count(rows)
        cur_conf = conflict_count(diff_counts)
        cur_scope = scope_of(rows)

        # one move per iteration
        i = rng.randrange(n_rows)
        old_row = rows[i]
        old_diffs = row_diffs(old_row)
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

        # build candidate row within target scope
        target_scope = max(2, best_scope - 1)
        new_row = None
        for _attempt in range(40):
            cand = [0]
            x = 0
            ok = True
            while len(cand) < k + 1:
                x += 1
                if x > target_scope:
                    ok = False
                    break
                if rng.random() < 0.25:
                    x += rng.randrange(0, 4)
                    if x > target_scope:
                        ok = False
                        break
                collide = False
                for a in cand:
                    if (x - a) in diff_counts:
                        collide = True
                        break
                if not collide:
                    cand.append(x)
            if ok and len(cand) == k + 1:
                new_row = cand
                break

        if new_row is None:
            # revert
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            T = max(T_min, T * cooling)
            continue

        new_diffs = row_diffs(new_row)
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        new_scope = scope_of(rows) if False else max(max(r) for r in rows[:i] + [new_row] + rows[i+1:])
        new_conf = conflict_count(diff_counts)

        rows[i] = new_row
        delta = (new_scope - cur_scope) + 0.1 * (new_conf - cur_conf)
        accept = delta < 0 or (rng.random() < math.exp(-delta / max(T, 1e-9)))
        if accept:
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                last_improve = time.time()
        else:
            # revert
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            rows[i] = old_row

        T = max(T_min, T * cooling)

    # validate best_rows
    used = set()
    valid = True
    for r in best_rows:
        for d in row_diffs(r):
            if d in used:
                valid = False
                break
            used.add(d)
        if not valid:
            break
    if valid:
        return best_rows
    # fallback greedy valid
    used, out = set(), []
    for _ in range(n_rows):
        r = greedy_row(used, 3000)
        if r is None:
            break
        used |= set(row_diffs(r))
        out.append(r)
    return out
# EVOLVE-BLOCK-END
