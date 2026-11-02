"""Structural shift: replace slow O(n^2) validity checks with an incremental
`used`-set difference representation, and replace the weak simulated-annealing
moves with a time-bounded MRV backtracker that actively targets a tighter
scope bound. Retains the current program's multi-start randomized greedy +
compaction refinement, but every validity/compaction step now runs in O(k)
via the shared used-set, and the backtracker pushes the scope down rather
than only passively compacting."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before the 150s kill

    L = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # counting bound on scope

    best_rows = None
    best_scope = float('inf')

    # ---- fast scope / validity ----
    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_used(rows):
        used = set()
        for row in rows:
            for i in range(L):
                ri = row[i]
                for j in range(i + 1, L):
                    used.add(row[j] - ri)
        return used

    def is_valid(rows):
        return len(build_used(rows)) == total_diffs

    # ---- incremental difference machinery ----
    def diffs_for_pos(row, pos):
        v = row[pos]
        s = set()
        for i in range(L):
            if i < pos:
                s.add(v - row[i])
            elif i > pos:
                s.add(row[i] - v)
        return s

    def can_place(row, pos, new_val, used_set):
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        old = diffs_for_pos(row, pos)
        other = used_set - old
        v = new_val
        for i in range(L):
            if i < pos:
                d = v - row[i]
                if d in other:
                    return False
            elif i > pos:
                d = row[i] - v
                if d in other:
                    return False
        return True

    def apply_set(row, pos, new_val, used_set):
        old = diffs_for_pos(row, pos)
        for d in old:
            used_set.discard(d)
        v = new_val
        for i in range(L):
            if i < pos:
                used_set.add(v - row[i])
            elif i > pos:
                used_set.add(row[i] - v)
        row[pos] = new_val

    def undo_set(row, pos, new_val, old_diffs, used_set):
        """Remove the diffs contributed by row[pos]=new_val and restore old_diffs."""
        v = new_val
        for i in range(L):
            if i < pos:
                used_set.discard(v - row[i])
            elif i > pos:
                used_set.discard(row[i] - v)
        for d in old_diffs:
            used_set.add(d)

    # ---- fast local compaction ----
    def local_compact(rows, time_limit):
        work = [list(r) for r in rows]
        used_set = build_used(work)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = work[r_idx]
                for pos in range(L - 1, 0, -1):
                    old_val = row[pos]
                    lower = row[pos - 1] + 1
                    if lower >= old_val:
                        continue
                    upper = row[pos + 1] if pos < L - 1 else old_val
                    for new_val in range(lower, old_val):
                        if new_val >= upper:
                            break
                        if can_place(row, pos, new_val, used_set):
                            apply_set(row, pos, new_val, used_set)
                            improved = True
                            break
        return work

    # ---- MRV backtracking to hit a target scope ----
    def backtrack_search(rows_init, max_scope, time_limit):
        rows = [list(r) for r in rows_init]
        used = build_used(rows)

        def mvr(row_idx, pos):
            if time.time() > time_limit:
                return False
            if pos == L:
                if row_idx == n_rows - 1:
                    return True
                rows[row_idx + 1][0] = 0
                return mvr(row_idx + 1, 1)
            row = rows[row_idx]
            lower = row[pos - 1] + 1 if pos > 0 else 1
            upper = row[pos + 1] if pos < L - 1 else max_scope
            if upper > max_scope:
                upper = max_scope
            if lower > upper:
                return False
            vals = list(range(lower, upper + 1))
            random.shuffle(vals)
            # bias toward smaller values for tighter scope
            vals.sort(key=lambda v: (v, random.random()))
            for v in vals:
                if not can_place(row, pos, v, used):
                    continue
                orig = row[pos]
                old = diffs_for_pos(row, pos)
                apply_set(row, pos, v, used)
                if mvr(row_idx, pos + 1):
                    return True
                undo_set(row, pos, v, old, used)
                row[pos] = orig
            return False

        rows[0][0] = 0
        used = build_used(rows)
        return mvr(0, 1)

    # ---- greedy construction (multi-strategy, from current program) ----
    def greedy_construct(seed, strategy='basic'):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < L:
                last = row[-1]
                found = False
                if strategy == 'compact':
                    search_end = last + 22 + rng.randint(0, 12)
                elif strategy == 'spread':
                    search_end = last + 45 + rng.randint(0, 35)
                else:
                    search_end = last + 32 + rng.randint(0, 22)
                candidates = list(range(last + 1, min(search_end, lb + 250)))
                if rng.random() < 0.5:
                    rng.shuffle(candidates)
                for x in candidates:
                    nd = [x - a for a in row]
                    if any(d in used for d in nd):
                        continue
                    row.append(x)
                    used.update(nd)
                    found = True
                    break
                if not found:
                    for x in range(last + 1, lb + 400):
                        nd = [x - a for a in row]
                        if any(d in used for d in nd):
                            continue
                        row.append(x)
                        used.update(nd)
                        found = True
                        break
                    if not found:
                        return None
                rows.append(row)
        return rows

    # ---- main search loop ----
    strategies = ['basic', 'compact', 'spread']
    seed_base = 12345
    attempt = 0
    while time.time() < deadline - 8.0:
        strategy = strategies[attempt % len(strategies)]
        rows = greedy_construct(seed_base + attempt * 7919, strategy)
        if rows is None:
            attempt += 1
            continue
        rows = local_compact(rows, time_limit=deadline - 15.0)
        if is_valid(rows):
            s = get_scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]

        # try MRV backtracking to beat current best (only with a baseline)
        if best_rows is not None and time.time() < deadline - 25.0:
            target = best_scope - 1
            if target >= lb:
                t_limit = min(12.0, max(1.0, deadline - 28.0 - time.time()))
                if backtrack_search(best_rows, target, t_limit):
                    if is_valid(best_rows):
                        s2 = get_scope(best_rows)
                        if s2 < best_scope:
                            best_scope = s2
                            best_rows = [r[:] for r in best_rows]
        attempt += 1
        if attempt > 200:
            break

    if best_rows is None:
        # fallback: pure greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                nd = [x - a for a in row]
                if not any(d in used for d in nd):
                    row.append(x)
                    used.update(nd)
            rows.append(row)
        return rows

    return best_rows
# EVOLVE-BLOCK-END
