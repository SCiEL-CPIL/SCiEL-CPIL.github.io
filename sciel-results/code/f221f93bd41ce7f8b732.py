import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.

    Strategy:
      1. Fast greedy baseline (deterministic + randomized restarts) to get a good upper bound.
      2. Local search / simulated annealing: repeatedly pick a row, try to rebuild it
         with a smaller scope (using backtracking with randomization), and accept
         improvements or occasional non-improvements to escape local optima.
      3. Final tightening pass to squeeze out the last few units.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin vs 120s/150s kill

    def now():
        return time.time()

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
            for d in diff_set_of_row(r):
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Fast greedy baseline ----------
    def greedy_row(used, temp=0.0, scope_limit=None, rng=None):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if temp > 0 and rng and rng.random() < temp:
                continue
            row.append(x)
            used = used | new_diffs
        return row

    # Deterministic greedy
    used = set()
    base_rows = []
    for _ in range(n_rows):
        r = greedy_row(used, temp=0.0)
        if r is None:
            break
        base_rows.append(r)
        used |= diff_set_of_row(r)
    if len(base_rows) < n_rows:
        for _ in range(n_rows - len(base_rows)):
            r = greedy_row(used, temp=0.0, scope_limit=500)
            base_rows.append(r)
            used |= diff_set_of_row(r)

    best_rows = [r[:] for r in base_rows]
    best_max = max_val(best_rows)

    # Randomized greedy restarts to find a better starting point
    rng = random.Random(42)
    for trial in range(50):
        if now() > deadline - 60:
            break
        temp = 0.05 + 0.1 * trial / 50
        used = set()
        rows = []
        ok = True
        for _ in range(n_rows):
            r = greedy_row(used, temp=temp, rng=rng)
            if r is None:
                ok = False
                break
            rows.append(r)
            used |= diff_set_of_row(r)
        if ok and len(rows) == n_rows:
            m = max_val(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]

    # ---------- Backtracking row builder ----------
    def backtrack_row(used_global, scope_limit, time_limit=None, rng=None, randomize_order=False):
        """Find a valid row of size k+1, max <= scope_limit, disjoint from used_global."""
        if time_limit is not None and now() > time_limit:
            return None
        target_len = k + 1

        def dfs(row, used_local):
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target_len:
                return row[:]
            last = row[-1]
            rem = target_len - len(row)
            max_x = scope_limit - (rem - 1)
            if last + 1 > max_x:
                return None
            candidates = list(range(last + 1, max_x + 1))
            if randomize_order and rng:
                rng.shuffle(candidates)
            for x in candidates:
                if time_limit is not None and now() > time_limit:
                    return None
                conflict = False
                for a in row:
                    if (x - a) in used_local:
                        conflict = True
                        break
                if conflict:
                    continue
                new_row = row + [x]
                new_used = set(used_local)
                for a in row:
                    new_used.add(x - a)
                res = dfs(new_row, new_used)
                if res is not None:
                    return res
            return None

        return dfs([0], set(used_global))

    # ---------- Local search / simulated annealing ----------
    # We repeatedly pick a row, try to rebuild it with a smaller scope, and accept
    # based on SA-like criteria.
    cur_rows = [r[:] for r in best_rows]
    cur_max = best_max
    rng2 = random.Random(7)

    # Precompute diff sets for each row to speed up validation
    def get_all_diffs(rows):
        return [diff_set_of_row(r) for r in rows]

    # SA parameters
    T_start = 5.0
    T_end = 0.1
    cool_rate = 0.995
    T = T_start

    sa_iter = 0
    while now() < deadline - 15 and T > T_end:
        sa_iter += 1
        # Pick a row to modify: prefer the one with the largest max
        idx = max(range(n_rows), key=lambda i: cur_rows[i][-1])
        current_scope = cur_rows[idx][-1]

        # Try to shrink by 1 (or a small random amount)
        shrink = 1 if rng2.random() < 0.8 else rng2.randint(1, 3)
        target_scope = current_scope - shrink
        if target_scope < k + 1:
            T *= cool_rate
            continue

        # Get diffs from all other rows
        other_diffs = set()
        for j in range(n_rows):
            if j != idx:
                other_diffs |= diff_set_of_row(cur_rows[j])

        # Try to build a new row with the target scope
        new_row = backtrack_row(other_diffs, target_scope, time_limit=deadline - 15,
                                rng=rng2, randomize_order=(rng2.random() < 0.3))
        if new_row is None:
            T *= cool_rate
            continue

        # Check if the new row's diffs are disjoint from all other rows' diffs
        new_diffs = diff_set_of_row(new_row)
        if new_diffs & other_diffs:
            T *= cool_rate
            continue

        # Compute new global max
        candidate_rows = cur_rows[:]
        candidate_rows[idx] = new_row
        new_max = max_val(candidate_rows)

        # Acceptance criterion
        delta = new_max - cur_max
        if delta < 0 or (T > 0 and rng2.random() < math.exp(-delta / T)):
            cur_rows = candidate_rows
            cur_max = new_max
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [r[:] for r in cur_rows]
        T *= cool_rate

    # ---------- Final tightening pass ----------
    improved = True
    while improved and now() < deadline - 5:
        improved = False
        cur_rows = [r[:] for r in best_rows]
        cur_max = best_max
        while now() < deadline - 5:
            idx = max(range(n_rows), key=lambda i: cur_rows[i][-1])
            current_scope = cur_rows[idx][-1]
            target_scope = current_scope - 1
            if target_scope < k + 1:
                break
            other_diffs = set()
            for j in range(n_rows):
                if j != idx:
                    other_diffs |= diff_set_of_row(cur_rows[j])
            new_row = backtrack_row(other_diffs, target_scope, time_limit=deadline - 5)
            if new_row is None:
                break
            candidate_rows = cur_rows[:]
            candidate_rows[idx] = new_row
            if not validate(candidate_rows):
                break
            new_max = max_val(candidate_rows)
            if new_max < cur_max:
                cur_max = new_max
                cur_rows = candidate_rows
                improved = True
            else:
                break
        if improved:
            best_max = cur_max
            best_rows = cur_rows

    return best_rows

# EVOLVE-BLOCK-END
