import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 138.0  # safety margin before 150s kill

    row_len = k + 1
    d = k * (k + 1) // 2  # differences per row
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def build_diff_index(rows):
        """Build dict: diff -> (row_idx, i, j) for all diffs. Returns None if conflict."""
        index = {}
        for ri, row in enumerate(rows):
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    dv = row[j] - row[i]
                    if dv in index:
                        return None
                    index[dv] = (ri, i, j)
        return index

    def try_move(rows, idx, pos, new_val, diff_index):
        """Try changing rows[idx][pos] to new_val. Returns (success, new_scope_delta).
        Modifies rows in place; caller must revert on failure."""
        row = rows[idx]
        old_val = row[pos]
        if new_val == old_val:
            return False, 0
        # Check strict increasing
        if pos > 0 and new_val <= row[pos - 1]:
            return False, 0
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False, 0

        # Collect all diffs involving this position
        n = len(row)
        affected = []  # list of (diff, i, j)
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d_old = old_val - row[i]
                d_new = new_val - row[i]
                affected.append((d_old, i, pos))
            else:
                d_old = row[i] - old_val
                d_new = row[i] - new_val
                affected.append((d_old, pos, i))

        # Check conflicts: for each affected diff, the other endpoint must not have same diff
        for d_old, i, j in affected:
            if d_old not in diff_index:
                return False, 0
            stored = diff_index[d_old]
            if stored != (idx, i, j):
                # Conflict: this diff is used elsewhere
                return False, 0

        # Now check new diffs don't conflict with existing
        new_diffs = []
        for d_old, i, j in affected:
            if i < pos:
                d_new = new_val - row[i]
            else:
                d_new = row[i] - new_val
            if d_new <= 0:
                return False, 0
            new_diffs.append((d_new, i, j))
            if d_new in diff_index:
                stored = diff_index[d_new]
                if stored != (idx, i, j):
                    return False, 0

        # Apply
        row[pos] = new_val
        for d_old, i, j in affected:
            del diff_index[d_old]
        for d_new, i, j in new_diffs:
            diff_index[d_new] = (idx, i, j)
        return True, 0

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    def rebuild_row(used_other, jitter=0.0, max_val=None, rng_seed=None):
        """Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = rng

        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else (best_scope + 30)
            for trial_x in range(x + 1, search_end + 1):
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

    def local_polish(rows, max_passes=4):
        """Reduce scope by locally re-optimizing each row."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 10:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(rows[j])
                # Try to rebuild row i with smaller max
                best_row = list(rows[i])
                best_max = max(rows[i])
                for attempt in range(30):
                    jitter = 0.1 if attempt > 5 else 0.0
                    new_row = rebuild_row(used_other, jitter=jitter, max_val=best_max - 1)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows

    def kick(rows, num_kicks=1):
        """Perturb by rebuilding random rows with high jitter."""
        rows = [list(r) for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(rows[j])
            success = False
            for attempt in range(15):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row(used_other, jitter=jitter, max_val=best_scope + 20)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                # Fallback: greedy rebuild
                new_row = rebuild_row(used_other, jitter=0.0, max_val=best_scope + 30)
                if new_row is not None:
                    rows[idx] = new_row
                else:
                    # Keep original
                    pass
        return rows

    def targeted_descent(rows, budget=3.0):
        """Try to reduce scope by shrinking the largest entry in max rows."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = False

        while time.time() - start < budget:
            max_val = max(max(r) for r in cur)
            max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
            r = rng.choice(max_rows)
            row = cur[r]
            pos = len(row) - 1
            old_val = row[pos]
            lower = row[pos - 1] + 1

            if old_val <= lower:
                # Try non-last element
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(8):
                    nv = rng.randint(lo2, hi2)
                    if nv == old2:
                        continue
                    row[pos2] = nv
                    scope = full_validate(cur)
                    if scope is not None and scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                    row[pos2] = old2
                if not improved:
                    break
                continue

            candidates = set()
            for v in range(lower, min(lower + 20, old_val)):
                candidates.add(v)
            for _ in range(12):
                candidates.add(rng.randint(lower, old_val - 1))

            for new_val in sorted(candidates, key=lambda x: -x):
                if new_val >= old_val:
                    continue
                row[pos] = new_val
                scope = full_validate(cur)
                if scope is not None:
                    if scope < cur_scope:
                        cur_scope = scope
                        improved = True
                        break
                row[pos] = old_val

            if not improved:
                break

        if improved:
            return cur, cur_scope
        return None, None

    # Phase 1: ILS with row reconstruction
    max_il_iter = 200
    for it in range(max_il_iter):
        if time.time() > deadline - 20:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))

        cur_scope = full_validate(cur)
        if cur_scope is None:
            continue

        # Local polish
        if time.time() < deadline - 15:
            polished = local_polish(cur, max_passes=3)
            pol_scope = full_validate(polished)
            if pol_scope is not None and pol_scope < cur_scope:
                cur = polished
                cur_scope = pol_scope

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Targeted descent
        if time.time() < deadline - 12:
            desc_rows, desc_scope = targeted_descent(cur, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    # Phase 2: Fine-grained SA on single elements with incremental conflict check
    if time.time() < deadline - 18:
        cur = [r[:] for r in best_rows]
        cur_scope = full_validate(cur)
        if cur_scope is None:
            cur = [r[:] for r in best_rows]
            cur_scope = best_scope

        diff_index = build_diff_index(cur)
        if diff_index is None:
            # Fallback to full_validate if index build fails (shouldn't happen)
            T = 0.8
            for step in range(20000):
                if time.time() > deadline - 5:
                    break
                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                old_val = cur[ri][pos]
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue
                cur[ri][pos] = new_val
                new_scope = full_validate(cur)
                if new_scope is not None:
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in cur]
                    delta = new_scope - cur_scope
                    if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                        cur_scope = new_scope
                    else:
                        cur[ri][pos] = old_val
                else:
                    cur[ri][pos] = old_val
                T *= 0.99998
        else:
            T = 0.8
            sa_steps = 200000
            for step in range(sa_steps):
                if time.time() > deadline - 5:
                    break

                ri = rng.randrange(n_rows)
                pos = rng.randint(1, len(cur[ri]) - 1)
                old_val = cur[ri][pos]
                lo = cur[ri][pos - 1] + 1
                hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 15
                if hi <= lo:
                    continue
                new_val = rng.randint(lo, hi)
                if new_val == old_val:
                    continue

                success, _ = try_move(cur, ri, pos, new_val, diff_index)
                if success:
                    new_scope = max(max(r) for r in cur)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in cur]
                    delta = new_scope - cur_scope
                    if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                        cur_scope = new_scope
                    else:
                        # Revert
                        try_move(cur, ri, pos, old_val, diff_index)
                T *= 0.99999

        # Final high-precision polish: systematic neighbor search on best solution
        if time.time() < deadline - 2:
            cur = [r[:] for r in best_rows]
            cur_scope = best_scope
            diff_index = build_diff_index(cur)
            if diff_index is not None:
                improved = True
                while improved and time.time() < deadline - 1:
                    improved = False
                    for ri in range(n_rows):
                        for pos in range(1, len(cur[ri])):
                            if time.time() > deadline - 1:
                                break
                            old_val = cur[ri][pos]
                            lo = cur[ri][pos - 1] + 1
                            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 5
                            for new_val in range(lo, hi + 1):
                                if new_val == old_val:
                                    continue
                                success, _ = try_move(cur, ri, pos, new_val, diff_index)
                                if success:
                                    new_scope = max(max(r) for r in cur)
                                    if new_scope < cur_scope:
                                        cur_scope = new_scope
                                        if new_scope < best_scope:
                                            best_scope = new_scope
                                            best_rows = [r[:] for r in cur]
                                        improved = True
                                        break
                                    else:
                                        # Revert
                                        try_move(cur, ri, pos, old_val, diff_index)
                    # Also try decreasing the last element of each row
                    for ri in range(n_rows):
                        if time.time() > deadline - 1:
                            break
                        pos = len(cur[ri]) - 1
                        old_val = cur[ri][pos]
                        lo = cur[ri][pos - 1] + 1
                        for new_val in range(old_val - 1, lo - 1, -1):
                            if new_val <= lo:
                                break
                            success, _ = try_move(cur, ri, pos, new_val, diff_index)
                            if success:
                                new_scope = max(max(r) for r in cur)
                                if new_scope < cur_scope:
                                    cur_scope = new_scope
                                    if new_scope < best_scope:
                                        best_scope = new_scope
                                        best_rows = [r[:] for r in cur]
                                    improved = True
                                    break
                                else:
                                    try_move(cur, ri, pos, old_val, diff_index)

            # Final targeted descent
            if time.time() < deadline - 1:
                desc_rows, desc_scope = targeted_descent(best_rows, budget=1.5)
                if desc_rows is not None and desc_scope < best_scope:
                    best_scope = desc_scope
                    best_rows = desc_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
