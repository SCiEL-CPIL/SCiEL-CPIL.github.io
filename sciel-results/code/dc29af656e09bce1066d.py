import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 135.0  # safety margin before 150s kill

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
            while len(row) < k + 1:
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
        while len(row) < k + 1:
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

    def gap_swap_move(rows, used_diffs, cur_scope):
        """
        Structural move: pick two rows and swap gap values between them.
        A gap is the difference between consecutive elements in a row.
        We try to rebalance spans by swapping gaps between rows.
        Returns (new_rows, new_scope) if improved, else None.
        """
        # Compute gaps for each row
        gaps = []
        for r in rows:
            g = [r[i+1] - r[i] for i in range(len(r)-1)]
            gaps.append(g)
        
        # Pick two random rows
        i, j = rng.sample(range(n_rows), 2)
        if len(gaps[i]) != len(gaps[j]):
            return None
        
        # Pick a position in the gap sequences
        pos = rng.randint(0, len(gaps[i]) - 1)
        
        # Try swapping gaps[i][pos] and gaps[j][pos]
        old_gap_i = gaps[i][pos]
        old_gap_j = gaps[j][pos]
        if old_gap_i == old_gap_j:
            return None
        
        # Apply swap
        gaps[i][pos] = old_gap_j
        gaps[j][pos] = old_gap_i
        
        # Reconstruct rows from gaps
        new_rows = [list(r) for r in rows]
        for idx in [i, j]:
            new_rows[idx] = [0]
            for g in gaps[idx]:
                new_rows[idx].append(new_rows[idx][-1] + g)
        
        # Validate
        new_scope = full_validate(new_rows)
        if new_scope is not None and new_scope < cur_scope:
            return new_rows, new_scope
        
        # Revert
        gaps[i][pos] = old_gap_i
        gaps[j][pos] = old_gap_j
        return None

    def gap_local_search(rows, budget=2.0):
        """Local search using gap-swap moves."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        if cur_scope is None:
            return None, None
        
        improved = True
        while improved and time.time() - start < budget:
            improved = False
            for _ in range(100):
                if time.time() - start > budget:
                    break
                result = gap_swap_move(cur, None, cur_scope)
                if result is not None:
                    cur, cur_scope = result
                    improved = True
                    break
        
        if cur_scope < get_scope(rows):
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

        # Gap-based local search
        if time.time() < deadline - 12:
            gap_rows, gap_scope = gap_local_search(cur, budget=1.5)
            if gap_rows is not None and gap_scope < best_scope:
                best_scope = gap_scope
                best_rows = gap_rows

        # Targeted descent
        if time.time() < deadline - 10:
            desc_rows, desc_scope = targeted_descent(cur, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

    # Phase 2: Fine-grained SA on single elements
    if time.time() < deadline - 15:
        cur = [r[:] for r in best_rows]
        cur_scope = full_validate(cur)
        if cur_scope is None:
            cur = [r[:] for r in best_rows]
            cur_scope = best_scope

        T = 0.8
        sa_steps = 50000
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

        # Final targeted descent
        if time.time() < deadline - 3:
            desc_rows, desc_scope = targeted_descent(best_rows, budget=2.0)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows

        # Final gap-based polish
        if time.time() < deadline - 2:
            gap_rows, gap_scope = gap_local_search(best_rows, budget=1.5)
            if gap_rows is not None and gap_scope < best_scope:
                best_scope = gap_scope
                best_rows = gap_rows

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
