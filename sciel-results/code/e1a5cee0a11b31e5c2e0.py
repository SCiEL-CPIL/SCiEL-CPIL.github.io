import time
import random
import math

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

    def row_diffs_bitmask(row):
        """Return a bitmask of all positive differences in row."""
        mask = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diff = row[j] - ri
                mask |= (1 << diff)
        return mask

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

    rng = random.Random(12345)

    def rebuild_row_fast(used_mask, max_val, jitter=0.0, local_rng=None):
        """Rebuild a single row using bitmask for used differences.
        Returns the row list or None if failed."""
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_mask = 0
        x = 0
        while len(row) < k + 1:
            found = False
            for trial_x in range(x + 1, max_val + 1):
                new_mask = 0
                for a in row:
                    new_mask |= (1 << (trial_x - a))
                if (new_mask & used_mask) == 0 and (new_mask & row_mask) == 0:
                    if jitter > 0 and local_rng.random() < jitter:
                        continue
                    row.append(trial_x)
                    row_mask |= new_mask
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def rebuild_row_set(used_set, max_val, jitter=0.0, local_rng=None):
        """Rebuild a single row using set for used differences."""
        if local_rng is None:
            local_rng = rng
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < k + 1:
            found = False
            for trial_x in range(x + 1, max_val + 1):
                new = {trial_x - a for a in row}
                if not (new & used_set) and not (new & row_diffs):
                    if jitter > 0 and local_rng.random() < jitter:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row

    def local_polish(rows, max_passes=8):
        """Reduce scope by locally re-optimizing each row."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 15 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 15:
                    break
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        for a in range(len(rows[j])):
                            for b in range(a + 1, len(rows[j])):
                                used_other.add(rows[j][b] - rows[j][a])
                best_row = list(rows[i])
                best_max = max(rows[i])
                for attempt in range(30):
                    jitter = 0.0 if attempt < 10 else (0.05 + 0.04 * (attempt - 10))
                    new_row = rebuild_row_set(used_other, best_max - 1, jitter=jitter)
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
                    for a in range(len(rows[j])):
                        for b in range(a + 1, len(rows[j])):
                            used_other.add(rows[j][b] - rows[j][a])
            success = False
            for attempt in range(15):
                jitter = 0.3 + 0.1 * attempt
                new_row = rebuild_row_set(used_other, best_scope + 30, jitter=jitter)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row_set(used_other, best_scope + 40, jitter=0.0)
                if new_row is not None:
                    rows[idx] = new_row
        return rows

    def swap_rows(rows, i, j):
        """Swap two rows in the list."""
        rows[i], rows[j] = rows[j], rows[i]

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
                # Try a middle position
                pos2 = rng.randint(1, len(row) - 2)
                old2 = row[pos2]
                lo2 = row[pos2 - 1] + 1
                hi2 = row[pos2 + 1] - 1
                if hi2 <= lo2:
                    continue
                for _ in range(10):
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
            for v in range(lower, min(lower + 25, old_val)):
                candidates.add(v)
            for _ in range(15):
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

    def mark_anneal(rows, cur_scope, budget=5.0):
        """Fine-grained simulated annealing on individual marks."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_sc = cur_scope
        best_local = [r[:] for r in rows]
        best_local_sc = cur_scope
        T = 0.8
        while time.time() - start < budget:
            ri = rng.randrange(n_rows)
            pos = rng.randint(1, len(cur[ri]) - 1)
            old_val = cur[ri][pos]
            lo = cur[ri][pos - 1] + 1
            hi = cur[ri][pos + 1] - 1 if pos < len(cur[ri]) - 1 else best_scope + 25
            if hi <= lo:
                continue
            new_val = rng.randint(lo, hi)
            if new_val == old_val:
                continue
            cur[ri][pos] = new_val
            new_scope = full_validate(cur)
            if new_scope is not None:
                if new_scope < best_local_sc:
                    best_local_sc = new_scope
                    best_local = [r[:] for r in cur]
                delta = new_scope - cur_sc
                if delta <= 0 or rng.random() < (2.718281828 ** (-delta / max(T, 0.01))):
                    cur_sc = new_scope
                else:
                    cur[ri][pos] = old_val
            else:
                cur[ri][pos] = old_val
            T *= 0.9998
        return best_local, best_local_sc

    def row_swap_search(rows, budget=4.0):
        """Try swapping row order to enable smaller scopes."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        best = [r[:] for r in cur]
        best_sc = cur_scope
        while time.time() - start < budget:
            i = rng.randrange(n_rows)
            j = rng.randrange(n_rows)
            if i == j:
                continue
            cur[i], cur[j] = cur[j], cur[i]
            # Try to polish after swap
            polished = local_polish(cur, max_passes=2)
            ps = full_validate(polished)
            if ps is not None and ps < best_sc:
                best_sc = ps
                best = [r[:] for r in polished]
            else:
                cur[i], cur[j] = cur[j], cur[i]
        return best, best_sc

    # Phase 1: ILS with row reconstruction + local polish
    max_il_iter = 300
    for it in range(max_il_iter):
        if time.time() > deadline - 20:
            break

        if it == 0:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            if rng.random() < 0.7:
                cur = kick(cur, num_kicks=rng.choice([1, 1, 2]))
            else:
                # Try a row swap perturbation
                i = rng.randrange(n_rows)
                j = rng.randrange(n_rows)
                if i != j:
                    cur[i], cur[j] = cur[j], cur[i]

        cur_scope = full_validate(cur)
        if cur_scope is None:
            continue

        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [r[:] for r in cur]

        # Local polish
        if time.time() < deadline - 18:
            polished = local_polish(cur)
            ps = full_validate(polished)
            if ps is not None and ps < best_scope:
                best_scope = ps
                best_rows = [r[:] for r in polished]
                cur = [r[:] for r in polished]

        # Targeted descent
        if time.time() < deadline - 12:
            desc_rows, desc_scope = targeted_descent(cur, budget=1.5)
            if desc_rows is not None and desc_scope < best_scope:
                best_scope = desc_scope
                best_rows = desc_rows
                cur = [r[:] for r in desc_rows]

    # Phase 2: Row swap search on the best solution
    if time.time() < deadline - 12:
        swap_rows_best, swap_scope = row_swap_search(best_rows, budget=5.0)
        if swap_scope < best_scope:
            best_scope = swap_scope
            best_rows = swap_rows_best

    # Phase 3: Fine-grained mark annealing
    if time.time() < deadline - 8:
        anneal_rows, anneal_scope = mark_anneal(best_rows, best_scope, budget=6.0)
        if anneal_scope < best_scope:
            best_scope = anneal_scope
            best_rows = anneal_rows

    # Phase 4: Final targeted descent
    if time.time() < deadline - 5:
        desc_rows, desc_scope = targeted_descent(best_rows, budget=3.0)
        if desc_rows is not None and desc_scope < best_scope:
            best_scope = desc_scope
            best_rows = desc_rows

    # Phase 5: One more polish pass
    if time.time() < deadline - 3:
        polished = local_polish(best_rows, max_passes=4)
        ps = full_validate(polished)
        if ps is not None and ps < best_scope:
            best_scope = ps
            best_rows = polished

    if best_rows is None:
        best_rows = greedy_init()

    return best_rows
