"""Combined program: row-reconstruction search (inspiration 1's structural idea)
with incremental diff-set maintenance and multi-phase optimization (current program's machinery).
Rebuilds entire rows under tighter scope constraints, using the diff-set of other rows
as the constraint. This explores a fundamentally different neighborhood than point mutations."""
import time
import random

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0

    row_len = k + 1
    d = k * (k + 1) // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def full_validate(rows):
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

    def greedy_init(rng, rand_prob=0.1):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                if rand_prob > 0 and rng.random() < rand_prob and x > 10:
                    x += rng.randint(1, 3)
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def rebuild_row(used_other, max_val, rng, jitter=0.0):
        """Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed."""
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            for trial_x in range(x + 1, max_val + 1):
                if jitter > 0 and rng.random() < jitter and trial_x < max_val - 5:
                    continue
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    row.append(trial_x)
                    row_diffs |= new
                    x = trial_x
                    found = True
                    break
            if not found:
                return None
        return row

    def local_polish(rows, time_budget):
        """Iteratively rebuild each row with tighter scope constraints."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        improved = True

        while improved and time.time() - start < time_budget:
            improved = False
            # Process rows in order of largest max first
            order = sorted(range(n_rows), key=lambda i: -max(cur[i]))
            for i in order:
                if time.time() - start >= time_budget:
                    break
                # Compute diffs used by all other rows
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        used_other |= row_diffs_set(cur[j])
                
                # Try to rebuild row i with progressively tighter max
                best_row = cur[i][:]
                best_max = max(cur[i])
                
                # Try from best_max-1 down to best_max - 50
                for target_max in range(best_max - 1, max(best_max - 50, 10), -1):
                    new_row = rebuild_row(used_other, target_max, rng, jitter=0.0)
                    if new_row is not None:
                        best_row = new_row
                        best_max = target_max
                        improved = True
                        break
                
                if best_row != cur[i]:
                    cur[i] = best_row
                    cur_scope = get_scope(cur)
        
        return cur, cur_scope

    def targeted_descent(rows, time_budget, rng):
        """Targeted descent: find global max, try to reduce it via row rebuild."""
        start = time.time()
        cur = [r[:] for r in rows]
        cur_scope = get_scope(cur)
        best_cur = [r[:] for r in cur]
        best_scope = cur_scope

        while time.time() - start < time_budget:
            # Find global max
            max_val = -1
            max_r = -1
            for ri, r in enumerate(cur):
                if max(r) > max_val:
                    max_val = max(r)
                    max_r = ri
            
            if max_r < 0:
                break

            row = cur[max_r]
            pos = len(row) - 1  # Last element is the max in a sorted row
            old_val = row[pos]
            lower = row[pos - 1] + 1 if pos > 0 else 1

            if old_val <= lower:
                break

            # Compute diffs used by all other rows
            used_other = set()
            for j in range(n_rows):
                if j != max_r:
                    used_other |= row_diffs_set(cur[j])
            
            # Try to rebuild the row with smaller max
            found = False
            for target_max in range(old_val - 1, lower, -1):
                if time.time() - start >= time_budget:
                    break
                new_row = rebuild_row(used_other, target_max, rng, jitter=0.0)
                if new_row is not None:
                    cur[max_r] = new_row
                    cur_scope = get_scope(cur)
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_cur = [r[:] for r in cur]
                    found = True
                    break
            
            if not found:
                # Try perturbing a different row
                ri = rng.randrange(n_rows)
                if ri == max_r:
                    continue
                used_other2 = set()
                for j in range(n_rows):
                    if j != ri:
                        used_other2 |= row_diffs_set(cur[j])
                old_row_max = max(cur[ri])
                for target_max in range(old_row_max - 1, max(old_row_max - 30, 10), -1):
                    if time.time() - start >= time_budget:
                        break
                    new_row = rebuild_row(used_other2, target_max, rng, jitter=0.0)
                    if new_row is not None:
                        cur[ri] = new_row
                        cur_scope = get_scope(cur)
                        if cur_scope < best_scope:
                            best_scope = cur_scope
                            best_cur = [r[:] for r in cur]
                        break
        
        return best_cur, best_scope

    def kick(rows, num_kicks, rng):
        """Perturb by rebuilding random rows with jitter to escape local minima."""
        cur = [r[:] for r in rows]
        indices = rng.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    used_other |= row_diffs_set(cur[j])
            old_max = max(cur[idx])
            success = False
            for attempt in range(10):
                jitter = 0.15 + 0.05 * attempt
                search_max = old_max + 10 + attempt * 5
                new_row = rebuild_row(used_other, search_max, rng, jitter=jitter)
                if new_row is not None:
                    cur[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, old_max + 30, rng, jitter=0.0)
                if new_row is not None:
                    cur[idx] = new_row
        return cur

    rng = random.Random(42)
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Initial greedy construction with multiple random seeds
    for init_try in range(5):
        if time.time() > deadline - 30:
            break
        rows = greedy_init(rng, rand_prob=0.05 + 0.02 * init_try)
        scope = full_validate(rows)
        if scope is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    if best_rows is None:
        best_rows = greedy_init(rng)
        best_scope = get_scope(best_rows)

    # Phase 2: Local polish on best solution
    if time.time() < deadline - 20:
        polished, scope = local_polish(best_rows, min(15.0, deadline - time.time() - 15))
        if scope < best_scope:
            best_scope = scope
            best_rows = polished

    # Phase 3: ILS with targeted descent and kicks
    n_iter = 0
    while time.time() < deadline - 10:
        n_iter += 1
        # Create candidate
        if n_iter <= 3:
            cur = [r[:] for r in best_rows]
        else:
            cur = [r[:] for r in best_rows]
            num_kicks = rng.choice([1, 1, 1, 2, 2, 3])
            cur = kick(cur, num_kicks, rng)
        
        scope = full_validate(cur)
        if scope is None:
            continue
        
        # Targeted descent
        remaining = deadline - time.time() - 5
        if remaining > 2:
            cur, scope = targeted_descent(cur, min(remaining, 8.0), rng)
        
        # Local polish
        remaining = deadline - time.time() - 5
        if remaining > 3:
            cur, scope = local_polish(cur, min(remaining, 5.0))
        
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in cur]

    return best_rows
