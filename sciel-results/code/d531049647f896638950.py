import time
import random
import math
import sys

def _row_diffs_list(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    seen = set()
    for row in rows:
        for d in _row_diffs_list(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _compute_all_diffs(rows):
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks

def _column_wise_construct(n_rows, num_elements, rng, max_scope_hint=250):
    """
    Build all rows simultaneously, column by column.
    For each column c (1..num_elements-1), assign a value to each row
    such that all differences remain globally unique.
    We use a greedy approach: for each row, try the smallest value > prev
    that doesn't create a duplicate difference.
    """
    rows = [[0] for _ in range(n_rows)]
    used_mask = 0

    for col in range(1, num_elements):
        # For each row, find the smallest valid value
        # Sort rows by their current last value to process in order
        row_order = list(range(n_rows))
        rng.shuffle(row_order)

        for r in row_order:
            row = rows[r]
            prev = row[-1]
            found = False
            for val in range(prev + 1, max_scope_hint + 1):
                # Check all new differences
                new_mask = 0
                ok = True
                for i in range(len(row)):
                    d = val - row[i]
                    bit = 1 << d
                    if new_mask & bit:
                        ok = False
                        break
                    new_mask |= bit
                if not ok:
                    continue
                if new_mask & used_mask:
                    continue
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            if not found:
                # Fallback: jump
                val = prev + 15 + rng.randint(0, 20)
                row.append(val)
                for i in range(len(row) - 1):
                    d = val - row[i]
                    used_mask |= (1 << d)

    return rows

def _backtrack_row(row_prefix, used_mask, num_elements, max_val, rng, deadline):
    """Complete a row from a prefix with backtracking."""
    if time.time() > deadline:
        return None
    if len(row_prefix) == num_elements:
        return row_prefix

    row = row_prefix
    last = row[-1]

    candidates = []
    for val in range(last + 1, max_val + 1):
        new_diffs = set()
        ok = True
        for prev in row:
            d = val - prev
            if d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if not ok:
            continue
        new_mask = 0
        for d in new_diffs:
            new_mask |= (1 << d)
        if new_mask & used_mask:
            continue
        cost = max(new_diffs) + 0.01 * val
        candidates.append((cost, val, new_mask))

    candidates.sort(key=lambda x: x[0])
    rng.shuffle(candidates[1:])  # slight randomness after best

    for cost, val, new_mask in candidates:
        res = _backtrack_row(row + [val], used_mask | new_mask,
                             num_elements, max_val, rng, deadline)
        if res is not None:
            return res
    return None

def _construct_rows_backtracking(n_rows, num_elements, max_scope, rng, deadline):
    rows = []
    used_mask = 0
    for _ in range(n_rows):
        row = [0]
        res = _backtrack_row(row, used_mask, num_elements, max_scope, rng, deadline)
        if res is None:
            return None
        rows.append(res)
        used_mask |= _row_mask(res)
    return rows

def _column_swap_sa(rows, num_elements, rng, time_limit, initial_temp=20.0):
    """
    SA that swaps entire columns between rows or shifts columns.
    State: the set of rows.
    Move: pick two rows and swap their values at a random pair of columns,
          or shift a column's value up/down by a small amount.
    """
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    cur_scope = best_scope
    gmask, rmasks = _compute_all_diffs(cur)
    n_rows = len(cur)
    n_elem = num_elements
    temp = initial_temp
    cooling = 0.9999
    min_temp = 0.3

    while time.time() < time_limit and temp > min_temp:
        # Choose move type
        move_type = rng.random()

        if move_type < 0.4:
            # Shift a single element
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, n_elem - 1)
            row = cur[row_idx]
            old = row[pos]
            low = row[pos - 1] + 1
            high = row[pos + 1] - 1 if pos < n_elem - 1 else cur_scope + 10
            if low >= high:
                temp *= cooling
                continue
            delta_move = rng.randint(-5, 5)
            new_val = old + delta_move
            if new_val < low or new_val > high:
                temp *= cooling
                continue

            row[pos] = new_val
            diffs = set()
            ok = True
            for i in range(n_elem):
                for j in range(i + 1, n_elem):
                    d = row[j] - row[i]
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                if not ok:
                    break
            if ok:
                new_mask = 0
                for d in diffs:
                    new_mask |= (1 << d)
                other = gmask & ~rmasks[row_idx]
                if new_mask & other:
                    ok = False
            if not ok:
                row[pos] = old
                temp *= cooling
                continue

            old_mask = rmasks[row_idx]
            rmasks[row_idx] = _row_mask(row)
            gmask = (gmask & ~old_mask) | rmasks[row_idx]
            new_scope = _scope(cur)
            delta = new_scope - cur_scope
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                cur_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in cur]
            else:
                row[pos] = old
                rmasks[row_idx] = old_mask
                gmask = (gmask & ~rmasks[row_idx]) | old_mask

        elif move_type < 0.7:
            # Swap two elements between two rows at the same column
            if n_rows < 2:
                temp *= cooling
                continue
            r1, r2 = rng.sample(range(n_rows), 2)
            pos = rng.randint(1, n_elem - 1)
            v1, v2 = cur[r1][pos], cur[r2][pos]
            if v1 == v2:
                temp *= cooling
                continue
            # Try swap
            cur[r1][pos], cur[r2][pos] = v2, v1

            # Validate both rows
            ok = True
            new_masks = []
            for ridx in (r1, r2):
                row = cur[ridx]
                # Check strictly increasing
                for i in range(1, n_elem):
                    if row[i] <= row[i-1]:
                        ok = False
                        break
                if not ok:
                    break
                diffs = set()
                for i in range(n_elem):
                    for j in range(i + 1, n_elem):
                        d = row[j] - row[i]
                        if d in diffs:
                            ok = False
                            break
                        diffs.add(d)
                    if not ok:
                        break
                if not ok:
                    break
                new_masks.append(_row_mask(row))
            if ok:
                combined = new_masks[0] | new_masks[1]
                other_mask = gmask & ~rmasks[r1] & ~rmasks[r2]
                if combined & other_mask:
                    ok = False
            if not ok:
                cur[r1][pos], cur[r2][pos] = v1, v2
                temp *= cooling
                continue

            old_m1, old_m2 = rmasks[r1], rmasks[r2]
            rmasks[r1] = new_masks[0]
            rmasks[r2] = new_masks[1]
            gmask = (gmask & ~old_m1 & ~old_m2) | new_masks[0] | new_masks[1]
            new_scope = _scope(cur)
            delta = new_scope - cur_scope
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                cur_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in cur]
            else:
                cur[r1][pos], cur[r2][pos] = v1, v2
                rmasks[r1] = old_m1
                rmasks[r2] = old_m2
                gmask = (gmask & ~new_masks[0] & ~new_masks[1]) | old_m1 | old_m2

        else:
            # Rebuild one row via backtracking at a tighter scope
            ridx = rng.randint(0, n_rows - 1)
            target_scope = cur[ridx][-1] - rng.randint(1, 3)
            if target_scope < n_elem:
                target_scope = n_elem
            other_mask = gmask & ~rmasks[ridx]
            res = _backtrack_row([0], other_mask, n_elem, target_scope, rng,
                                 time.time() + 3)
            if res is not None:
                new_mask = _row_mask(res)
                if not (new_mask & other_mask):
                    old_mask = rmasks[ridx]
                    cur[ridx] = res
                    rmasks[ridx] = new_mask
                    gmask = (gmask & ~old_mask) | new_mask
                    new_scope = _scope(cur)
                    delta = new_scope - cur_scope
                    if delta <= 0 or rng.random() < math.exp(-delta / temp):
                        cur_scope = new_scope
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best = [r[:] for r in cur]
                    else:
                        cur[ridx] = [0] + [0] * (n_elem - 1)  # placeholder, will be fixed
                        # Actually revert properly
                        cur[ridx] = [0]
                        # We lost the old row, so just revert masks and skip
                        rmasks[ridx] = old_mask
                        gmask = (gmask & ~new_mask) | old_mask

        temp *= cooling

    return best, best_scope

def _local_search_reduce(rows, num_elements, rng, time_limit):
    """Fast local search: try to move elements downward."""
    start = time.time()
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    best_scope = _scope(cur)
    cur_scope = best_scope
    gmask, rmasks = _compute_all_diffs(cur)
    n_rows = len(cur)
    n_elem = num_elements
    stagnation = 0
    max_stag = 30000

    while time.time() < time_limit and stagnation < max_stag:
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = cur[row_idx]
        old = row[pos]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else cur_scope
        if low >= old:
            continue
        new_val = rng.randint(low, old - 1)
        if new_val == old:
            continue

        row[pos] = new_val
        diffs = set()
        ok = True
        for i in range(n_elem):
            for j in range(i + 1, n_elem):
                d = row[j] - row[i]
                if d in diffs:
                    ok = False
                    break
                diffs.add(d)
            if not ok:
                break
        if ok:
            new_mask = 0
            for d in diffs:
                new_mask |= (1 << d)
            other = gmask & ~rmasks[row_idx]
            if new_mask & other:
                ok = False
        if not ok:
            row[pos] = old
            stagnation += 1
            continue

        old_mask = rmasks[row_idx]
        rmasks[row_idx] = _row_mask(row)
        gmask = (gmask & ~old_mask) | rmasks[row_idx]
        new_scope = _scope(cur)
        if new_scope < best_scope:
            best_scope = new_scope
            best = [r[:] for r in cur]
            stagnation = 0
        else:
            row[pos] = old
            rmasks[row_idx] = old_mask
            gmask = (gmask & ~rmasks[row_idx]) | old_mask
            stagnation += 1
        cur_scope = _scope(cur)

    return best, best_scope

def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """Greedy: build rows one by one, smallest valid value per position."""
    rows = []
    used_mask = 0
    high = max_scope_hint if max_scope_hint else 250

    for _ in range(n_rows):
        row = [0]
        for pos in range(1, num_elements):
            found = False
            for val in range(row[-1] + 1, high + 1):
                new_diffs = set(val - prev for prev in row)
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                new_mask = 0
                for d in new_diffs:
                    new_mask |= (1 << d)
                if new_mask & used_mask:
                    continue
                row.append(val)
                used_mask |= new_mask
                found = True
                break
            if not found:
                val = row[-1] + 20 + rng.randint(0, 30)
                row.append(val)
                for d in (val - prev for prev in row[:-1]):
                    used_mask |= (1 << d)
        rows.append(row)
    return rows

def _construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    # Phase 1: Column-wise simultaneous construction
    for seed in range(10):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 13 + 7)
        rows = _column_wise_construct(n_rows, num_elements, rng_local,
                                      max_scope_hint=counting_bound + 40)
        if _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows

    # Phase 2: Row-by-row backtracking
    if time.time() < DEADLINE - 60:
        for seed in range(8):
            if time.time() > DEADLINE - 55:
                break
            rng_local = random.Random(seed * 17 + 3)
            start_scope = min(220, counting_bound + 50)
            for scope in range(start_scope, max(start_scope - 35, counting_bound - 5), -1):
                if time.time() > DEADLINE - 50:
                    break
                rows = _construct_rows_backtracking(n_rows, num_elements, scope,
                                                    rng_local, DEADLINE - 45)
                if rows is not None and _is_valid(rows):
                    s = _scope(rows)
                    if s < best_scope:
                        best_scope = s
                        best_rows = rows
                    break
            if best_rows is not None and best_scope <= counting_bound + 10:
                break

    # Phase 3: Greedy fallback
    if best_rows is None or best_scope > counting_bound + 20:
        for seed in range(8):
            if time.time() > DEADLINE - 40:
                break
            rng_local = random.Random(seed * 11 + 5)
            rows = _greedy_construct(n_rows, num_elements, rng_local)
            if _is_valid(rows):
                s = _scope(rows)
                if s < best_scope:
                    best_scope = s
                    best_rows = rows

    # Fallback sequential
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)

    # Phase 4: Column-swap SA (the new core method)
    if time.time() < DEADLINE - 35:
        improved, imp_scope = _column_swap_sa(
            best_rows, num_elements, rng, DEADLINE - 30,
            initial_temp=max(10.0, best_scope * 0.15))
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 5: Local search polish
    if time.time() < DEADLINE - 15:
        improved, imp_scope = _local_search_reduce(
            best_rows, num_elements, rng, DEADLINE - 10)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Phase 6: Another SA round if time
    if time.time() < DEADLINE - 8 and best_scope > counting_bound:
        improved, imp_scope = _column_swap_sa(
            best_rows, num_elements, rng, DEADLINE - 5,
            initial_temp=max(5.0, best_scope * 0.08))
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
