"""Difference Triangle Set (4,7) solver.

New structural approach:
- Represent the problem as a constraint satisfaction problem over the difference set.
- Use backtracking search with dynamic variable ordering (MRV) that selects the next
  element to place based on the most constrained available difference slots.
- This shifts the search from optimizing row values directly to satisfying the global
  uniqueness constraint on differences.
- Combine with local search (simulated annealing) for refinement.
"""
import time
import random
import numpy as np


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction with optional randomization."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03, shuffle_rows=False):
    """Greedy with configurable skip and jump probabilities, optional row-order shuffle."""
    used = set()
    rows = []
    order = list(range(n_rows))
    if shuffle_rows:
        rng.shuffle(order)
    row_results = {}
    for r_idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < jump_p:
                x += rng.randint(1, 3)
            if rng.random() < skip_p:
                continue
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        if len(row) != k + 1:
            return None
        row_results[r_idx] = row
    for r_idx in range(n_rows):
        rows.append(row_results[r_idx])
    return rows


def _build_used_set(rows, k):
    L = k + 1
    used = set()
    for row in rows:
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _get_diffs_for_pos(row, pos, L):
    diffs = set()
    for i in range(L):
        if i < pos:
            diffs.add(row[pos] - row[i])
        elif i > pos:
            diffs.add(row[i] - row[pos])
    return diffs


def _can_set(row, pos, new_val, used_set, L):
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    other = used_set - old_diffs
    if new_diffs & other:
        return False
    return True


def _apply_set(row, pos, new_val, used_set, L):
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    for d in old_diffs:
        used_set.discard(d)
    for d in new_diffs:
        used_set.add(d)
    row[pos] = new_val


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place, return True if any change made."""
    changed = False
    for pos in range(L - 1, 0, -1):
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        old_diffs = _get_diffs_for_pos(row, pos, L)
        for new_val in range(lower, old_val):
            if pos < L - 1 and new_val >= row[pos + 1]:
                break
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            other = used_set - old_diffs
            if new_diffs & other:
                continue
            for d in old_diffs:
                used_set.discard(d)
            for d in new_diffs:
                used_set.add(d)
            row[pos] = new_val
            changed = True
            break
    return changed


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work, k)
    L = k + 1
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
                old_diffs = _get_diffs_for_pos(row, pos, L)
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    return work


def _backtracking_mrv(n_rows, k, max_scope, rng, time_limit):
    """
    Backtracking search with MRV (Minimum Remaining Values) heuristic.
    
    We model this as: for each row, we need to place k+1 elements (starting with 0).
    The constraint is that all pairwise differences across all rows must be unique.
    
    MRV: at each step, choose the next position to fill that has the fewest valid
    candidate values given the current partial assignment.
    """
    L = k + 1
    used_diffs = set()
    rows = [[0] for _ in range(n_rows)]
    
    # Track which positions are filled for each row
    filled = [0] * n_rows  # number of elements placed in each row
    
    best_solution = None
    best_max = max_scope + 1
    
    def backtrack(depth, current_max):
        nonlocal best_solution, best_max
        
        if time.time() > time_limit:
            return False
        
        # Check if we're done
        if all(filled[r] == L for r in range(n_rows)):
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(row) for row in rows]
            return True
        
        # MRV: find the row with the fewest valid next values
        # We prioritize rows that are more constrained
        candidates = []
        for r in range(n_rows):
            if filled[r] >= L:
                continue
            pos = filled[r]
            row = rows[r]
            
            # Find valid values for this position
            lower_bound = row[-1] + 1 if pos > 0 else 1
            valid_values = []
            
            # Try values from lower_bound up to some upper bound
            # Upper bound: we want to keep scope small
            upper_bound = min(current_max, max_scope)
            
            # For efficiency, limit how many values we check
            max_check = min(upper_bound - lower_bound + 1, 50)
            
            for v in range(lower_bound, lower_bound + max_check):
                if v > max_scope:
                    break
                new_diffs = set()
                for i in range(len(row)):
                    new_diffs.add(v - row[i])
                if not (new_diffs & used_diffs):
                    valid_values.append(v)
                    if len(valid_values) >= 5:  # Early termination for MRV
                        break
            
            if valid_values:
                candidates.append((len(valid_values), r, valid_values))
        
        if not candidates:
            return False
        
        # Sort by number of valid values (MRV heuristic)
        candidates.sort(key=lambda x: x[0])
        
        # Try the most constrained variable first
        _, r, valid_values = candidates[0]
        
        # Shuffle valid values for randomness
        rng.shuffle(valid_values)
        
        for v in valid_values:
            row = rows[r]
            new_diffs = set()
            for i in range(len(row)):
                new_diffs.add(v - row[i])
            
            # Check if adding these diffs is valid
            if new_diffs & used_diffs:
                continue
            
            # Apply
            used_diffs.update(new_diffs)
            row.append(v)
            filled[r] += 1
            
            new_current_max = max(current_max, v)
            
            if backtrack(depth + 1, new_current_max):
                return True
            
            # Undo
            for d in new_diffs:
                used_diffs.discard(d)
            row.pop()
            filled[r] -= 1
        
        return False
    
    # Try multiple random restarts with different max_scope bounds
    for attempt in range(20):
        if time.time() > time_limit:
            break
        
        # Try to find a solution with scope <= max_scope
        # Start with a reasonable bound and decrease
        scope_bound = max_scope - attempt * 2
        if scope_bound < 100:
            break
        
        # Reset state
        used_diffs.clear()
        rows = [[0] for _ in range(n_rows)]
        filled = [0] * n_rows
        
        if backtrack(0, 0):
            break
    
    return best_solution if best_solution else None


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 8)
    end_temp = 0.1
    cooling = 0.9996

    t = start_temp

    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()

        if move_type < 0.4:
            # Move 1: Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                t *= cooling
                continue
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]

        elif move_type < 0.7:
            # Move 2: Increase a random element (perturbation), then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(1, current_max // 4))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
            else:
                new_val = old_val + 1
                if _can_set(row, pos, new_val, used_set, L):
                    _apply_set(row, pos, new_val, used_set, L)
                    _fast_compact_row(row, used_set, L)
                    new_max = _max_entry(work)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in work]

        elif move_type < 0.85:
            # Move 3: Rebuild a random row
            r_idx = rng.randint(0, n_rows - 1)
            _rebuild_row(r_idx, work, used_set, L, rng, current_max)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max

        else:
            # Move 4: Full recompact of all rows
            for r_idx in range(n_rows):
                _fast_compact_row(work[r_idx], used_set, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max

        t *= cooling

    return best_rows


def _rebuild_row(r_idx, work, used_set, L, rng, max_val):
    """Rebuild a row from scratch, trying to keep its max below max_val."""
    row = work[r_idx]
    old_diffs = set()
    for i in range(L):
        for j in range(i + 1, L):
            old_diffs.add(row[j] - row[i])
    for d in old_diffs:
        used_set.discard(d)

    best_new_row = None
    best_new_max = max_val + 1

    for attempt in range(50):
        new_row = [0]
        x = 0
        valid = True
        while len(new_row) < L:
            x += 1
            if rng.random() < 0.03:
                x += rng.randint(1, 2)
            new_diffs = {x - a for a in new_row}
            if new_diffs & used_set:
                continue
            new_row.append(x)
            if x > best_new_max:
                valid = False
                break
        if valid and len(new_row) == L:
            new_max = new_row[-1]
            if new_max < best_new_max:
                best_new_max = new_max
                best_new_row = list(new_row)

    if best_new_row is not None:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(best_new_row[j] - best_new_row[i])
        work[r_idx] = best_new_row
        return True
    else:
        for d in old_diffs:
            used_set.add(d)
        work[r_idx] = row
        return False


def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 135 s to leave room for final validation
    time_limit = start_time + 135

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts with varied parameters
    rng = random.Random(42)
    greedy_deadline = time_limit - 60
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.12
        jump_p = rng.random() * 0.05
        shuffle = rng.random() < 0.5
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 30)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 4) Backtracking with MRV for structural improvement
    if time.time() < time_limit - 10:
        bt_result = _backtracking_mrv(n_rows, k, best_max, rng, time_limit - 10)
        if bt_result is not None and _check_valid(bt_result, k):
            bmax = _max_entry(bt_result)
            if bmax < best_max:
                best_rows = bt_result
                best_max = bmax

    # 5) Simulated annealing with remaining time
    if time.time() < time_limit - 5:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 5)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
