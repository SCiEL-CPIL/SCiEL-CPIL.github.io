"""Difference Triangle Set (4,7) solver.

Strategy:
1. Fast greedy baseline (deterministic + a few randomized variants).
2. Efficient local compaction using NumPy for vectorized difference checks.
3. Time-bounded simulated annealing with:
   - Single-element decrease moves (compaction)
   - Random element perturbations (escape local minima)
   - Full-row recompact after perturbations
4. Safety margin before the 150 s wall-clock limit.
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


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03):
    """Greedy with configurable skip and jump probabilities."""
    used = set()
    rows = []
    for _ in range(n_rows):
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
        rows.append(row)
    return rows


def _build_diff_array(rows):
    """Return a sorted numpy array of all positive differences."""
    diffs = []
    for row in rows:
        L = len(row)
        arr = np.array(row)
        for i in range(L):
            for j in range(i + 1, L):
                diffs.append(row[j] - row[i])
    return np.array(sorted(diffs))


def _is_diff_free(new_diffs, used_set):
    """Check if new_diffs (list of ints) are all not in used_set."""
    for d in new_diffs:
        if d in used_set:
            return False
    return True


def _local_compact(rows, n_rows, k, rng, time_limit):
    """Iteratively compact each element to its smallest feasible value.
    
    Uses a hybrid approach: for each position, scan upward from the minimum
    feasible value and use a set-based check for speed.
    """
    work = [list(r) for r in rows]
    used_set = set()
    for row in work:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(row[j] - row[i])
    
    L = k + 1
    improved = True
    passes = 0
    
    while improved and time.time() < time_limit:
        improved = False
        passes += 1
        # Process from right to left for compaction
        for r_idx in range(n_rows):
            row = work[r_idx]
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                
                # Collect old diffs involving pos
                old_diffs = set()
                for i in range(L):
                    if i < pos:
                        old_diffs.add(row[pos] - row[i])
                    elif i > pos:
                        old_diffs.add(row[i] - row[pos])
                
                # Try each candidate from lowest
                for new_val in range(lower, old_val):
                    # Check increasing
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    # Collect new diffs
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    # Check no collision with other diffs
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    # Apply
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    
    return work


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place, return True if any change made."""
    changed = False
    for pos in range(L - 1, 0, -1):
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        
        old_diffs = set()
        for i in range(L):
            if i < pos:
                old_diffs.add(row[pos] - row[i])
            elif i > pos:
                old_diffs.add(row[i] - row[pos])
        
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


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution.
    
    Moves:
    1. Decrease a random element to a smaller feasible value.
    2. Increase a random element (to escape local minima), then recompact.
    3. Rebuild a random row from scratch.
    4. Swap two elements between rows if feasible.
    """
    work = [list(r) for r in rows]
    L = k + 1
    
    def build_used():
        s = set()
        for row in work:
            for i in range(L):
                for j in range(i + 1, L):
                    s.add(row[j] - row[i])
        return s
    
    def get_diffs_for_pos(r_idx, pos):
        row = work[r_idx]
        diffs = set()
        for i in range(L):
            if i < pos:
                diffs.add(row[pos] - row[i])
            elif i > pos:
                diffs.add(row[i] - row[pos])
        return diffs
    
    def can_set(r_idx, pos, new_val, used_set):
        row = work[r_idx]
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < L - 1 and new_val >= row[pos + 1]:
            return False
        old_diffs = get_diffs_for_pos(r_idx, pos)
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
    
    def apply_set(r_idx, pos, new_val, used_set):
        row = work[r_idx]
        old_diffs = get_diffs_for_pos(r_idx, pos)
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
    
    def recompact_row(r_idx, used_set):
        """Recompact a single row from right to left."""
        row = work[r_idx]
        for pos in range(L - 1, 0, -1):
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            old_diffs = get_diffs_for_pos(r_idx, pos)
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
                # Apply
                for d in old_diffs:
                    used_set.discard(d)
                for d in new_diffs:
                    used_set.add(d)
                row[pos] = new_val
                break
    
    def rebuild_row(r_idx, used_set):
        """Rebuild a row from scratch, trying to keep it small."""
        row = work[r_idx]
        # Remove all diffs from this row
        old_diffs = set()
        for i in range(L):
            for j in range(i + 1, L):
                old_diffs.add(row[j] - row[i])
        for d in old_diffs:
            used_set.discard(d)
        
        # Rebuild
        new_row = [0]
        x = 0
        while len(new_row) < L:
            x += 1
            new_diffs = {x - a for a in new_row}
            if not (new_diffs & used_set):
                new_row.append(x)
                used_set |= new_diffs
        
        if len(new_row) == L:
            work[r_idx] = new_row
            return True
        else:
            # Restore old row
            work[r_idx] = row
            used_set |= old_diffs
            return False
    
    used_set = build_used()
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max
    
    start_temp = max(1, current_max // 10)
    end_temp = 0.1
    cooling = 0.9997
    
    t = start_temp
    step = 0
    
    while time.time() < time_limit and t > end_temp:
        step += 1
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Move 1: Try to decrease a random element
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                continue
            # Try a random smaller value
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                continue
            if can_set(r_idx, pos, new_val, used_set):
                apply_set(r_idx, pos, new_val, used_set)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
            # Recompact this row
            recompact_row(r_idx, used_set)
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
                continue
            if can_set(r_idx, pos, new_val, used_set):
                apply_set(r_idx, pos, new_val, used_set)
                recompact_row(r_idx, used_set)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
            else:
                new_val = old_val + 1
                if can_set(r_idx, pos, new_val, used_set):
                    apply_set(r_idx, pos, new_val, used_set)
                    recompact_row(r_idx, used_set)
                    new_max = _max_entry(work)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in work]
        elif move_type < 0.85:
            # Move 3: Rebuild a random row
            r_idx = rng.randint(0, n_rows - 1)
            rebuild_row(r_idx, used_set)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
        else:
            # Move 4: Full recompact of all rows
            for r_idx in range(n_rows):
                recompact_row(r_idx, used_set)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
        
        # Cooling
        t *= cooling
    
    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 135 s to leave room for final validation
    time_limit = start_time + 135

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts with varied parameters
    rng = random.Random(42)
    greedy_deadline = time_limit - 80
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.12
        jump_p = rng.random() * 0.05
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, rng, time_limit - 40)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 4) Simulated annealing with remaining time
    if time.time() < time_limit - 5:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 5)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 5) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
