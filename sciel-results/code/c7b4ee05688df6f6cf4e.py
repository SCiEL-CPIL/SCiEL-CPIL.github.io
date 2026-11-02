"""Difference Triangle Set (4,7) solver.

Structural change: Introduces a backtracking search with dynamic variable ordering
and randomized restarts to refine solutions found by heuristic methods.
The backtracking solver attempts to construct rows under a strict maximum value constraint,
using depth-first search with pruning based on the global difference set.
"""
import time
import random
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)


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


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03, shuffle_rows=False, bias_p=0.0, bias_range=5):
    """Greedy with configurable skip and jump probabilities."""
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
                x += rng.randint(1, 4)
            if rng.random() < skip_p:
                continue
            if bias_p > 0 and rng.random() < bias_p:
                x += rng.randint(1, bias_range)
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


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place."""
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
                    improved = True
                    break
    return work


def _backtrack_find_row(used_diffs, k, max_val, rng, time_limit):
    """
    Finds a valid row of length k+1 starting with 0, with all elements <= max_val,
    such that its differences do not intersect with used_diffs.
    Uses randomized backtracking.
    Returns the row list or None.
    """
    L = k + 1
    row = [0]
    
    # Pre-calculate potential candidates? No, dynamic.
    
    def backtrack(pos):
        if time.time() > time_limit:
            return False
        if pos == L:
            return True
        
        # Heuristic: Try values close to the previous one first, or random?
        # For DTS, smaller values are often better for packing.
        # But we need to avoid collisions.
        
        # Determine range for row[pos]
        lower_bound = row[pos-1] + 1
        upper_bound = max_val
        
        # To improve performance, we can shuffle the candidate range or pick a subset.
        # However, iterating from lower_bound to upper_bound is O(M).
        # We can skip values that are "immediately" invalid?
        
        candidates = list(range(lower_bound, upper_bound + 1))
        # Randomize order to escape local traps, but keep some bias towards smaller values
        # to keep the row compact.
        # We can split candidates: some small, some random.
        # Let's just shuffle for now, it's a structural change from sequential greedy.
        rng.shuffle(candidates)
        
        # Optional: Bias towards smaller values
        # candidates.sort(key=lambda x: (x > row[pos-1] + 5, x)) # Not quite
        
        for val in candidates:
            # Check validity against current row
            valid = True
            new_diffs = set()
            for i in range(pos):
                d = val - row[i]
                if d in new_diffs: # Duplicate within row? No, strictly increasing so diffs are unique within row if values unique?
                    # Actually, a_j - a_i are unique if a_j are distinct?
                    # Yes, if a_j > a_i, the difference is positive.
                    # Can we have a_j - a_i == a_k - a_l?
                    # e.g. 0, 2, 4. 2-0=2, 4-2=2. Duplicate!
                    # So we must check for duplicates within the row too.
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
                
            # Check against global used_diffs
            if new_diffs & used_diffs:
                continue
            
            # If valid, place and recurse
            row[pos] = val
            # We don't update used_diffs yet to keep backtracking clean?
            # No, we should update it to prune future branches.
            # But we need to undo it.
            
            # For simplicity and speed in Python, passing a set copy is expensive.
            # We will update in place and undo.
            
            old_len = len(used_diffs)
            used_diffs.update(new_diffs)
            
            if backtrack(pos + 1):
                return True
            
            # Undo
            for d in new_diffs:
                used_diffs.discard(d)
            
        return False

    return backtrack(1) if backtrack(1) else None

def _backtracking_search(n_rows, k, max_val, rng, time_limit):
    """
    Attempts to find n_rows valid rows with max entry <= max_val.
    Uses backtracking for each row.
    """
    rows = []
    used_diffs = set()
    
    for r in range(n_rows):
        if time.time() > time_limit:
            return None
        
        # Try to find a row
        # We might need to restart this row construction if it fails
        # But _backtrack_find_row does internal backtracking.
        # If it returns None, we failed to find a row for this max_val.
        
        # To increase diversity, we can randomize the "starting point" or strategy.
        # For now, just call it.
        
        # Optimization: If we have previous rows, we can try to "compact" them?
        # No, we are building from scratch for this max_val.
        
        # Add a small random perturbation to the search?
        # The shuffle in _backtrack_find_row provides randomness.
        
        row = _backtrack_find_row(used_diffs, k, max_val, rng, time_limit)
        if row is None:
            return None
        
        rows.append(row)
        # Update used_diffs
        L = k + 1
        for i in range(L):
            for j in range(i + 1, L):
                used_diffs.add(row[j] - row[i])
                
    return rows

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 145 s to leave room for final validation
    time_limit = start_time + 145

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)
    
    # 2) Quick local compaction
    compacted = _local_compact(best_rows, n_rows, k, start_time + 5)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 3) Randomized Greedy Restarts (Fast)
    rng = random.Random(42)
    greedy_deadline = start_time + 20
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.18
        jump_p = rng.random() * 0.10
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.45
        bias_range = rng.randint(3, 10)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax
                # Optional: Compact immediately
                compacted = _local_compact(rows, n_rows, k, start_time + 2)
                if _check_valid(compacted, k):
                    cmax = _max_entry(compacted)
                    if cmax < best_max:
                        best_rows = compacted
                        best_max = cmax

    # 4) Backtracking Search for Improvement
    # Try to find a solution with max_val < best_max
    # We step down by 1 or 2 each time.
    # The backtracking solver is expensive, so we limit its time per attempt.
    
    current_target = best_max - 1
    bt_start_time = time.time()
    bt_time_limit = time_limit - 5 # Leave 5s for final checks
    
    # We can try a few different targets
    # Or just try to beat the current best.
    
    # To make backtracking more effective, we can use the best_rows found so far
    # as a "hint" or initial state? No, it's a decision problem for a specific M.
    
    # Strategy: Binary search or linear search downwards.
    # Since M is around 110-130, linear search from best_max-1 down is feasible
    # if the solver is fast. If it's slow, we might only try a few values.
    
    # We will try to find a solution for M = best_max - 1, then best_max - 2, etc.
    # If we find one, we update best_max and continue.
    
    # To avoid getting stuck in a single hard instance, we add a time limit per M.
    # If it takes too long for one M, we give up on that M and try a lower one?
    # No, if M is too tight, lower M is harder.
    # So if it fails quickly, we stop. If it runs long, it might be close.
    
    # Let's try a randomized approach: pick a random M < best_max and try.
    # Or just iterate downwards.
    
    # Given the complexity, let's limit the backtracking attempts.
    # We'll try to improve best_max by 1, then 2, etc.
    
    # Note: _backtracking_search is a complete search for a given M.
    # It might be very slow.
    # We need a "soft" timeout.
    
    # We will run backtracking in a loop, trying to lower the bound.
    # To prevent hanging on a single M, we'll cap the time per M to, say, 2 seconds.
    # If it doesn't find a solution in 2s, we assume it's too hard for now and move on?
    # No, that's risky.
    
    # Alternative: Use the backtracking solver to *repair* the best solution?
    # No, let's stick to the decision problem.
    
    # To make it faster, we can use the existing best_rows to initialize the "used_diffs"
    # and try to replace one row at a time with a smaller one?
    
    # Let's try a "Row Replacement" backtracking.
    # 1. Take best_rows.
    # 2. Pick a row i.
    # 3. Remove its diffs from used_set.
    # 4. Try to find a new row i with max_val < best_max using backtracking.
    # 5. If found, update best_rows and best_max (if the new row's max is smaller).
    # 6. Compact.
    
    # This is more targeted and likely faster than rebuilding all 4 rows.
    
    def _try_replace_row(rows, r_idx, max_val, rng, time_limit):
        L = k + 1
        used_set = _build_used_set(rows, k)
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(L):
            for j in range(i + 1, L):
                old_diffs.add(old_row[j] - old_row[i])
        
        for d in old_diffs:
            used_set.discard(d)
            
        # Find new row
        new_row = _backtrack_find_row(used_set, k, max_val, rng, time_limit)
        
        if new_row is not None:
            # Add new diffs
            for i in range(L):
                for j in range(i + 1, L):
                    used_set.add(new_row[j] - new_row[i])
            
            rows[r_idx] = new_row
            return True
        else:
            # Restore old row
            for d in old_diffs:
                used_set.add(d)
            return False

    # Loop to improve
    improvement_found = True
    while improvement_found and time.time() < bt_time_limit:
        improvement_found = False
        # Try to reduce the max value
        current_best_max = _max_entry(best_rows)
        target_max = current_best_max - 1
        
        if target_max < 100: # Don't go too low
            break
            
        # Try to replace each row with one that has max <= target_max
        for r_idx in range(n_rows):
            if time.time() > bt_time_limit:
                break
            
            # Work on a copy
            test_rows = [list(r) for r in best_rows]
            
            # Time limit for this specific replacement attempt: 1 second
            attempt_limit = time.time() + 1.0
            
            if _try_replace_row(test_rows, r_idx, target_max, rng, attempt_limit):
                # Validate
                if _check_valid(test_rows, k):
                    new_max = _max_entry(test_rows)
                    if new_max < best_max:
                        best_rows = test_rows
                        best_max = new_max
                        improvement_found = True
                        # Compact
                        compacted = _local_compact(best_rows, n_rows, k, time.time() + 1.0)
                        if _check_valid(compacted, k):
                            cmax = _max_entry(compacted)
                            if cmax < best_max:
                                best_rows = compacted
                                best_max = cmax
                        break # Restart the while loop with new best_max

    # 5) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows

