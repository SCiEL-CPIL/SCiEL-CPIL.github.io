import time
import random
import sys
import math
from collections import Counter

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows, k=None):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        if k is not None and len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, skip_prob=0.0, rng=None):
    """
    Greedy initialization with optional random skipping.
    """
    if rng is None:
        rng = random
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if skip_prob > 0 and rng.random() < skip_prob and x < 300:
                    skipped = False
                    for extra in range(rng.randint(1, 4)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                counts[d] += 1
    return counts

def _remove_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]

def _add_row_diffs(diff_counts, row):
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            diff_counts[d] = diff_counts.get(d, 0) + 1

def _row_rebuild(row, k, diff_counts, rng, max_scope):
    """
    Rebuild a row to minimize its max element, subject to diff_counts constraints.
    Returns the new row or None if not possible within max_scope.
    """
    best_row = None
    best_score = float('inf')
    
    # Try a few random restarts for the rebuild
    for attempt in range(15):
        new_row = [0]
        x = 0
        local_diffs = set()
        ok = True
        for j in range(1, k + 1):
            found = False
            start = x + 1
            if attempt > 0 and rng.random() < 0.5:
                start = x + rng.randint(1, 3)
            
            # Look for the smallest valid t
            # We prioritize smaller t to keep scope low
            candidates = []
            # Limit search window to speed up, but ensure we find a solution if one exists close by
            search_limit = min(max_scope + 1, start + 20 + attempt * 5)
            
            for t in range(start, search_limit):
                conflicts = 0
                new_diffs = []
                for a in new_row:
                    d = t - a
                    if d <= 0:
                        continue
                    new_diffs.append(d)
                    if diff_counts.get(d, 0) > 0:
                        conflicts += 1
                if conflicts == 0:
                    candidates.append((0, t, new_diffs))
                elif conflicts <= 1 and attempt > 5:
                    # Allow 1 conflict in later attempts to escape local minima? 
                    # Actually, for rebuild, we want 0 conflicts to be valid.
                    # But we are rebuilding one row while others are fixed.
                    # If we allow conflicts, the global state becomes invalid.
                    # So we must have 0 conflicts with the *other* rows.
                    # diff_counts contains counts from *other* rows (since we removed this row).
                    # So conflicts must be 0.
                    pass
            
            if candidates:
                # Pick the smallest t with 0 conflicts
                # candidates are sorted by (conflicts, t). We only added 0 conflicts.
                # So we just pick the first one (smallest t).
                _, t, new_diffs = candidates[0]
                new_row.append(t)
                local_diffs.update(new_diffs)
                x = t
                found = True
            else:
                # No valid t in window. Try a larger jump?
                # If we can't find one, this attempt fails.
                ok = False
                break
        
        if ok and len(new_row) == k + 1:
            # Verify it's within max_scope
            if max(new_row) <= max_scope:
                score = max(new_row)
                if score < best_score:
                    best_score = score
                    best_row = list(new_row)
                    if best_score == max_scope - 1: # Can't do much better than max_scope-1 if we are trying to reduce
                        break
                    if best_score <= max_scope - 5: # Good enough
                        break
    
    return best_row

def _sa_optimize(rows, n_rows, k, deadline, rng=None):
    """
    Optimized Simulated Annealing.
    """
    if rng is None:
        rng = random
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    current_scope = best_scope
    
    diff_counts = _build_diff_counts(rows)
    
    def get_cost(rows, dc):
        scope = _get_scope(rows)
        collisions = sum(max(0, c - 1) for c in dc.values())
        # High penalty for collisions to ensure validity
        return scope + 1000 * collisions
    
    current_cost = get_cost(rows, diff_counts)
    best_cost = current_cost
    
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    no_improve_steps = 0
    max_no_improve = 300
    
    while T > min_T and time.time() < deadline - 10:
        r = rng.random()
        
        if r < 0.3:
            # Row rebuild move
            i = rng.randint(0, n_rows - 1)
            old_row = list(rows[i])
            _remove_row_diffs(diff_counts, old_row)
            
            # Try to rebuild with current scope, or slightly lower
            target_scope = current_scope
            if rng.random() < 0.2:
                target_scope = current_scope - 1
            
            new_row = _row_rebuild(old_row, k, diff_counts, rng, target_scope)
            
            if new_row is not None:
                _add_row_diffs(diff_counts, new_row)
                rows[i] = new_row
                new_cost = get_cost(rows, diff_counts)
                
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r2) for r2 in rows]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    _remove_row_diffs(diff_counts, new_row)
                    _add_row_diffs(diff_counts, old_row)
                    rows[i] = old_row
            else:
                _add_row_diffs(diff_counts, old_row)
                
        elif r < 0.5:
            # Scope compression: try to reduce the max element
            max_val = current_scope
            max_row_idx = -1
            max_pos_idx = -1
            for ri in range(n_rows):
                for pi in range(k, 0, -1):
                    if rows[ri][pi] == max_val:
                        max_row_idx = ri
                        max_pos_idx = pi
                        break
                if max_row_idx >= 0:
                    break
            
            if max_row_idx < 0:
                continue
                
            i = max_row_idx
            j = max_pos_idx
            old_val = rows[i][j]
            
            prev_val = rows[i][j-1]
            # Try to reduce old_val
            improved = False
            for _ in range(5):
                # Try a value between prev_val+1 and old_val-1
                if old_val > prev_val + 1:
                    new_val = rng.randint(prev_val + 1, old_val - 1)
                else:
                    continue
                
                _remove_row_diffs(diff_counts, rows[i])
                rows[i][j] = new_val
                _add_row_diffs(diff_counts, rows[i])
                
                new_cost = get_cost(rows, diff_counts)
                
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r2) for r2 in rows]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                    improved = True
                    break
                else:
                    _remove_row_diffs(diff_counts, rows[i])
                    rows[i][j] = old_val
                    _add_row_diffs(diff_counts, rows[i])
            
            if not improved:
                no_improve_steps += 1
                
        else:
            # Single-element move
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            
            if rng.random() < 0.5:
                new_val = rng.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                new_val = rng.randint(1, current_scope + 5)
                
            if new_val == old_val:
                continue
                
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
                
            _remove_row_diffs(diff_counts, row)
            row[pos] = new_val
            _add_row_diffs(diff_counts, row)
            
            new_cost = get_cost(rows, diff_counts)
            
            if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                current_cost = new_cost
                new_scope = _get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r2) for r2 in rows]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                _remove_row_diffs(diff_counts, row)
                row[pos] = old_val
                _add_row_diffs(diff_counts, row)
                no_improve_steps += 1
                
        T *= cooling
        
        # Restart if stuck
        if no_improve_steps > max_no_improve and time.time() < deadline - 20:
            no_improve_steps = 0
            # Re-initialize with best_rows to keep good structure, or random restart?
            # Let's try random restart with greedy
            skip_prob = 0.05
            new_rows, new_used = _greedy_init(n_rows, k, skip_prob=skip_prob, rng=rng)
            if _is_valid(new_rows, k):
                new_scope = _get_scope(new_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r2) for r2 in new_rows]
                    rows = [list(r2) for r2 in new_rows]
                    diff_counts = _build_diff_counts(rows)
                    current_scope = new_scope
                    current_cost = get_cost(rows, diff_counts)
                    T = 30.0
                else:
                    # Use as fresh start for SA if it's valid, even if worse
                    # Actually, if it's worse, don't switch. Just reset T.
                    T = 20.0
            else:
                T = 20.0
    
    # Final Polish: Greedy Scope Reduction
    # Try to reduce the scope further by rebuilding rows
    if time.time() < deadline - 5:
        current_best = [list(r) for r in best_rows]
        current_dc = _build_diff_counts(current_best)
        current_scope = _get_scope(current_best)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n_rows):
                if time.time() > deadline - 5:
                    break
                _remove_row_diffs(current_dc, current_best[i])
                # Try to rebuild row i with max_scope = current_scope - 1
                new_row = _row_rebuild(current_best[i], k, current_dc, rng, current_scope - 1)
                if new_row is not None:
                    _add_row_diffs(current_dc, new_row)
                    current_best[i] = new_row
                    new_scope = _get_scope(current_best)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        improved = True
                        # Reset loop to check again
                        break
                else:
                    # Restore original row
                    _add_row_diffs(current_dc, current_best[i])
            
    return current_best

def construct(n_rows, k):
    deadline = time.time() + 145.0 # Safety margin
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Pure greedy
    rows1, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
    if _is_valid(rows1, k):
        scope1 = _get_scope(rows1)
        if scope1 < best_scope:
            best_scope = scope1
            best_rows = [list(r) for r in rows1]
    
    # Strategy 2: Randomized greedy
    for attempt in range(10):
        if time.time() > deadline - 120:
            break
        rows2, _ = _greedy_init(n_rows, k, skip_prob=0.02 + 0.01 * attempt, rng=rng)
        if _is_valid(rows2, k):
            scope2 = _get_scope(rows2)
            if scope2 < best_scope:
                best_scope = scope2
                best_rows = [list(r) for r in rows2]
    
    if best_rows is None:
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0, rng=rng)
        return rows
    
    # Run SA optimization
    rows = _sa_optimize(best_rows, n_rows, k, deadline, rng)
    
    if not _is_valid(rows, k):
        return best_rows
    
    # Final check
    if _is_valid(rows, k):
        final_scope = _get_scope(rows)
        if final_scope < best_scope:
            best_scope = final_scope
            best_rows = rows
    
    return best_rows
