import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    """
    Fast greedy construction. For each row, greedily pick the smallest
    value that doesn't create a duplicate difference.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _csp_backtrack_row(pos, current_row, current_diffs, used_diffs, k, max_val, deadline, rng):
    """
    Backtrack to build a single valid row of length k+1 starting at 0,
    avoiding conflicts with used_diffs and internal conflicts.
    """
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)
    
    prev = current_row[-1]
    min_next = prev + 1
    
    # Forward checking
    if min_next + (n - pos - 1) > max_val:
        return None
    
    # Generate candidates: prefer smaller values, with some randomization
    max_next = min(max_val, prev + (max_val - prev))
    candidates = list(range(min_next, max_next + 1))
    
    # Shuffle upper portion for diversity
    if len(candidates) > 4:
        mid = len(candidates) // 3
        rng.shuffle(candidates[mid:])
    
    for val in candidates:
        if time.time() > deadline:
            return None
        
        # Check conflicts
        conflict = False
        new_diffs = []
        for idx in range(pos):
            d = val - current_row[idx]
            if d in current_diffs or d in used_diffs:
                conflict = True
                break
            new_diffs.append(d)
        
        if conflict:
            continue
        
        current_row.append(val)
        current_diffs.update(new_diffs)
        
        result = _csp_backtrack_row(pos + 1, current_row, current_diffs, used_diffs, k, max_val, deadline, rng)
        
        current_row.pop()
        for d in new_diffs:
            current_diffs.discard(d)
        
        if result is not None:
            return result
    
    return None

def _construct_rows_csp(n_rows, k, deadline, rng, target_scope):
    """
    Construct n_rows valid rows using CSP backtracking.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        best_row = None
        best_max_val = float('inf')
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            current_row = [0]
            current_diffs = set()
            
            result = _csp_backtrack_row(1, current_row, current_diffs, used_diffs, k, target_scope, deadline, rng)
            
            if result is not None:
                row_max = result[-1]
                if row_max < best_max_val:
                    best_max_val = row_max
                    best_row = result
                if best_max_val <= target_scope:
                    break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)
    
    return rows

def _row_rebuild_search(rows, k, deadline, rng):
    """
    Structural optimization: repeatedly pick a random row, remove it,
    and rebuild it via CSP backtracking with a tighter scope target.
    This allows global reorganization that single-element moves cannot achieve.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best = [list(r) for r in current]
    best_scope = current_scope
    
    # Precompute diff sets for each row
    row_diff_sets = [_row_diffs_set(r) for r in current]
    
    # Build global used set
    used_diffs = set()
    for ds in row_diff_sets:
        used_diffs.update(ds)
    
    T = 30.0
    cooling = 0.999
    min_T = 0.5
    
    it = 0
    while time.time() < deadline:
        it += 1
        
        # Pick a random row to rebuild
        i = rng.randint(0, n_rows - 1)
        old_row = current[i]
        old_diffs = row_diff_sets[i]
        old_max = old_row[-1]
        
        # Determine target scope for rebuild
        # Try to fit within current scope or slightly less
        target = current_scope
        if rng.random() < 0.3:
            target = current_scope - 1
        
        # Remove old row's diffs from used set
        for d in old_diffs:
            used_diffs.discard(d)
        
        # Try to rebuild the row
        best_new_row = None
        best_new_max = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            current_row = [0]
            current_diffs = set()
            
            result = _csp_backtrack_row(1, current_row, current_diffs, used_diffs, k, target, deadline, rng)
            
            if result is not None:
                row_max = result[-1]
                if row_max < best_new_max:
                    best_new_max = row_max
                    best_new_row = result
                if best_new_max <= target:
                    break
        
        # If no new row found, restore and try a different approach
        if best_new_row is None:
            for d in old_diffs:
                used_diffs.add(d)
            continue
        
        new_diffs = _row_diffs_set(best_new_row)
        new_max = best_new_row[-1]
        
        # Check if this improves scope
        new_scope = current_scope
        if new_max > current_scope:
            new_scope = new_max
        elif old_max == current_scope and new_max < old_max:
            # Check if any other row has the old scope
            any_other_at_scope = any(
                current[j][-1] == current_scope for j in range(n_rows) if j != i
            )
            if not any_other_at_scope:
                new_scope = new_max
        
        # Acceptance: always accept if scope doesn't increase, occasionally accept if it does
        if new_scope <= current_scope:
            accept = True
        else:
            delta = new_scope - current_scope
            prob = math.exp(-delta / max(T, min_T))
            accept = rng.random() < prob
        
        if accept:
            current[i] = best_new_row
            row_diff_sets[i] = new_diffs
            for d in new_diffs:
                used_diffs.add(d)
            current_scope = new_scope
            
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        else:
            # Restore old row
            for d in new_diffs:
                used_diffs.discard(d)
            for d in old_diffs:
                used_diffs.add(d)
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _scope_squeeze_fast(rows, k, deadline, rng):
    """
    Fast scope squeeze: try to reduce the last element of rows at max scope.
    Uses precomputed diff sets for O(1) conflict checks.
    """
    rows = [list(r) for r in rows]
    n_rows = len(rows)
    current_scope = _scope(rows)
    
    # Precompute diff sets for all rows
    row_diff_sets = [_row_diffs_set(r) for r in rows]
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Find rows at current scope
        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break
        
        progress_made = False
        for i in max_rows:
            if time.time() > deadline:
                break
            
            row = rows[i]
            old_max = row[-1]
            prev = row[k - 1]
            old_diffs = row_diff_sets[i]
            
            # Try reducing the last element
            for new_val in range(old_max - 1, prev, -1):
                # Compute new diffs for this row
                new_diffs = set()
                valid = True
                for idx in range(k):
                    d = new_val - row[idx]
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check conflicts with other rows
                conflict = False
                for j in range(n_rows):
                    if j == i:
                        continue
                    if new_diffs & row_diff_sets[j]:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept the change
                    row[k] = new_val
                    row_diff_sets[i] = new_diffs
                    current_scope = new_val
                    progress_made = True
                    break
            
            if progress_made:
                break
        
        if not progress_made:
            break
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: CSP backtracking construction
    for target_scope in [112, 114, 116, 120, 125, 130]:
        if time.time() > deadline - 25:
            break
        
        rows = _construct_rows_csp(n_rows, k, deadline - 15, rng, target_scope)
        
        if rows is not None:
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows
            if s <= target_scope:
                break
    
    # Phase 2: Fallback to greedy if CSP failed
    if best_result is None:
        rows = _fast_greedy(n_rows, k, rng)
        if rows is not None:
            best_result = rows
            best_score = _scope(rows)
        else:
            used = set()
            rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                rows.append(row)
            best_result = rows
            best_score = _scope(rows)
    
    # Phase 3: Row-rebuild structural optimization
    if time.time() < deadline - 15:
        remaining = deadline - time.time()
        rebuild_deadline = time.time() + remaining * 0.7
        s, rows = _row_rebuild_search(best_result, k, rebuild_deadline, rng)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 4: Fast scope squeeze
    if time.time() < deadline - 3:
        remaining = deadline - time.time()
        squeeze_deadline = time.time() + remaining * 0.85
        squeezed = _scope_squeeze_fast(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed
    
    # Final validation
    if best_result is not None and not _valid(best_result, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                rows.append(row)
            best_result = rows
    
    return best_result
