"""Difference triangle set (7,5) – compact construction via randomized greedy + local search."""
import time
import random
import copy

# EVOLVE-BLOCK-START

def _validate(rows):
    """Check that all positive differences are distinct across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True

def _scope(rows):
    """Largest entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_row(used, k, start_x=1, max_x=200):
    """Build one row greedily: pick smallest x such that all diffs to existing marks are unused."""
    row = [0]
    x = start_x
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used |= new_diffs
        x += 1
        if x > max_x:
            return None
    return row

def _greedy_construct(n_rows, k, used=None, rows=None, seed=0):
    """Build all rows greedily."""
    if used is None:
        used = set()
    if rows is None:
        rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, start_x=1, max_x=200)
        if row is None:
            return None
        rows.append(row)
    return rows

def _remove_row_differences(used, row):
    """Remove all positive differences of a row from the used set."""
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            used.discard(row[j] - row[i])

def _add_row_differences(used, row):
    """Add all positive differences of a row to the used set."""
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            used.add(row[j] - row[i])

def _rebuild_row(rows, idx, k, used, max_x=200, attempts=50):
    """Try to rebuild row idx by removing it, then greedily inserting a new row.
    Returns True if a valid replacement was found."""
    old_row = rows[idx]
    _remove_row_differences(used, old_row)
    
    for _ in range(attempts):
        new_row = _greedy_row(used, k, start_x=1, max_x=max_x)
        if new_row is not None:
            rows[idx] = new_row
            _add_row_differences(used, new_row)
            return True
    
    # Restore old row
    rows[idx] = old_row
    _add_row_differences(used, old_row)
    return False

def _local_search(rows, k, used, max_iters=2000, max_x=200, time_limit=10.0):
    """Iteratively try to reduce scope by rebuilding rows."""
    start_time = time.time()
    best_scope = _scope(rows)
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random row to try rebuilding
        idx = random.randint(0, len(rows) - 1)
        current_scope = _scope(rows)
        
        if _rebuild_row(rows, idx, k, used, max_x=max_x, attempts=10):
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
                # Also try to tighten other rows
                for _ in range(3):
                    j = random.randint(0, len(rows) - 1)
                    _rebuild_row(rows, j, k, used, max_x=max_x, attempts=5)
        
        if _scope(rows) > best_scope + 20:
            # Revert if it got too bad (shouldn't happen with valid replacements)
            pass
    
    return rows

def _diverse_greedy(n_rows, k, seed=0):
    """Generate diverse greedy constructions by randomizing the order and start offsets."""
    rng = random.Random(seed)
    used = set()
    rows = []
    order = list(range(n_rows))
    
    for _ in range(n_rows):
        # Try different starting points
        best_row = None
        best_scope = float('inf')
        for start in [1, 2, 3, 1, 2, 1, 3, 2]:
            saved_used = set(used)
            row = _greedy_row(used, k, start_x=start, max_x=200)
            if row is not None:
                s = max(row)
                if s < best_scope:
                    best_scope = s
                    best_row = row
            used = saved_used
            if best_row is not None and best_scope <= 50:
                break
        
        if best_row is None:
            return None
        rows.append(best_row)
        _add_row_differences(used, best_row)
    
    return rows

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0 with distinct differences."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # First, get a valid baseline with the original greedy
    baseline = _greedy_construct(n_rows, k, seed=0)
    if baseline is None:
        # Fallback: very conservative greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, start_x=1, max_x=500)
            if row is None:
                # Should not happen for reasonable inputs
                rows.append(list(range(k + 1)))
            else:
                rows.append(row)
        return rows
    
    best_rows = baseline
    best_scope = _scope(best_rows)
    
    # Build used set for best_rows
    best_used = set()
    for row in best_rows:
        _add_row_differences(best_used, row)
    
    # Try multiple diverse greedy constructions
    for seed in range(50):
        if time.time() > deadline - 30.0:
            break
        rows = _diverse_greedy(n_rows, k, seed=seed + 1)
        if rows is not None:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
                best_used = set()
                for row in best_rows:
                    _add_row_differences(best_used, row)
    
    # Local search on the best solution
    if time.time() < deadline - 20.0:
        rows_copy = [list(r) for r in best_rows]
        used_copy = set(best_used)
        _local_search(rows_copy, k, used_copy, max_iters=3000, max_x=200, time_limit=min(20.0, deadline - time.time() - 5.0))
        s = _scope(rows_copy)
        if s < best_scope:
            best_scope = s
            best_rows = rows_copy
    
    # More local search rounds with different parameters
    remaining = deadline - time.time()
    if remaining > 10.0:
        for rnd in range(5):
            if time.time() > deadline - 5.0:
                break
            rows_copy = [list(r) for r in best_rows]
            used_copy = set()
            for row in rows_copy:
                _add_row_differences(used_copy, row)
            _local_search(rows_copy, k, used_copy, max_iters=2000, max_x=180, time_limit=min(10.0, deadline - time.time() - 5.0))
            s = _scope(rows_copy)
            if s < best_scope:
                best_scope = s
                best_rows = rows_copy
    
    return best_rows

# EVOLVE-BLOCK-END
