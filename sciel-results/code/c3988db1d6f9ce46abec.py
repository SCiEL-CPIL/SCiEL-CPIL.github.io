import time
import random
import math

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

def _greedy_init(n_rows, k, rng, scope_limit=None):
    """
    Greedy construction that tries to minimize scope.
    For each row, pick the smallest valid next element.
    Uses randomized restarts with different orderings.
    """
    best_rows = None
    best_scope = float('inf')
    
    for restart in range(20):
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                # Try values in increasing order, with some randomization
                start = row[-1] + 1
                if scope_limit is not None:
                    end = scope_limit + 1
                else:
                    end = start + 200
                for x in range(start, end):
                    if scope_limit is not None and x > scope_limit:
                        break
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        
        if success:
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Last resort: no scope limit
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
        best_rows = rows
        best_scope = _scope(rows)
    
    return best_rows, best_scope

def _csp_row(pos, current_row, used_diffs, max_val, deadline):
    """
    Backtrack to build a single row given used_diffs from other rows.
    Returns the completed row or None.
    """
    if time.time() > deadline:
        return None
    k = len(used_diffs)  # not used directly, we use len(current_row)
    n = max_val  # placeholder
    # We need k+1 elements total, currently have len(current_row)
    # pos = len(current_row) - 1 (last placed index)
    
    if len(current_row) > 6:  # safety
        return None
    
    target_len = 6  # k+1 for k=5; we'll make this generic
    # Actually let's pass k explicitly
    return None  # replaced below

def _csp_build_row(pos, current_row, used_diffs, max_val, deadline, k):
    """
    Build a row of length k+1 starting at 0, strictly increasing,
    with all diffs not in used_diffs and internally distinct.
    """
    if time.time() > deadline:
        return None
    n = k + 1
    if pos == n:
        return list(current_row)
    
    prev = current_row[-1]
    min_next = prev + 1
    max_next = min(max_val, prev + (n - pos - 1))  # need at least 1 gap per remaining element
    
    if min_next > max_next:
        return None
    
    # Forward check: enough available diffs
    remaining_elements = n - pos - 1
    # Each new element at position p (0-indexed from start) creates p new diffs
    # We're placing element at index pos (0-indexed), so it creates pos new diffs
    # Remaining elements at pos+1, ..., n-1 create (pos+1), ..., (n-1) diffs
    total_remaining_diffs = sum(range(pos, n))  # pos + (pos+1) + ... + (n-1)
    # Actually: element at index i creates i diffs (to indices 0..i-1)
    # We've placed indices 0..pos-1. Next is index pos, which creates pos diffs.
    # Then pos+1 creates pos+1 diffs, etc.
    total_remaining_diffs = sum(range(pos, n))
    available = max_val - len(used_diffs)
    if available < total_remaining_diffs:
        return None
    
    # Try candidates
    candidates = list(range(min_next, max_next + 1))
    # Sort by how "tight" they are (prefer smaller to minimize scope)
    # But add some randomness
    if len(candidates) > 12:
        rng_local = random.Random(pos * 1000 + len(current_row))
        mid = len(candidates) // 2
        rng_local.shuffle(candidates[mid:])
    
    for val in candidates:
        if time.time() > deadline:
            return None
        
        # Check conflicts with used_diffs and current row's internal diffs
        conflict = False
        for idx in range(pos):
            d = val - current_row[idx]
            if d in used_diffs:
                conflict = True
                break
        if conflict:
            continue
        
        # Check internal consistency (no duplicate diffs within this row)
        # We need to check that the new diffs don't duplicate existing ones in current_row
        new_diffs = [val - current_row[idx] for idx in range(pos)]
        existing_diffs = set()
        for i in range(len(current_row)):
            for j in range(i + 1, len(current_row)):
                existing_diffs.add(current_row[j] - current_row[i])
        for d in new_diffs:
            if d in existing_diffs:
                conflict = True
                break
        if conflict:
            continue
        
        current_row.append(val)
        result = _csp_build_row(pos + 1, current_row, used_diffs, max_val, deadline, k)
        current_row.pop()
        
        if result is not None:
            return result
    
    return None

def _csp_construct(n_rows, k, max_val, deadline, rng):
    """
    Construct n_rows using CSP backtracking with forward checking.
    Tries to build each row within max_val scope.
    """
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        
        best_row = None
        best_max = max_val + 1
        
        # Try multiple times with slight variations
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            current_row = [0]
            result = _csp_build_row(1, current_row, used_diffs, max_val, deadline, k)
            
            if result is not None:
                row_max = result[-1]
                if row_max < best_max:
                    best_max = row_max
                    best_row = result
                if best_max <= max_val:
                    break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        for d in _diffs_for_row(best_row):
            used_diffs.add(d)
    
    return rows

def _row_rebuild_sa(rows, k, deadline, rng):
    """
    Simulated Annealing with row-rebuild moves.
    Each move rebuilds an entire row from scratch, trying to minimize scope
    while respecting other rows' differences.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Precompute diff sets for each row
    row_diff_sets = [_row_diffs_set(r) for r in current]
    
    # Build global used set (should be all distinct for valid input)
    all_used = set()
    for s in row_diff_sets:
        all_used.update(s)
    
    best = [list(r) for r in current]
    best_scope = current_scope
    
    T = 30.0
    cooling = 0.99995
    min_T = 0.01
    
    def get_scope():
        return max(r[-1] for r in current)
    
    while time.time() < deadline:
        # Pick a row to rebuild
        i = rng.randint(0, n_rows - 1)
        
        # Remove this row's diffs from the used set
        other_used = set()
        for j in range(n_rows):
            if j != i:
                other_used.update(row_diff_sets[j])
        
        # Try to rebuild row i with a target max
        # Target: try to keep it below current_scope, or slightly above
        target_max = min(current_scope, current[i][-1] + rng.randint(-5, 5))
        target_max = max(target_max, k)  # minimum possible
        
        new_row = None
        for attempt in range(20):
            if time.time() > deadline:
                break
            current_row = [0]
            new_row = _csp_build_row(1, current_row, other_used, target_max, deadline, k)
            if new_row is not None:
                break
        
        if new_row is None:
            continue
        
        new_row_max = new_row[-1]
        new_diff_set = _row_diffs_set(new_row)
        
        # Check: new row's diffs must not conflict with other rows
        if new_diff_set & other_used:
            continue
        
        # Compute new scope
        new_scope = max(r[-1] for j, r in enumerate(current) if j != i)
        new_scope = max(new_scope, new_row_max)
        
        old_c = current_scope
        new_c = new_scope
        
        if new_c <= old_c:
            accept = True
        else:
            delta = new_c - old_c
            prob = math.exp(-delta / max(T, min_T))
            accept = rng.random() < prob
        
        if accept:
            current[i] = new_row
            row_diff_sets[i] = new_diff_set
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        else:
            pass  # don't modify
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _compress(rows, k, deadline, rng):
    """
    Iteratively try to reduce scope by 1.
    For each row that has the max value, try to find a replacement
    that doesn't use the max value.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Find rows with the maximum value
        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break
        
        # Try to reduce one of these rows
        progress = False
        for i in max_rows:
            if time.time() > deadline:
                break
            
            row = rows[i]
            old_diff_set = _row_diffs_set(row)
            
            # Build other rows' diff set
            other_used = set()
            for j in range(len(rows)):
                if j != i:
                    other_used.update(_row_diffs_set(rows[j]))
            
            # Try to rebuild this row with max_val = current_scope - 1
            target_max = current_scope - 1
            if target_max < k:
                continue
            
            new_row = None
            for attempt in range(30):
                if time.time() > deadline:
                    break
                current_row = [0]
                new_row = _csp_build_row(1, current_row, other_used, target_max, deadline, k)
                if new_row is not None:
                    break
            
            if new_row is not None:
                new_diff_set = _row_diffs_set(new_row)
                if not (new_diff_set & other_used):
                    rows[i] = new_row
                    current_scope = target_max
                    progress = True
                    break
        
        if not progress:
            break
    
    return rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin
    rng = random.Random(12345)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: CSP construction with binary search on scope
    if time.time() < deadline - 60:
        total_diffs = n_rows * k * (k + 1) // 2
        low = max(1, total_diffs)
        high = 200
        
        best_rows_csp = None
        best_M = high
        
        # Try a few values from high to low
        for M in range(high, low - 1, -5):
            if time.time() > deadline - 60:
                break
            rows = _csp_construct(n_rows, k, M, deadline - 60, rng)
            if rows is not None:
                s = _scope(rows)
                if s < best_score:
                    best_score = s
                    best_result = rows
                    best_M = M
                if M <= low:
                    break
        
        # Fine-tune around best_M
        if best_result is not None and time.time() < deadline - 40:
            for M in range(max(low, best_M - 10), min(high, best_M + 1)):
                if time.time() > deadline - 40:
                    break
                rows = _csp_construct(n_rows, k, M, deadline - 40, rng)
                if rows is not None:
                    s = _scope(rows)
                    if s < best_score:
                        best_score = s
                        best_result = rows
    
    # Phase 2: Greedy fallback if CSP failed
    if best_result is None:
        rows, s = _greedy_init(n_rows, k, rng)
        best_result = rows
        best_score = s
    
    # Phase 3: Row-rebuild SA optimization
    if time.time() < deadline - 15:
        sa_deadline = time.time() + (deadline - time.time()) * 0.7
        s, rows = _row_rebuild_sa(best_result, k, sa_deadline, rng)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 4: Compress
    if time.time() < deadline - 3:
        compress_deadline = time.time() + (deadline - time.time()) * 0.85
        squeezed = _compress(best_result, k, compress_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed
    
    # Final validation
    if best_result is not None and not _valid(best_result, k):
        # Fallback: simple greedy
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
