import time
import random

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

def _row_diffs_set(row):
    s = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            s.add(row[j] - row[i])
    return s

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
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

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

def _greedy_row_with_exclusions(k, used_diffs, scope_limit, rng):
    """
    Greedily construct a row of length k+1 starting at 0,
    avoiding differences in used_diffs.
    Returns None if not possible within scope_limit.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > scope_limit:
            return None
        conflict = False
        for a in row:
            if (x - a) in used_diffs:
                conflict = True
                break
        if not conflict:
            row.append(x)
            # Note: We don't add to used_diffs here because this is a "virtual" row
            # being tested against others. The caller manages the global set.
    return row

def _iterative_reconstruction(rows, k, deadline, rng):
    """
    Iteratively try to reconstruct rows to reduce the maximum scope.
    """
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    n_rows = len(rows)
    
    # Pre-compute diffs for all rows
    all_diffs_sets = [_diffs_for_row(r) for r in rows]
    global_used = set()
    for s in all_diffs_sets:
        global_used.update(s)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_scope = _scope(rows)
        
        # Try to improve each row
        # Sort rows by their max value descending to tackle the largest first?
        # Or random order? Random is safer to avoid bias.
        indices = list(range(n_rows))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            current_row = rows[idx]
            current_max = current_row[-1]
            
            # If this row doesn't determine the scope, we might still want to optimize it
            # to free up space for others, but primarily we target the max.
            
            # Compute used diffs excluding this row
            temp_used = set(global_used)
            for d in all_diffs_sets[idx]:
                temp_used.discard(d)
            
            # Try to find a better row with a smaller max
            # We try to reduce the max value of this row.
            # The new max must be < current_max to be an improvement for this row.
            # But even if it's not < current_max, if it frees up a difference
            # that allows *another* row to shrink, it could help.
            # However, simple hill climbing: try to reduce this row's max.
            
            # Target max: start from current_max - 1 down to some reasonable bound
            # But we must ensure we can fill k+1 elements.
            
            # A lower bound on the max value for a row of length k+1 with distinct diffs
            # is roughly k(k+1)/2 if packed tightly, but with exclusions it's larger.
            
            # Let's just try to find ANY valid row with max < current_max.
            # If we find one, we accept it.
            
            found = False
            # Try reducing max by 1, 2, ...
            # To save time, we can just try to construct a row with max = current_max - 1
            # If that fails, try current_max - 2, etc.
            # But greedy construction is fast.
            
            # Optimization: Just try to construct a row with max < current_max.
            # The greedy algorithm will pick the smallest possible numbers.
            # So if we call _greedy_row_with_exclusions with scope_limit = current_max - 1,
            # it will return the tightest possible row within that limit.
            
            new_row = _greedy_row_with_exclusions(k, temp_used, current_max - 1, rng)
            
            if new_row is not None:
                # Check if it's actually better (it should be, since max < current_max)
                if new_row[-1] < current_max:
                    # Update global state
                    global_used.difference_update(all_diffs_sets[idx])
                    rows[idx] = new_row
                    all_diffs_sets[idx] = _diffs_for_row(new_row)
                    global_used.update(all_diffs_sets[idx])
                    improved = True
                    # Continue to try to improve other rows
                else:
                    # Greedy found a row but it wasn't smaller?
                    # This implies current_max - 1 was not tight enough or greedy failed to pack tightly?
                    # Actually, if greedy succeeds with limit M, the max is <= M.
                    # So if it succeeds with limit current_max - 1, max <= current_max - 1 < current_max.
                    # So this branch is technically redundant if logic holds.
                    pass
            
            # If we didn't find a smaller row with current_max - 1,
            # it might be possible with a different structure, but greedy is deterministic
            # for a given exclusion set. So if greedy fails, we can't easily fix it
            # without backtracking. We skip for now to keep it fast.
            
    return rows

def construct(n_rows, k):
    # Safety margin: 120s limit, return by 115s
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    # Phase 1: Fast Greedy Initialization
    rows = _fast_greedy(n_rows, k, rng)
    
    if rows is None:
        # Fallback: Simple sequential (should not happen for small n,k)
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
    
    # Phase 2: Iterative Row Reconstruction
    # This is the main optimization step.
    improved_rows = _iterative_reconstruction(rows, k, deadline, rng)
    
    # Final validation
    if not _valid(improved_rows, k):
        # Should not happen, but fallback to initial greedy
        improved_rows = rows
    
    return improved_rows
