"""Difference triangle set solver with backtracking and local search."""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)


def _valid(rows):
    """Check if all positive differences across all rows are distinct."""
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _greedy_row(used, k, max_val=None):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    """
    row = [0]
    new_diffs = set()
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        x += 1
        if x > limit:
            return None, None
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs


def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    return rows


def _backtrack_row(used, k, max_val, rng):
    """
    Backtracking search for a single row of k+1 integers starting at 0,
    with maximum value at most max_val, avoiding differences in 'used'.
    """
    row = [0]
    new_diffs = set()
    
    def backtrack(pos, last_val):
        if pos == k:  # Need k more elements (we have 1 already, need k+1 total)
            return True
        # Try candidates from last_val+1 to max_val
        for candidate in range(last_val + 1, max_val + 1):
            new = {candidate - a for a in row}
            if new & used or new & new_diffs:
                continue
            row.append(candidate)
            old_diffs = set(new_diffs)
            new_diffs |= new
            if backtrack(pos + 1, candidate):
                return True
            row.pop()
            new_diffs.clear()
            new_diffs |= old_diffs
        return False
    
    if backtrack(1, 0):
        return row, new_diffs
    return None, None


def _construct_with_backtracking(n_rows, k, scope_limit, rng):
    """
    Try to construct n_rows difference triangles with scope at most scope_limit.
    Uses backtracking for each row.
    """
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row, new_diffs = _backtrack_row(used, k, scope_limit, rng)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    
    return rows


def _local_search_improve(rows, k, rng, deadline):
    """
    Local search that tries to reduce the scope by moving elements downward.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Precompute all differences
    all_diffs = set()
    for r in current_rows:
        all_diffs |= _row_diffs(r)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = current_rows[row_idx]
            for elem_idx in range(len(row) - 1, 0, -1):
                old_val = row[elem_idx]
                # Try to decrease this element
                for new_val in range(old_val - 1, row[elem_idx - 1], -1):
                    # Check if moving this element creates conflicts
                    old_diffs = set()
                    for j in range(elem_idx):
                        old_diffs.add(old_val - row[j])
                    for j in range(elem_idx + 1, len(row)):
                        old_diffs.add(row[j] - old_val)
                    
                    new_diffs = set()
                    for j in range(elem_idx):
                        new_diffs.add(new_val - row[j])
                    for j in range(elem_idx + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    
                    remaining = all_diffs - old_diffs
                    if new_diffs & remaining:
                        continue
                    
                    if new_val <= row[elem_idx - 1]:
                        break
                    if elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]:
                        continue
                    
                    # Apply move
                    row[elem_idx] = new_val
                    all_diffs = remaining | new_diffs
                    new_scope = _scope(current_rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                    current_scope = new_scope
                    improved = True
                    break
                
                if time.time() > deadline:
                    return best_rows
        
        # If no improvement, try random perturbations
        if not improved and time.time() < deadline - 1:
            for _ in range(30):
                if time.time() > deadline:
                    break
                row_idx = rng.randint(0, n_rows - 1)
                row = current_rows[row_idx]
                elem_idx = rng.randint(1, len(row) - 1)
                old_val = row[elem_idx]
                step = rng.randint(-15, 15)
                if step == 0:
                    continue
                new_val = old_val + step
                if new_val <= row[elem_idx - 1] or (elem_idx < len(row) - 1 and new_val >= row[elem_idx + 1]):
                    continue
                
                old_diffs = set()
                for j in range(elem_idx):
                    old_diffs.add(old_val - row[j])
                for j in range(elem_idx + 1, len(row)):
                    old_diffs.add(row[j] - old_val)
                
                new_diffs = set()
                for j in range(elem_idx):
                    new_diffs.add(new_val - row[j])
                for j in range(elem_idx + 1, len(row)):
                    new_diffs.add(row[j] - new_val)
                
                remaining = all_diffs - old_diffs
                if new_diffs & remaining:
                    continue
                
                row[elem_idx] = new_val
                all_diffs = remaining | new_diffs
                new_scope = _scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                current_scope = new_scope
                improved = True
    
    return best_rows


def _difference_swap_mutation(rows, k, rng, deadline):
    """
    Try to swap differences between rows to reduce scope.
    This is a more aggressive local search move.
    """
    n_rows = len(rows)
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    all_diffs = set()
    for r in current_rows:
        all_diffs |= _row_diffs(r)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Pick two random rows
        r1_idx, r2_idx = rng.sample(range(n_rows), 2)
        r1 = current_rows[r1_idx]
        r2 = current_rows[r2_idx]
        
        # Pick random elements from each row (not index 0)
        e1_idx = rng.randint(1, len(r1) - 1)
        e2_idx = rng.randint(1, len(r2) - 1)
        
        old_v1 = r1[e1_idx]
        old_v2 = r2[e2_idx]
        
        # Try swapping values (with bounds check)
        # This is complex; instead try small shifts
        shift1 = rng.randint(-5, 5)
        shift2 = rng.randint(-5, 5)
        if shift1 == 0 and shift2 == 0:
            continue
        
        new_v1 = old_v1 + shift1
        new_v2 = old_v2 + shift2
        
        # Check bounds
        if new_v1 <= r1[e1_idx - 1] or (e1_idx < len(r1) - 1 and new_v1 >= r1[e1_idx + 1]):
            continue
        if new_v2 <= r2[e2_idx - 1] or (e2_idx < len(r2) - 1 and new_v2 >= r2[e2_idx + 1]):
            continue
        
        # Check validity
        # Compute diffs for modified rows
        new_diffs_r1 = set()
        for i in range(len(r1)):
            for j in range(i + 1, len(r1)):
                a = r1[i] if i != e1_idx else new_v1
                b = r1[j] if j != e1_idx else new_v1
                if b > a:
                    new_diffs_r1.add(b - a)
        
        new_diffs_r2 = set()
        for i in range(len(r2)):
            for j in range(i + 1, len(r2)):
                a = r2[i] if i != e2_idx else new_v2
                b = r2[j] if j != e2_idx else new_v2
                if b > a:
                    new_diffs_r2.add(b - a)
        
        # Other rows' diffs
        other_diffs = set()
        for idx in range(n_rows):
            if idx != r1_idx and idx != r2_idx:
                other_diffs |= _row_diffs(current_rows[idx])
        
        # Check for conflicts
        if new_diffs_r1 & new_diffs_r2 or new_diffs_r1 & other_diffs or new_diffs_r2 & other_diffs:
            continue
        
        # Apply
        r1[e1_idx] = new_v1
        r2[e2_idx] = new_v2
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
    
    return best_rows


def _randomized_backtracking(n_rows, k, scope_limit, rng, deadline):
    """
    Randomized backtracking with multiple restarts.
    """
    best_rows = None
    best_scope = float('inf')
    
    for restart in range(30):
        if time.time() > deadline:
            break
        rows = _construct_with_backtracking(n_rows, k, scope_limit, rng)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: Pure greedy to get an initial solution
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: very simple construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Local search on greedy solution
    if time.time() < deadline - 30:
        ls_rows = _local_search_improve(best_rows, k, rng, deadline - 30)
        scope = _scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]
    
    # Phase 3: Try backtracking with progressively lower scope limits
    # Start from current best and try to go lower
    if time.time() < deadline - 20:
        for scope_try in range(best_scope - 1, max(best_scope - 20, 50), -1):
            if time.time() > deadline - 20:
                break
            bt_rows = _construct_with_backtracking(n_rows, k, scope_try, rng)
            if bt_rows is not None:
                scope = _scope(bt_rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [r[:] for r in bt_rows]
                break  # Found one, stop trying lower
    
    # Phase 4: More local search + difference swaps
    if time.time() < deadline - 10:
        ls_rows = _local_search_improve(best_rows, k, rng, deadline - 10)
        scope = _scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]
        
        if time.time() < deadline - 5:
            swap_rows = _difference_swap_mutation(best_rows, k, rng, deadline - 5)
            scope = _scope(swap_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in swap_rows]
    
    # Phase 5: Final local search pass
    if time.time() < deadline - 2:
        ls_rows = _local_search_improve(best_rows, k, rng, deadline - 2)
        scope = _scope(ls_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in ls_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
