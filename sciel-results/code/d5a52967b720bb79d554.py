import time
import random
import sys

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _csp_build_row(forbidden, k, max_val, rng):
    """
    Try to build a single row of k+1 increasing integers starting at 0,
    with all elements <= max_val, such that no difference is in forbidden.
    Uses randomized backtracking.
    Returns the row if found, None otherwise.
    """
    # We need to pick k elements from 1..max_val such that:
    # 1. They are strictly increasing
    # 2. No difference between any two elements (including 0) is in forbidden
    # 3. No difference between any two elements is in the set of differences already in this row
    
    def backtrack(start, chosen, local_diffs):
        if len(chosen) == k:
            return chosen[:]
        
        # Pruning: can we still pick enough elements?
        remaining_needed = k - len(chosen)
        available = max_val - start + 1
        if available < remaining_needed:
            return None
        
        # Randomize the order of candidates to explore different solutions
        candidates = list(range(start, max_val + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            # Check if x can be added
            new_diffs = set()
            conflict = False
            for a in chosen:
                d = x - a
                if d in forbidden or d in local_diffs:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            # Also check against 0
            if x in forbidden or x in local_diffs:
                continue
            
            # Try adding x
            chosen.append(x)
            local_diffs = local_diffs | new_diffs | {x}
            
            result = backtrack(x + 1, chosen, local_diffs)
            if result is not None:
                return result
            
            chosen.pop()
            local_diffs = local_diffs - new_diffs - {x}
        
        return None
    
    return backtrack(1, [], set())

def _csp_solve(n_rows, k, max_scope, rng, deadline):
    """
    Try to find a complete valid set of n_rows rows with scope <= max_scope
    using CSP backtracking with forward checking.
    """
    forbidden = set()
    rows = []
    
    def solve_row(idx):
        if time.time() > deadline:
            return False
        
        if idx == n_rows:
            return True
        
        # Try to build row idx
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            if time.time() > deadline:
                return False
            
            row = _csp_build_row(forbidden, k, max_scope, rng)
            if row is not None:
                row_diffs = _compute_diffs(row)
                # Verify no conflict
                if not (row_diffs & forbidden):
                    rows.append(row)
                    forbidden |= row_diffs
                    
                    if solve_row(idx + 1):
                        return True
                    
                    # Backtrack
                    rows.pop()
                    forbidden -= row_diffs
            
            attempts += 1
        
        return False
    
    if solve_row(0):
        return rows
    
    return None

def _greedy_init(n_rows, k, rng):
    """
    Greedy initialization: build rows one by one.
    For each row, pick the smallest possible next integer that doesn't create a conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _local_search(rows, n_rows, k, deadline, rng):
    """
    Local search to minimize the maximum element.
    Uses simulated annealing with element perturbations and row rebuilds.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    current_scope = best_scope
    
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
    
    T = 50.0
    cooling = 0.999
    min_T = 1.0
    
    while T > min_T and time.time() < deadline:
        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        pos = rng.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try to move the element
        # Bias towards smaller values to reduce scope
        if rng.random() < 0.6:
            new_val = rng.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
        else:
            new_val = rng.randint(1, current_scope + 10)
        
        if new_val == old_val:
            T *= cooling
            continue
        
        # Check order
        if pos > 0 and new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            T *= cooling
            continue
        
        # Compute old diffs involving pos
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(old_val - row[i])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - old_val)
        
        # Compute new diffs involving pos
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, len(row)):
            new_diffs.add(row[j] - new_val)
        
        # Check conflict
        remaining = used - old_diffs
        if new_diffs & remaining:
            T *= cooling
            continue
        
        # Valid move
        row[pos] = new_val
        used = (used - old_diffs) | new_diffs
        
        new_scope = _get_scope(rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Revert
            row[pos] = old_val
            used = (used - new_diffs) | old_diffs
        
        T *= cooling
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = random.Random()
    
    # Start with greedy initialization
    rows = _greedy_init(n_rows, k, rng)
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Try to improve with local search
    rows = _local_search(rows, n_rows, k, deadline - 30, rng)
    scope = _get_scope(rows)
    if scope < best_scope:
        best_scope = scope
        best_rows = [list(r) for r in rows]
    
    # If time remains, try CSP to find potentially much better solutions
    if time.time() < deadline - 5:
        # Try CSP with decreasing max_scope
        for max_scope in range(best_scope - 1, best_scope - 20, -1):
            if time.time() > deadline - 5:
                break
            
            csp_rows = _csp_solve(n_rows, k, max_scope, rng, deadline - 5)
            if csp_rows is not None and _is_valid(csp_rows):
                csp_scope = _get_scope(csp_rows)
                if csp_scope < best_scope:
                    best_scope = csp_scope
                    best_rows = [list(r) for r in csp_rows]
                
                # Continue local search from CSP solution
                if time.time() < deadline - 5:
                    improved = _local_search(csp_rows, n_rows, k, deadline - 5, rng)
                    imp_scope = _get_scope(improved)
                    if imp_scope < best_scope:
                        best_scope = imp_scope
                        best_rows = [list(r) for r in improved]
    
    # Final verification
    if not _is_valid(best_rows):
        # Fallback to greedy
        rows = _greedy_init(n_rows, k, rng)
        return rows
    
    return best_rows
