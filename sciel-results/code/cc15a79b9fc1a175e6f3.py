"""Difference triangle set (7,5) - constraint-satisfaction backtracking with MRV heuristic."""
import time
import random

_DEADLINE = 120.0


def _row_diffs(row):
    """All positive differences a_j - a_i for i < j in a single row."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences are distinct."""
    all_d = []
    for r in rows:
        all_d.extend(_row_diffs(r))
    return len(all_d) == len(set(all_d))


def _greedy(n_rows, k, used=None, rows=None, start_offset=0):
    """Greedy ruler construction. Returns (rows, used)."""
    if used is None:
        used = set()
    if rows is None:
        rows = []
    for r_idx in range(n_rows):
        row, x = [0], start_offset
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows, used


def _count_valid_placements(row, pos, used_set, max_val):
    """
    Count how many values v in valid range for position pos would not conflict
    with used_set. This is a heuristic for 'degree of future constraint violations'.
    """
    earlier = row[:pos]
    later = row[pos+1:]
    
    low = (earlier[-1] + 1) if earlier else 1
    high = (later[0] - 1) if later else max_val
    
    if low > high:
        return 0, low, high
    
    count = 0
    for v in range(low, high + 1):
        cand_diffs = set()
        for a in earlier:
            cand_diffs.add(v - a)
        for b in later:
            cand_diffs.add(b - v)
        if not (cand_diffs & used_set):
            count += 1
    
    return count, low, high


def _backtrack(rows, k, deadline, rng, max_scope_hint=None):
    """
    Backtracking search that builds rows one element at a time.
    Uses MRV (minimum remaining values) heuristic to choose next position.
    """
    start_time = time.time()
    
    # We'll build all rows together, tracking used differences
    # State: list of rows (partially filled), used set of differences
    # We process rows in order, and within each row, positions left to right
    # But we use a smarter ordering: always pick the next (row, pos) that has
    # the fewest valid placements
    
    # Actually, let's do a different approach:
    # Build row by row. For each row, try to fill it using backtracking.
    # The key insight: when choosing the next value for a position, prefer
    # values that leave the most options for subsequent positions/rows.
    
    best_found = None
    best_scope = max_scope_hint if max_scope_hint else 200
    
    # Start from a greedy solution as initial upper bound
    greedy_rows, greedy_used = _greedy(len(rows), k)
    greedy_scope = _scope(greedy_rows)
    if greedy_scope < best_scope:
        best_scope = greedy_scope
        best_found = [list(r) for r in greedy_rows]
    
    # Now try to improve via backtracking with a tighter bound
    # We'll try to build rows with max value < best_scope
    
    def solve_row(row, pos, used, remaining_rows, max_val):
        """
        Recursively fill position pos in row, then continue with remaining rows.
        Returns True if a complete valid solution is found, updating best_found.
        """
        nonlocal best_found, best_scope
        
        if time.time() > deadline - 1.0:
            return False
        
        # Check time
        if len(row) == k + 1:
            # Row complete, move to next row or finish
            if remaining_rows == 0:
                # All rows complete
                scope = max(max(r) for r in [row] + [r for r in _current_rows])
                if scope < best_scope:
                    best_scope = scope
                    best_found = [list(r) for r in _current_rows]
                    best_found.append(list(row))
                    # Actually, let's track this differently
                return True
            else:
                # Start next row
                return solve_row([0], 1, used, remaining_rows - 1, max_val)
        
        # Choose next position using MRV: find position with fewest valid values
        # But we're filling left to right, so just pick the current pos
        # Instead, let's use a heuristic: try values that minimize future conflicts
        
        earlier = row[:pos]
        low = (earlier[-1] + 1) if earlier else 1
        high = min(max_val, (row[pos+1] - 1) if pos + 1 < len(row) else max_val)
        
        if low > high:
            return False
        
        # Generate candidates: prefer smaller values (to reduce scope)
        # but also consider values that leave more options
        candidates = list(range(low, high + 1))
        
        # Score candidates by how many future options they leave
        # For efficiency, just try in order but with some randomization
        rng.shuffle(candidates)
        
        # Try candidates, preferring those that don't create many conflicts
        for v in candidates:
            if time.time() > deadline - 1.0:
                return False
            
            # Compute differences this value creates
            cand_diffs = set()
            for a in earlier:
                cand_diffs.add(v - a)
            # Also need to check against future positions in this row, but
            # we don't know them yet. So just check against current used.
            
            if cand_diffs & used:
                continue
            
            # Place this value
            new_row = row + [v]
            new_used = used | cand_diffs
            
            if pos + 1 < k + 1:
                # Continue filling this row
                if solve_row(new_row, pos + 1, new_used, remaining_rows, max_val):
                    return True
            else:
                # Row is complete (has k+1 elements)
                if remaining_rows == 0:
                    # Check if this completes a valid solution
                    all_rows = list(_current_rows) + [new_row]
                    scope = _scope(all_rows)
                    if scope < best_scope:
                        best_scope = scope
                        best_found = [list(r) for r in all_rows]
                    return True
                else:
                    # Start next row
                    if solve_row([0], 1, new_used, remaining_rows - 1, max_val):
                        return True
        
        return False
    
    # This approach is getting complicated. Let's use a cleaner iterative deepening.
    # Try scopes from greedy_scope-1 down to some minimum.
    
    min_scope = 100  # counting bound is 105, so go a bit below
    
    for target_scope in range(best_scope - 1, min_scope - 1, -1):
        if time.time() > deadline - 2.0:
            break
        
        # Try to find a solution with max entry <= target_scope
        found = _solve_with_bound(n_rows=len(rows), k=k, max_val=target_scope, 
                                   deadline=deadline, rng=rng)
        if found:
            best_found = found
            best_scope = target_scope
            # Continue trying to improve (lower scope)
            continue
        else:
            # No solution at this scope, try lower
            continue
    
    return best_found if best_found else None


def _solve_with_bound(n_rows, k, max_val, deadline, rng):
    """
    Try to construct n_rows rows of k+1 increasing integers starting at 0,
    all with max entry <= max_val, such that all differences are distinct.
    Returns the solution (list of rows) or None.
    """
    start_time = time.time()
    
    # State: current row being built, index in row, used differences, 
    #         list of completed rows
    # Use backtracking with forward checking
    
    def backtrack(row, pos, used, completed):
        if time.time() > deadline - 1.0:
            return None
        
        if len(row) == k + 1:
            # Row complete
            if len(completed) == n_rows:
                # All rows done
                return completed + [row]
            else:
                # Start next row
                return backtrack([0], 1, used, completed + [row])
        
        # Fill position pos in current row
        earlier = row[:pos]
        low = (earlier[-1] + 1) if earlier else 1
        high = min(max_val, (row[pos+1] - 1) if pos + 1 < len(row) else max_val)
        
        if low > high:
            return None
        
        # Generate candidate values
        # Heuristic: try values that minimize the number of differences they use
        # (i.e., values closer to the middle of the range might be better)
        # Actually, for scope minimization, try smaller values first
        # But also add some randomness to escape local minima
        
        candidates = list(range(low, high + 1))
        rng.shuffle(candidates)
        
        for v in candidates:
            if time.time() > deadline - 1.0:
                return None
            
            # Check if this value creates any conflicts
            cand_diffs = set()
            for a in earlier:
                cand_diffs.add(v - a)
            
            if cand_diffs & used:
                continue
            
            new_row = row + [v]
            new_used = used | cand_diffs
            
            result = backtrack(new_row, pos + 1, new_used, completed)
            if result is not None:
                return result
        
        return None
    
    return backtrack([0], 1, set(), [])


def _local_search_shrink(rows, k, deadline):
    """
    Given a valid set of rows, try to shrink the scope by moving elements.
    Returns improved rows or the original if no improvement.
    """
    rows = [list(r) for r in rows]
    used = set()
    for r in rows:
        used.update(_row_diffs(r))
    
    best_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    
    while time.time() < deadline - 2.0:
        improved = False
        for i in range(len(rows)):
            row = rows[i]
            for p in range(len(row) - 1, 0, -1):
                old_val = row[p]
                earlier = row[:p]
                later = row[p+1:]
                
                old_diffs = set()
                for a in earlier:
                    old_diffs.add(old_val - a)
                for b in later:
                    old_diffs.add(b - old_val)
                
                low = (earlier[-1] + 1) if earlier else 1
                high = (later[0] - 1) if later else old_val
                
                if low > high:
                    continue
                
                for v in range(old_val - 1, low - 1, -1):
                    cand_diffs = set()
                    for a in earlier:
                        cand_diffs.add(v - a)
                    for b in later:
                        cand_diffs.add(b - v)
                    
                    if not (cand_diffs & (used - old_diffs)):
                        used -= old_diffs
                        used |= cand_diffs
                        row[p] = v
                        improved = True
                        break
            if time.time() > deadline - 2.0:
                break
        
        if not improved:
            break
    
    if _scope(rows) < best_scope:
        return rows
    return best_rows


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - 5.0
    
    # Phase 1: Greedy baseline
    rows, used = _greedy(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Local search to shrink
    if time.time() < deadline - 10.0:
        ls_rows = _local_search_shrink(best_rows, k, min(deadline - 5.0, start + 15.0))
        if _is_valid(ls_rows) and _scope(ls_rows) < best_scope:
            best_rows = ls_rows
            best_scope = _scope(best_rows)
    
    # Phase 3: Try backtracking with decreasing scope bounds
    rng = random.Random(42)
    
    # Try to find solutions with scope less than current best
    for target in range(best_scope - 1, max(100, best_scope - 30), -1):
        if time.time() > deadline - 5.0:
            break
        
        result = _solve_with_bound(n_rows, k, target, 
                                    min(deadline - 3.0, time.time() + 15.0), rng)
        if result and _is_valid(result):
            scope = _scope(result)
            if scope < best_scope:
                best_rows = [list(r) for r in result]
                best_scope = scope
                # Do a quick local search
                if time.time() < deadline - 5.0:
                    ls = _local_search_shrink(best_rows, k, min(deadline - 3.0, time.time() + 5.0))
                    if _is_valid(ls) and _scope(ls) < best_scope:
                        best_rows = ls
                        best_scope = _scope(best_rows)
    
    # Phase 4: Multiple random restarts with backtracking
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 5.0:
            break
        
        rng2 = random.Random(seed * 1000)
        # Start from a slightly perturbed greedy
        offset = rng2.randint(0, 10)
        g_rows, g_used = _greedy(n_rows, k, used=set(), rows=[], start_offset=offset)
        
        g_scope = _scope(g_rows)
        if g_scope < best_scope:
            best_rows = [list(r) for r in g_rows]
            best_scope = g_scope
        
        # Try backtracking from this starting point with a tight bound
        target = min(best_scope - 1, g_scope - 1)
        if target > 100:
            result = _solve_with_bound(n_rows, k, target,
                                        min(deadline - 3.0, time.time() + 10.0), rng2)
            if result and _is_valid(result):
                scope = _scope(result)
                if scope < best_scope:
                    best_rows = [list(r) for r in result]
                    best_scope = scope
    
    return best_rows
