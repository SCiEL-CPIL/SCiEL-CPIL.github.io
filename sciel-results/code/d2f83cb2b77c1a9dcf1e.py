import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Get a quick initial solution with simple greedy
    init_rows = _simple_greedy(n_rows, k)
    if not init_rows:
        return []
    current_best_scope = max(max(r) for r in init_rows)
    current_best_rows = init_rows

    # Try to improve with backtracking search targeting lower scopes
    # Start from a scope near the counting bound and work up
    num_diffs_per_row = (k + 1) * k // 2
    counting_bound = n_rows * num_diffs_per_row
    # The counting bound is a lower bound on the largest difference needed
    # But the scope (largest entry) must be at least the counting bound in some sense
    # Actually, the counting bound is on the number of distinct differences, not scope
    # Let's try scopes from a reasonable lower bound up to current best
    
    # A reasonable lower bound for scope: we need enough room
    # For k+1 elements starting at 0, the max is at least k (if consecutive)
    # But with the difference constraint, it's much larger
    # Let's try to find a solution with scope < current_best_scope
    
    best_scope = current_best_scope
    best_rows = current_best_rows

    # Try backtracking with increasing scope limits
    # Start from a scope that might be achievable
    min_possible_scope = k  # trivial lower bound
    # Better lower bound: we need at least n_rows * num_diffs_per_row distinct positive differences
    # Each difference is between 1 and scope, so scope >= n_rows * num_diffs_per_row
    # Wait, that's not right either. The differences can repeat across rows as long as
    # within a row they're distinct, but across rows they must all be distinct.
    # So total distinct differences needed = n_rows * num_diffs_per_row
    # Each difference is in [1, scope], so we need scope >= n_rows * num_diffs_per_row
    # That's the counting bound!
    lower_bound_scope = n_rows * num_diffs_per_row
    
    # Try scopes from lower_bound_scope up to current_best_scope - 1
    # But this could be a huge range. Let's be smart about it.
    
    # Actually, for (7,5): lower_bound = 7 * 15 = 105, current best is 112
    # So we only need to try 105, 106, 107, 108, 109, 110, 111
    
    # Use backtracking with MRV heuristic to try to find solutions with lower scope
    for target_scope in range(lower_bound_scope, current_best_scope):
        if time.time() > deadline - 5.0:
            break
        result = _backtracking_solve(n_rows, k, target_scope, deadline - 5.0)
        if result is not None:
            best_scope = target_scope
            best_rows = result
            # Found a solution with this scope, try to go lower
            # But since we're going up from lower bound, we can break
            # Actually we want the smallest scope, so break immediately
            break

    # If we found a better solution, try to polish it
    if best_scope < current_best_scope and time.time() < deadline - 10.0:
        polished_rows, polished_scope = _polish(best_rows, k, deadline - 5.0)
        if polished_scope < best_scope:
            best_scope = polished_scope
            best_rows = polished_rows

    return best_rows


def _simple_greedy(n_rows, k):
    """Deterministic smallest-first greedy."""
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
                used.update(new_diffs)
        rows.append(row)
    return rows


def _backtracking_solve(n_rows, k, max_scope, deadline):
    """
    Backtracking search to find n_rows rows of k+1 integers starting at 0,
    all within [0, max_scope], with all positive differences distinct across all rows.
    
    Uses MRV (minimum remaining values) heuristic.
    """
    num_diffs_per_row = (k + 1) * k // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Available differences: 1 to max_scope
    available_diffs = set(range(1, max_scope + 1))
    
    if len(available_diffs) < total_diffs_needed:
        return None
    
    rows = []
    used_diffs = set()
    
    def _solve_row(row_idx):
        """Try to construct row row_idx using backtracking."""
        if time.time() > deadline:
            return None
        
        row = [0]
        # We need to choose k more values, all > 0 and <= max_scope
        # Each new value x adds differences {x - a for a in row}
        
        def _backtrack(pos, current_row):
            if time.time() > deadline:
                return None
            
            if pos == k + 1:
                return list(current_row)
            
            # Try candidate values from current_row[-1] + 1 to max_scope
            # Use MRV: try values that use "rarer" differences first?
            # Or just try in order for determinism
            
            # For efficiency, we can be smarter about which values to try
            # A value x is valid if {x - a for a in current_row} are all in available_diffs
            # and not in used_diffs
            
            last = current_row[-1]
            # Try values in increasing order
            for x in range(last + 1, max_scope + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                if not new_diffs.issubset(available_diffs):
                    continue
                
                # This value is valid
                used_diffs.update(new_diffs)
                current_row.append(x)
                
                result = _backtrack(pos + 1, current_row)
                if result is not None:
                    return result
                
                # Backtrack
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return None
        
        return _backtrack(1, row)
    
    # Solve each row
    for i in range(n_rows):
        if time.time() > deadline:
            return None
        row = _solve_row(i)
        if row is None:
            # Need to backtrack to previous rows
            # This is complex; for now, just return None
            return None
        rows.append(row)
    
    # Verify the solution
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return None
                all_diffs.add(d)
    
    return rows


def _polish(rows, k, deadline):
    """
    Local search: try to move any element to a smaller value that keeps validity.
    """
    best = [list(r) for r in rows]
    best_scope = max(max(r) for r in best)

    def _is_valid(rows):
        all_d = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in all_d:
                        return False
                    all_d.add(d)
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to reduce the scope by moving elements in rows with large values
        for row_idx in range(len(best)):
            if time.time() > deadline:
                return best, best_scope
            row = best[row_idx]
            
            # Try moving each element (except the first, which is 0) to a smaller value
            for pos in range(len(row) - 1, 0, -1):
                if time.time() > deadline:
                    return best, best_scope
                old_val = row[pos]
                
                # Try values from row[pos-1]+1 to old_val-1
                for new_val in range(row[pos - 1] + 1, old_val):
                    row[pos] = new_val
                    
                    if _is_valid(best):
                        new_scope = max(max(r) for r in best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                            break
                        else:
                            row[pos] = old_val
                    else:
                        row[pos] = old_val
                
                if improved:
                    break
            
            if improved:
                break

    return best, best_scope
