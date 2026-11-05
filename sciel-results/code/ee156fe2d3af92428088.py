import random
import time
import math
import sys

def _row_diffs_list(row):
    """Return list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _row_mask(row, max_diff):
    """Return bitmask of differences used by this row, where bit d represents diff d."""
    mask = 0
    for d in _row_diffs_list(row):
        if d <= max_diff:
            mask |= (1 << d)
    return mask

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(r[-1] for r in rows)

def _is_valid(rows, k):
    """Check if rows form a valid difference triangle set."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    # Check strictly increasing
    for row in rows:
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
    return True

def _column_wise_construct(n_rows, k, max_scope, rng):
    """
    Column-wise list scheduling: for each column position (1..k), assign values
    to all rows in that column, respecting global difference constraints.
    """
    # rows[r][c] = value at row r, column c
    rows = [[0] for _ in range(n_rows)]
    used_diff_mask = 0  # bitmask of all used differences
    
    max_diff = max_scope
    
    for col in range(1, k + 1):
        # For this column, we need to assign values to each row
        # such that:
        # 1. Value > previous value in same row
        # 2. All new differences (with previous columns) don't conflict
        # 3. All new differences within this column assignment are distinct
        # 4. Value <= max_scope
        
        # Collect all partial assignments and their constraints
        # For each row, the candidate range is (prev_val+1, max_scope]
        
        # Use a greedy approach with backtracking
        # Process rows in order of most constrained first (smallest available range)
        
        row_order = list(range(n_rows))
        
        # Sort by how constrained they are (fewer options first)
        # A row with a larger previous value has fewer options
        row_order.sort(key=lambda r: rows[r][-1], reverse=True)
        
        assignments = {}  # row_idx -> value
        local_used_diffs = set()  # differences introduced in this column assignment
        
        # Backtracking search for this column
        def assign_col(pos_in_order):
            if pos_in_order == n_rows:
                return True
            
            r = row_order[pos_in_order]
            prev_val = rows[r][-1]
            
            # Try candidate values
            # Heuristic: prefer smaller values to keep scope tight,
            # but must avoid conflicts
            
            candidates = list(range(prev_val + 1, max_scope + 1))
            
            # Shuffle for diversity
            rng.shuffle(candidates)
            
            # Sort by a heuristic: prefer values that create larger differences
            # (to leave room for future columns)
            # But also prefer smaller values to minimize scope
            # Balance: score = -value + sum(1/d for new diffs)
            
            scored_candidates = []
            for x in candidates:
                new_diffs = set()
                for c in range(col):
                    new_diffs.add(x - rows[r][c])
                
                # Check conflicts with global used_diffs
                conflict_global = False
                for d in new_diffs:
                    if d <= max_diff and (used_diff_mask >> d) & 1:
                        conflict_global = True
                        break
                if conflict_global:
                    continue
                
                # Check conflicts with local used diffs (from other rows in this column)
                conflict_local = False
                for d in new_diffs:
                    if d in local_used_diffs:
                        conflict_local = True
                        break
                if conflict_local:
                    continue
                
                # Check that new_diffs are all distinct within themselves
                # They are a set, so they are distinct
                # But we also need to ensure no duplicate with existing row diffs
                # which are already in used_diff_mask
                
                # Heuristic score: prefer smaller x (to keep scope low)
                # but penalize small differences
                score = x * 0.1 + sum(1.0 / d for d in new_diffs) * 0.5
                
                scored_candidates.append((score, x, new_diffs))
            
            if not scored_candidates:
                return False
            
            # Sort by score (ascending)
            scored_candidates.sort(key=lambda item: item[0])
            
            # Try top candidates
            for score, x, new_diffs in scored_candidates[:20]:  # Limit to top 20 for speed
                assignments[r] = x
                local_used_diffs.update(new_diffs)
                
                # Update rows temporarily
                rows[r].append(x)
                
                if assign_col(pos_in_order + 1):
                    return True
                
                # Backtrack
                rows[r].pop()
                local_used_diffs.difference_update(new_diffs)
                del assignments[r]
            
            return False
        
        if not assign_col(0):
            return None
        
        # Commit assignments
        for r, x in assignments.items():
            # x is already appended to rows[r]
            pass
        
        # Update global used_diff_mask
        for r in range(n_rows):
            x = rows[r][-1]
            for c in range(col):
                d = x - rows[r][c]
                if d <= max_diff:
                    used_diff_mask |= (1 << d)
    
    return rows

def _backtrack_row_with_mask(row, pos, k, max_scope, used_mask, rng, max_depth=1000):
    """
    Backtracking search for a single row, given current used_mask.
    Returns the completed row or None.
    """
    if pos == k + 1:
        return row[:]
    
    last_val = row[-1]
    candidates = list(range(last_val + 1, max_scope + 1))
    rng.shuffle(candidates)
    
    # Score candidates
    scored = []
    for x in candidates:
        new_diffs = set(x - a for a in row)
        
        # Check global conflicts
        conflict = False
        for d in new_diffs:
            if d <= max_scope and (used_mask >> d) & 1:
                conflict = True
                break
        if conflict:
            continue
        
        # Check local conflicts (within current row)
        # new_diffs is a set, so internal duplicates are handled
        # But we need to check against existing row diffs
        existing_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                existing_diffs.add(row[j] - row[i])
        
        if new_diffs & existing_diffs:
            continue
        
        score = x * 0.1 + sum(1.0 / d for d in new_diffs) * 0.5
        scored.append((score, x, new_diffs))
    
    if not scored:
        return None
    
    scored.sort(key=lambda item: item[0])
    
    for score, x, new_diffs in scored[:15]:
        new_mask = used_mask
        for d in new_diffs:
            if d <= max_scope:
                new_mask |= (1 << d)
        
        result = _backtrack_row_with_mask(row + [x], pos + 1, k, max_scope, new_mask, rng)
        if result is not None:
            return result
    
    return None

def _precompute_row_catalog(k, max_scope, num_rows, rng):
    """
    Precompute a catalog of valid rows (each row of k+1 increasing ints starting at 0,
    max entry <= max_scope) that have no internal duplicate differences.
    Returns list of (row, mask) where mask is the bitmask of differences used.
    """
    catalog = []
    
    # Generate random valid rows
    attempts = 0
    max_attempts = 50000
    while len(catalog) < num_rows * 10 and attempts < max_attempts:
        attempts += 1
        
        # Random row generation
        row = [0]
        for pos in range(1, k + 1):
            min_val = row[-1] + 1
            max_val = max_scope
            if min_val > max_val:
                break
            x = rng.randint(min_val, max_val)
            row.append(x)
        
        if len(row) != k + 1:
            continue
        
        # Check internal validity
        diffs = []
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.append(d)
            if not valid:
                break
        
        if not valid:
            continue
        
        # Create mask
        mask = 0
        for d in diffs:
            if d <= max_scope:
                mask |= (1 << d)
        
        catalog.append((row, mask))
    
    return catalog

def _row_pair_swap_search(rows, k, max_scope, used_mask, rng, num_iterations=100):
    """
    Try to improve by swapping elements between pairs of rows.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    for it in range(num_iterations):
        # Pick two random rows
        r1, r2 = rng.sample(range(len(rows)), 2)
        
        # Try swapping a random element between them
        pos1 = rng.randint(1, k)  # Not the first element (must be 0)
        pos2 = rng.randint(1, k)
        
        old_v1 = best_rows[r1][pos1]
        old_v2 = best_rows[r2][pos2]
        
        # Try swap
        best_rows[r1][pos1] = old_v2
        best_rows[r2][pos2] = old_v1
        
        # Check validity
        if not _is_valid(best_rows, k):
            # Revert
            best_rows[r1][pos1] = old_v1
            best_rows[r2][pos2] = old_v2
            continue
        
        new_scope = _scope(best_rows)
        if new_scope < best_scope:
            best_scope = new_scope
        else:
            # Revert if no improvement
            best_rows[r1][pos1] = old_v1
            best_rows[r2][pos2] = old_v2
    
    return best_rows, best_scope

def _element_local_search(rows, k, max_scope, rng, num_iterations=200):
    """
    Local search: try changing individual elements to reduce scope.
    """
    best_rows = [r[:] for r in rows]
    best_scope = _scope(best_rows)
    
    for it in range(num_iterations):
        # Pick a random row and position
        r = rng.randint(0, len(rows) - 1)
        pos = rng.randint(1, k)  # Not the first element
        
        old_val = best_rows[r][pos]
        
        # Try nearby values
        for delta in [-2, -1, 1, 2]:
            new_val = old_val + delta
            if new_val <= 0 or new_val > max_scope:
                continue
            # Check strictly increasing
            if pos > 0 and new_val <= best_rows[r][pos - 1]:
                continue
            if pos < k and new_val >= best_rows[r][pos + 1]:
                continue
            
            best_rows[r][pos] = new_val
            
            if _is_valid(best_rows, k):
                new_scope = _scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                else:
                    best_rows[r][pos] = old_val
                break
            else:
                best_rows[r][pos] = old_val
    
    return best_rows, best_scope

def _construct(n_rows, k):
    """
    Main construction function using column-wise scheduling and local search.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_result = None
    best_scope = float('inf')
    
    # Start with a reasonable bound and tighten
    for bound in range(106, 113):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple restarts for this bound
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Strategy 1: Column-wise construction
            rows = _column_wise_construct(n_rows, k, bound, rng)
            if rows is not None and _is_valid(rows, k):
                scope = _scope(rows)
                if scope <= bound and scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows]
                    # Local search to improve
                    improved_rows, improved_scope = _element_local_search(rows, k, bound, rng, 100)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_result = [r[:] for r in improved_rows]
                    if best_scope <= 105:
                        return best_result
            
            # Strategy 2: Row-by-row backtracking
            used_mask = 0
            rows2 = []
            success = True
            for row_idx in range(n_rows):
                row = _backtrack_row_with_mask([0], 1, k, bound, used_mask, rng)
                if row is None:
                    success = False
                    break
                rows2.append(row)
                # Update mask
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d <= bound:
                            used_mask |= (1 << d)
            
            if success and _is_valid(rows2, k):
                scope = _scope(rows2)
                if scope <= bound and scope < best_scope:
                    best_scope = scope
                    best_result = [r[:] for r in rows2]
                    # Local search
                    improved_rows, improved_scope = _element_local_search(rows2, k, bound, rng, 100)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_result = [r[:] for r in improved_rows]
                    if best_scope <= 105:
                        return best_result
    
    # If no solution found, use a greedy fallback
    if best_result is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_result = rows
        best_scope = _scope(rows)
    
    return best_result

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    return _construct(n_rows, k)
