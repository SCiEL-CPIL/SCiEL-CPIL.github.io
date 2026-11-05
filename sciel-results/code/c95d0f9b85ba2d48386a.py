import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of k+1 marks.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # 1. Baseline: Greedy construction
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    baseline_m = max(max(r) for r in rows)
    
    # If the specific hard case (7,5) is requested, try to optimize
    # The known best is 112. Greedy is likely > 112.
    if n_rows == 7 and k == 5:
        best_sol = rows
        best_m = baseline_m
        
        # Try to find a solution with M < best_m
        # We know M=112 is possible.
        # We will search M from 105 up to best_m - 1.
        
        for M in range(105, best_m):
            if time.time() > deadline - 15:
                break
                
            # Generate all valid Golomb rulers (rows) with max entry <= M
            # Row length is k+1 = 6
            rulers = _generate_golomb_rulers(k, M)
            if not rulers:
                continue
            
            # Sort rulers by max entry (already generated in increasing order? 
            # My generator should yield them. Let's ensure they are sorted or just use as is)
            
            # Convert to bitmasks for fast intersection check
            # Diff d maps to bit (d-1)
            ruler_bitmasks = []
            for r in rulers:
                mask = 0
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        mask |= (1 << (d-1))
                ruler_bitmasks.append((max(r), mask, r))
            
            # Backtracking search for n_rows disjoint rulers
            sol = _find_disjoint_set(ruler_bitmasks, n_rows, time.time(), deadline)
            
            if sol:
                best_sol = [list(r) for r in sol]
                best_m = M
                break # Found the smallest M, no need to check larger M for "smallest"
                
        return best_sol

    return rows

def _generate_golomb_rulers(k, M):
    """
    Generates all Golomb rulers of length k+1 (k intervals) with max entry <= M.
    Returns a list of tuples (mark1, ..., mark_{k+1}).
    """
    rulers = []
    # Recursive generation
    def backtrack(current_row, current_diffs, last_mark):
        if len(current_row) == k + 1:
            if current_row[-1] <= M:
                rulers.append(tuple(current_row))
            return
        
        # Pruning: even if we add the smallest possible next marks, 
        # can we reach k+1 marks within M?
        # Min next mark is last_mark + 1.
        # We need (k+1 - len(current_row)) more marks.
        # The smallest possible max mark is last_mark + (k+1 - len(current_row))
        if last_mark + (k + 1 - len(current_row)) > M:
            return

        # Try next mark x
        # x must be > last_mark
        # x <= M
        # Also, the differences x - a_i must be distinct and not in current_diffs
        # And distinct among themselves.
        
        start_x = last_mark + 1
        # Heuristic: don't go too high? 
        # x can be up to M.
        # But we can prune if x is too large such that we can't fit remaining marks?
        # Actually, we just iterate x.
        
        for x in range(start_x, M + 1):
            # Check differences
            new_diffs = []
            valid = True
            for a in current_row:
                d = x - a
                if d in current_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                current_row.append(x)
                current_diffs.update(new_diffs)
                backtrack(current_row, current_diffs, x)
                current_row.pop()
                current_diffs.difference_update(new_diffs)

    backtrack([0], set(), 0)
    return rulers

def _find_disjoint_set(ruler_bitmasks, n_rows, start_time, deadline):
    """
    Finds n_rows rulers with disjoint difference bitmasks.
    ruler_bitmasks: list of (max_entry, mask, row_tuple)
    """
    # Sort by max_entry to prefer smaller rulers? 
    # Or maybe sort by mask density?
    # Let's sort by max_entry.
    ruler_bitmasks.sort(key=lambda x: x[0])
    
    # Optimization: If we have many rulers, we might want to limit the search space
    # or use a heuristic.
    
    # For 7 rows, depth is 7.
    # Branching factor is len(ruler_bitmasks).
    
    # Use a recursive search with time limit
    def search(idx, used_mask, chosen):
        if time.time() > deadline:
            return None
        
        if len(chosen) == n_rows:
            return chosen
        
        # Heuristic: if not enough rulers left, fail
        if len(ruler_bitmasks) - idx < n_rows - len(chosen):
            return None
            
        for i in range(idx, len(ruler_bitmasks)):
            max_e, mask, row = ruler_bitmasks[i]
            if mask & used_mask == 0:
                res = search(i + 1, used_mask | mask, chosen + [row])
                if res:
                    return res
        return None

    return search(0, 0, [])
