import time

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Uses backtracking search with degree-based heuristic to minimize largest entry.
    """
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute total differences needed
    # Each row has k+1 elements, so C(k+1,2) = k*(k+1)/2 differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Binary search on scope
    lo = 0
    hi = 200  # Upper bound for binary search
    
    def can_construct(scope):
        """Try to construct with max entry <= scope using backtracking with heuristics."""
        if time.time() - start_time > time_limit:
            return None
        
        used_diffs = set()
        rows = []
        
        # Backtracking state
        def backtrack(row_idx, row, current_max):
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                return True
            
            # If we've completed this row, move to next
            if len(row) == k + 1:
                rows.append(row[:])
                return backtrack(row_idx + 1, [0], current_max)
            
            last_val = row[-1]
            # Find candidates for next element
            candidates = []
            
            # We need to pick x > last_val, x <= scope
            # And all new differences x - a for a in row must not be in used_diffs
            for x in range(last_val + 1, scope + 1):
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Heuristic: count how many future difference slots remain
                # A candidate is "good" if it uses differences that are "rare" or "large"
                # and leaves many small differences available for future rows
                # Degree-based: count how many of the new_diffs are "critical" (small)
                
                # Count small differences (these are most constrained)
                small_diff_count = 0
                for d in new_diffs:
                    if d <= 15:
                        small_diff_count += 1
                
                # Also consider: how many total differences will be used after this choice
                # vs how many are available
                # Simpler: prefer candidates that use larger differences (less constraining)
                
                # Score: lower is better
                # Primary: fewer small differences
                # Secondary: smaller x (to keep scope tight)
                score = (small_diff_count, x)
                candidates.append((score, x, new_diffs))
            
            if not candidates:
                return False
            
            # Sort by score
            candidates.sort(key=lambda item: item[0])
            
            # Try candidates in order, with some randomization for diversity
            # Use backtracking: try each candidate, if it leads to dead end, try next
            
            # For efficiency, limit the number of candidates to try at each level
            max_candidates_to_try = min(20, len(candidates))
            
            for i in range(max_candidates_to_try):
                score, x, new_diffs = candidates[i]
                
                # Make choice
                row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, row, max(current_max, x)):
                    return True
                
                # Undo choice
                row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
            
            return False
        
        # Start backtracking
        if backtrack(0, [0], 0):
            return rows
        return None
    
    # Binary search for minimum scope
    best_result = None
    while lo <= hi and time.time() - start_time < time_limit:
        mid = (lo + hi) // 2
        result = can_construct(mid)
        if result is not None:
            best_result = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search didn't find a solution, try a higher scope
    if best_result is None:
        for scope in range(150, 250):
            if time.time() - start_time > time_limit:
                break
            result = can_construct(scope)
            if result is not None:
                best_result = result
                break
    
    # Fallback: simple greedy
    if best_result is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows
    
    return best_result
