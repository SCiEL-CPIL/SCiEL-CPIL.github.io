def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    min_m = n_rows * k * (k + 1) // 2
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, max(max(r) for r in rows)

    baseline_rows, baseline_scope = greedy_construct()
    best_rows = baseline_rows
    best_scope = baseline_scope
    
    if time.time() > deadline - 10:
        return best_rows

    K = k * (k + 1) // 2

    def can_construct_with_scope(M):
        used = set()
        rows = []
        
        def backtrack(row_idx):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Pruning: Check if enough differences remain
            remaining_rows = n_rows - row_idx
            required_diffs = remaining_rows * K
            available_diffs = M - len(used)
            if available_diffs < required_diffs:
                return False
            
            # Construct row row_idx
            row = [0]
            
            def build_row(current_row):
                if time.time() > deadline:
                    return None
                
                if len(current_row) == k + 1:
                    diffs = set()
                    for i in range(len(current_row)):
                        for j in range(i+1, len(current_row)):
                            diffs.add(current_row[j] - current_row[i])
                    
                    if diffs & used:
                        return None
                    
                    # Additional prune for next rows
                    # If we accept this row, used grows by K
                    # Check if next row is possible
                    if row_idx + 1 < n_rows:
                         next_required = (n_rows - row_idx - 1) * K
                         next_available = M - (len(used) + len(diffs))
                         if next_available < next_required:
                             return None

                    rows.append(current_row)
                    used.update(diffs)
                    
                    if backtrack(row_idx + 1):
                        return current_row
                    
                    used.difference_update(diffs)
                    rows.pop()
                    return None
                
                last = current_row[-1]
                # Try candidates for x
                # We can limit the search space for x
                # x must be such that diffs x-a are not in used
                # Also x <= M
                for x in range(last + 1, M + 1):
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used:
                        continue
                    
                    # Heuristic: Prefer smaller x? Yes, to keep scope small.
                    # But maybe we should try a few larger x if small x fails?
                    # The loop does that naturally by backtracking.
                    
                    res = build_row(current_row + [x])
                    if res is not None:
                        return res
                return None
            
            return build_row([0])
        
        return backtrack(0)

    # Search for minimal M
    for M in range(baseline_scope - 1, min_m - 1, -1):
        if time.time() > deadline - 5:
            break
        
        result = can_construct_with_scope(M)
        if result:
            best_rows = result
            best_scope = M
            # Continue to try smaller M
        else:
            # If M fails, smaller M will fail
            break

    return best_rows
