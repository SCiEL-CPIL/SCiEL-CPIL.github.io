def construct(n_rows, k):
    import time
    start_time = time.time()
    TIME_LIMIT = 115
    
    total_diffs_per_row = k * (k + 1) // 2
    MAX_VAL = 200  # upper bound on scope
    
    # Use a list for used differences for faster lookup
    used = [False] * (MAX_VAL + 1)
    rows = []
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs
    
    def find_row(used_arr, max_scope, min_first):
        """Find a valid row with max element <= max_scope, first element >= min_first."""
        row = [0]
        
        def dfs(current_row, current_diffs_set):
            if time.time() - start_time > TIME_LIMIT:
                return None
            if len(current_row) == k + 1:
                return current_row
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            # Pruning: need 'remaining' more elements, each > last, <= max_scope
            if max_scope - last < remaining:
                return None
            
            for next_val in range(last + 1, max_scope + 1):
                # Check new differences
                valid = True
                new_diffs = []
                for a in current_row:
                    d = next_val - a
                    if d > MAX_VAL or used_arr[d] or d in current_diffs_set:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Pruning: check if remaining elements can fit
                if remaining - 1 > 0 and max_scope - next_val < remaining - 1:
                    break
                
                current_diffs_set.add(next_val - current_row[-1])  # Just for tracking, not all
                # Actually, let me pass a frozenset or use a different approach
                result = dfs(current_row + [next_val], current_diffs_set | set(new_diffs))
                current_diffs_set.discard(next_val - current_row[-1])
                
                if result is not None:
                    return result
            return None
        
        for first in range(min_first, max_scope + 1):
            if first > MAX_VAL or used_arr[first]:
                continue
            result = dfs([0, first], {first})
            if result is not None:
                return result
        return None
    
    def backtrack(row_idx, current_rows, used_arr, max_scope_so_far):
        if time.time() - start_time > TIME_LIMIT:
            return None, max_scope_so_far
        if row_idx == n_rows:
            return current_rows, max_scope_so_far
        
        if row_idx == 0:
            min_first = 1
        else:
            min_first = current_rows[row_idx - 1][1] + 1
        
        for max_scope in range(min_first, MAX_VAL + 1):
            if time.time() - start_time > TIME_LIMIT:
                break
            row = find_row(used_arr, max_scope, min_first)
            if row is not None:
                # Verify all diffs are unused
                new_diffs = get_diffs(row)
                all_valid = True
                for d in new_diffs:
                    if d > MAX_VAL or used_arr[d]:
                        all_valid = False
                        break
                if all_valid:
                    # Mark used
                    for d in new_diffs:
                        used_arr[d] = True
                    
                    new_rows = current_rows + [row]
                    result, scope = backtrack(row_idx + 1, new_rows, used_arr, max(max_scope_so_far, max_scope))
                    
                    # Unmark
                    for d in new_diffs:
                        used_arr[d] = False
                    
                    if result is not None:
                        return result, scope
        return None, max_scope_so_far
    
    result, scope = backtrack(0, [], used, 0)
    if result is None:
        # Fallback to greedy
        used_set = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used_set:
                    row.append(x)
                    used_set |= new
            rows.append(row)
        return rows
    return result
