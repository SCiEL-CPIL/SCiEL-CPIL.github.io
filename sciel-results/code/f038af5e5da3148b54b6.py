import sys

def construct(n_rows, k):
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    lower_bound = total_diffs_needed
    
    # Try increasing M
    # We cap the search to avoid infinite loops if something goes wrong
    # Greedy provides a safe fallback
    greedy_result = _greedy_construct(n_rows, k)
    greedy_max = max(max(row) for row in greedy_result)
    
    # Search up to greedy_max, as we know a solution exists there
    for M in range(lower_bound, greedy_max + 1):
        result = _dfs_construct(n_rows, k, M)
        if result is not None:
            return result
            
    return greedy_result

def _greedy_construct(n_rows, k):
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
    return rows

def _dfs_construct(n_rows, k, M):
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    rows = []
    node_count = 0
    MAX_NODES = 500000  # Safety limit per M

    # used_diffs: set of all differences used in completed rows
    # current_row: list of elements in the row being built
    # current_row_diffs: set of differences within the current row
    
    def build_row(pos, last_val, current_row, current_row_diffs, used_diffs):
        nonlocal node_count
        node_count += 1
        if node_count > MAX_NODES:
            return False

        if pos == k + 1:
            # Row complete
            # Check if current_row_diffs are disjoint from used_diffs
            # We already checked new diffs against used_diffs at each step?
            # Yes, in the loop below we check `if new_diffs & used_diffs`.
            # And `if new_diffs & current_row_diffs`.
            # So if we reach here, the row is a Golomb ruler and disjoint from previous rows.
            rows.append(list(current_row))
            
            if len(rows) == n_rows:
                return True
            else:
                # Recurse to next row
                new_used = used_diffs | current_row_diffs
                return build_next_row(len(rows), new_used)

        # Pruning:
        # 1. Can we fit the remaining elements of this row in [last_val+1, M]?
        if last_val + (k + 1 - pos) > M:
            return False
            
        # 2. Do we have enough available differences globally?
        # This is a weak prune, but helps.
        # Remaining diffs needed for this row: (k+1-pos)*(pos+1)/2 ? No.
        # Each new element adds `pos` new differences.
        # Total new diffs needed = sum_{i=pos}^{k} i = (k+1)(k+2)/2 - pos(pos+1)/2
        # But we just check if len(used_diffs) + total_diffs_needed_remaining > M
        # This is always true if M is small.
        
        for next_val in range(last_val + 1, M + 1):
            # Calculate new differences
            new_diffs = set()
            for a in current_row:
                d = next_val - a
                new_diffs.add(d)
                
            # Check against global used
            if new_diffs & used_diffs:
                continue
                
            # Check against current row's existing diffs
            if new_diffs & current_row_diffs:
                continue
                
            # Valid, recurse
            current_row.append(next_val)
            current_row_diffs.update(new_diffs)
            
            if build_row(pos + 1, next_val, current_row, current_row_diffs, used_diffs):
                return True
                
            current_row.pop()
            # Remove new_diffs from current_row_diffs
            for d in new_diffs:
                current_row_diffs.discard(d)
                
        return False

    def build_next_row(idx, used_diffs):
        # Start new row with 0
        # Prune: check if we can fit remaining rows
        # Simple check: used_diffs size + remaining rows * diffs_per_row <= M
        if len(used_diffs) + (n_rows - idx) * diffs_per_row > M:
            return False
            
        return build_row(1, 0, [0], set(), used_diffs)

    if build_next_row(0, set()):
        return rows
    return None
