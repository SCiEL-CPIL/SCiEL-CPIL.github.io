import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # Leave margin for safety
    
    # Calculate lower bound for the maximum entry
    # Total distinct differences required = n_rows * C(k+1, 2)
    total_diffs = n_rows * (k * (k + 1) // 2)
    M_start = total_diffs
    
    # Limit the number of nodes visited in DFS to avoid hanging on infeasible M
    node_limit = 20000000  # 20 million nodes
    
    def dfs(M):
        rows = [[0] for _ in range(n_rows)]
        node_count = [0]
        
        def recurse(row_idx, col_idx, current_row, used_diffs):
            node_count[0] += 1
            if node_count[0] > node_limit:
                return False
            
            if col_idx == k + 1:
                if row_idx == n_rows:
                    return True
                # Start next row
                # We don't enforce symmetry breaking on the rows themselves to keep the code simple,
                # but the bitset pruning is strong enough.
                return recurse(row_idx + 1, 1, [0], used_diffs)
            
            prev_val = current_row[col_idx - 1]
            remaining_slots = k + 1 - col_idx - 1
            
            # Try values for the next element
            # We can iterate from smallest to largest.
            for v in range(prev_val + 1, M + 1):
                # Pruning: Check if enough room for remaining elements
                if M - v < remaining_slots:
                    break
                
                # Check for difference conflicts
                new_diffs = 0
                conflict = False
                for u in current_row:
                    d = v - u
                    bit = 1 << d
                    # Check if this difference is already used globally
                    if used_diffs & bit:
                        conflict = True
                        break
                    # Check if this difference is already used in the current row (new_diffs)
                    if new_diffs & bit:
                        conflict = True
                        break
                    new_diffs |= bit
                
                if conflict:
                    continue
                
                # Recurse
                current_row.append(v)
                if recurse(row_idx, col_idx + 1, current_row, used_diffs | new_diffs):
                    return True
                current_row.pop()
            
            return False
        
        if recurse(0, 1, [0], 0):
            return rows
        return None

    # Try to find a solution for increasing M
    for M in range(M_start, 250):  # Upper bound 250 is safe
        if time.time() - start_time > time_limit:
            break
        result = dfs(M)
        if result is not None:
            return result
            
    # Fallback: Greedy construction (same as seed)
    # This ensures we always return a valid result
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
