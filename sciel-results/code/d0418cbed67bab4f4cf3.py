import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    time_limit = 145.0
    
    # 1. Greedy baseline
    baseline_rows, baseline_max = _greedy_construct(n_rows, k)
    best_rows = baseline_rows
    best_max = baseline_max
    
    # 2. Search for better M
    # We know M = baseline_max is valid.
    # We try M = baseline_max - 1, -2, ...
    # If a search for M fails (proven unsolvable), we stop.
    # If a search times out, we stop (cannot prove better).
    
    for M in range(baseline_max - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Calculate remaining time
        remaining_time = time_limit - (time.time() - start_time)
        
        # We give the search a fraction of the remaining time
        # As M decreases, the problem becomes harder (more constrained), 
        # but the search space for valid solutions is smaller.
        # Actually, unsolvability proofs can be hard.
        # Let's just pass the remaining time.
        
        result = _backtracking_search(n_rows, k, M, remaining_time)
        
        if result is not None:
            best_rows = result
            best_max = M
            # Continue to try M-1
        else:
            # Search failed (either unsolvable or timeout)
            # If unsolvable, lower M are also unsolvable.
            # If timeout, we don't know. 
            # To be safe, we break.
            break
            
    return best_rows

def _greedy_construct(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            conflict = False
            new_diffs_bits = 0
            for a in row:
                d = x - a
                bit = 1 << d
                if (used & bit) or (new_diffs_bits & bit):
                    conflict = True
                    break
                new_diffs_bits |= bit
            if not conflict:
                row.append(x)
                used |= new_diffs_bits
                x += 1
            else:
                x += 1
        rows.append(row)
    max_val = max(max(r) for r in rows)
    return rows, max_val

def _backtracking_search(n_rows, k, M, time_budget):
    start_time = time.time()
    rows = []
    
    def dfs(row_idx, current_row, last_val, used_bits, completed_lasts):
        if time.time() - start_time > time_budget:
            return False
            
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            if completed_lasts and current_row[-1] < completed_lasts[-1]:
                return False
            
            new_lasts = completed_lasts + [current_row[-1]]
            rows.append(current_row)
            if dfs(row_idx + 1, [0], 0, used_bits, new_lasts):
                return True
            rows.pop()
            return False
        else:
            for x in range(last_val + 1, M + 1):
                new_diffs_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    bit = 1 << d
                    if (used_bits & bit) or (new_diffs_bits & bit):
                        conflict = True
                        break
                    new_diffs_bits |= bit
                
                if conflict:
                    continue
                
                current_row.append(x)
                if dfs(row_idx, current_row, x, used_bits | new_diffs_bits, completed_lasts):
                    return True
                current_row.pop()
            return False

    if dfs(0, [0], 0, 0, []):
        return rows
    return None
# EVOLVE-BLOCK-END
